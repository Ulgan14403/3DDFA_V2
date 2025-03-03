import trimesh.visual
import pyrender
import trimesh 
import numpy as np
import cv2
import pyrealsense2 as rs
from PIL import Image
import copy


def param_intrinsics():
    '''
    Donne les paramètres intrinsèques de la caméra (RealSense D415) 
    
    '''
    pipeline = rs.pipeline()
    config = rs.config()
    cfg = pipeline.start(config)
    profile=(cfg.get_stream(rs.stream.color,0))
    intr = profile.as_video_stream_profile().get_intrinsics()
    return(intr)

def Mat_intr(intr):
    '''
    Crée la matrice intrinsèque d'une caméra à partir de ses caractéristiques intrinsèques
    '''
    camMat = np.eye(3)
    camMat[0,0] = intr.fx
    camMat[1,1] = intr.fy
    camMat[0,2] = intr.ppx
    camMat[1,2] = intr.ppy
    return(camMat)

def create_renderer():
    return(pyrender.OffscreenRenderer(viewport_width=1920,
                               viewport_height=1080,
                               point_size=1.0))
    

def create_screen(img):
    '''
    Création de «l'écran»
    '''
    img = Image.fromarray(img)
    edge_lengths = np.array([np.shape(img)[1],np.shape(img)[0],1.0],np.float64)
    scr_transform = np.eye(4)
    screen = trimesh.creation.box(extents=edge_lengths,transform=scr_transform,bounds=None)
    
    return(screen)
    
def create_scene(img,screen): 
    '''
    Créer la scène avec la bonne lumière, caméra et initialisation du fond
    '''
    #Image sous le format PIL
    img = Image.fromarray(img)
    #Creation du mesh pour pyrender
    tm= trimesh.load(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\masque.stl")
    mesh = pyrender.Mesh.from_trimesh(tm)
    
    #Creating lights
    pl = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=200.0)
    dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=20.0)
    
    #Creating camera
    intr = param_intrinsics()
    intr.height = img.height 
    intr.width  = img.width 
    oc = pyrender.camera.IntrinsicsCamera(fx=intr.fx,fy=intr.fy,cx=960,cy=510,zfar=5000,name='main_camera')
    
    #Creation de la scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3],bg_color=[1.0, 0.0, 0.0])
    
    #Creation de la texture du screen
    uv= [
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0],
         [0.0, 0.0],
         [1.0, 0.0],
         [0.0, 0.0],  
         [1.0, 1.0],
    ]
    
    material = trimesh.visual.texture.SimpleMaterial(image=img)
    color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=img, material=material)
    screen=trimesh.Trimesh(vertices=screen.vertices, faces=screen.faces,visual=color_visuals,validate=True, process=False)
    screen = pyrender.Mesh.from_trimesh(screen)
    
    
    #Creation des nodes
    pose_mesh = np.eye(4)
    pose_mesh[:3,3]=[-500,-500,0]
    pose_camera = np.eye(4)
    z = max((616/1920)*img.width,(616/1080)*img.height) #ajuste la caméra à la taille de l'image
    pose_camera[:3,3]=[0,0,z] 
    pose_pl = copy.deepcopy(pose_camera)
    pose_screen = np.eye(4)
    
    
    npl = pyrender.Node(light=pl,matrix = pose_pl)
    nm = pyrender.Node(mesh=mesh, matrix=pose_mesh)
    nl = pyrender.Node(light=dl, matrix=pose_camera )
    nc = pyrender.Node(camera=oc, matrix=pose_camera)
    n_scr = pyrender.Node(name='screen',mesh=screen,matrix = pose_screen)
    
    #Ajout des nodes
    #scene.add_node(npl)
    scene.add_node(nm)
    scene.add_node(nl)
    scene.add_node(nc)
    scene.add_node(n_scr)
    
    return(scene)

def update_screen(img,scene,screen):
    '''
    Update l'image de fond
    '''
    #image sous la forme PIL
    img = Image.fromarray(img)
    uv= [ # uv mapping
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0],
         [0.0, 0.0],
         [1.0, 0.0],
         [0.0, 0.0],  
         [1.0, 1.0],
    ]
    #Enlever l'ecran precedent
    node_scr = scene.get_nodes(name='screen')
    for i in node_scr:
        scene.remove_node(i)
    
    #Ajouter un nouvel ecran avec la nouvelle texture
    material = trimesh.visual.texture.SimpleMaterial(image=img)
    color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=img, material=material)
    screen2=trimesh.Trimesh(vertices=screen.vertices, faces=screen.faces, visual=color_visuals, validate=True, process=False)
    screen2 = pyrender.Mesh.from_trimesh(screen2)
    n_scr = pyrender.Node(name='screen',mesh=screen2,matrix = np.eye(4))
    scene.add_node(n_scr)
    
    return(scene)

def update_masque(scene,masque):
    return(0)

if __name__ == '__main__':
    #tm= trimesh.load(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\masque.stl")
    
    
    img = Image.open(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\image_fond.PNG")
    img = np.array(img)
    img1 = Image.open(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\image_fond_1.PNG")
    img1 = np.array(img1)
    
    screen = create_screen(img)
    r=create_renderer()
    
    
    scene = create_scene(img,screen)
    color, depth = r.render(scene)
    color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
    cv2.imshow('fenetre',color)
    cv2.waitKey()
    
    scene = update_screen(img1,scene,screen)
    color, depth = r.render(scene)
    color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
    cv2.imshow('fenetre',color)
    cv2.waitKey()
    r.delete()