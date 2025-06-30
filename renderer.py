import trimesh.creation
import trimesh.visual
import pyrender
import trimesh 
import numpy as np
import cv2
import pyrealsense2 as rs 
from PIL import Image
import copy
from video_utils import pyvistaToTrimesh,trimeshToPyvista
import pyvista

#optimize adapter le renderer a la taille de l image

def param_intrinsics():
    '''
    Donne les paramètres intrinsèques de la caméra (RealSense D415) 
    
    '''
    pipeline = rs.pipeline()
    config = rs.config()
    cfg = pipeline.start(config)
    profile=(cfg.get_stream(rs.stream.color,0))
    intr = profile.as_video_stream_profile().get_intrinsics()
    pipeline.stop()
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

def create_renderer(Resolution):
    '''
    Crée le renderer
    Input : la résolution de la camera sous la forme (1280,720) «-- (example)
    '''
    return(pyrender.OffscreenRenderer(viewport_width=Resolution[0],
                               viewport_height=Resolution[1],
                               point_size=1.0))
    

def create_screen(img,resolution):
    '''
    Création de «l'écran»
    Input :image sous forme de tableau numpy
    Output : ecran de fond
    '''
    img = Image.fromarray(img) 
    edge_lengths = np.array([resolution[0],resolution[1],1.0],np.float64)
    scr_transform = np.eye(4)
    screen = trimesh.creation.box(extents=edge_lengths,transform=scr_transform,bounds=None)
    
    return(screen)
    
def create_scene(img,resolution): 
    '''
    Créer la scène avec la bonne lumière, caméra et initialisation du fond 
    
    Input : image de fond (1ere frame de la vidéo), screen (à initialiser)
    Output : scène initialisée
    '''
    
    #Creation de l'écran
    screen = create_screen(img,resolution)
    #Image sous le format PIL
    img = Image.fromarray(img)
    #Creation du mesh pour pyrender
    tm= trimesh.load(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\masque.stl") #marker pour l'origine
    mesh = pyrender.Mesh.from_trimesh(tm)
    
    #Creating lights
    pl = pyrender.PointLight(color=[1.0, 1.0, 1.0],intensity = 0) #candela
    dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.0) #lux
    
    #Creating camera
    oc = pyrender.camera.OrthographicCamera(xmag=resolution[0]/2,ymag=resolution[1]/2,zfar=50000,name='main_camera') #xmag = height/2 ymag = width/2
    
    
    #Creation de la scene
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0],bg_color=[1.0, 0.0, 0.0])
    
    #Creation de la texture du screen
    uv= [ # uv mapping
         [0.0, 0.0],
         [0.0, 1.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [1.0, 1.0],
         [0.0, 0.0],  
         [1.0, 0.0],
    ]
    
    
    
    
    
    text = pyrender.Texture(name = 'image de fond',source = img,source_channels = 'RGBA' )
    
    material = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    roughnessFactor=1.0,
    baseColorTexture=text
    )
    
    #material = trimesh.visual.texture.SimpleMaterial(image=img)
    color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=img)
    screen=trimesh.Trimesh(vertices=screen.vertices, faces=screen.faces,visual=color_visuals,validate=True, process=False)
    screen = pyrender.Mesh.from_trimesh(screen,material=material)
    
    
    
    
    
    
    #Creation des nodes
    pose_mesh = np.eye(4)
    
    pose_camera = np.eye(4)
    pose_camera[:3,:3]=[[-1,0,0],
                       [0,-1,0],
                       [0,0,1]]
    
    z = 616 #ajuste la caméra à la taille de l'image 
    
    pose_camera[:3,3]=[resolution[0]/2,resolution[1]/2,z] # orthographic camera
    pose_pl = np.eye(4)
    pose_pl[:3,3] = [resolution[0]/2,resolution[1]/2,-10]
    pose_screen = np.eye(4)
    pose_screen[:3,3]=[resolution[0]/2,resolution[1]/2,0]
    
    npl = pyrender.Node(light=pl,matrix = pose_pl)
    nl = pyrender.Node(light=dl, matrix=pose_camera )
    nm = pyrender.Node(name='masque', mesh=mesh, matrix=pose_mesh)
    nc = pyrender.Node(camera=oc, matrix=pose_camera)
    n_scr = pyrender.Node(name='screen',mesh=screen,matrix = pose_screen)
    
    #Ajout des nodes
    scene.add_node(npl)
    #scene.add_node(nm)
    scene.add_node(nl)
    scene.add_node(nc)
    scene.add_node(n_scr)
    
    #Ajout Origine
    origine = trimesh.creation.box((20,20,20),np.eye(4))
    origine = pyrender.Mesh.from_trimesh(origine)
    scene.add(origine)
    
    return(scene)

def update_screen(img,scene,resolution): 
    '''
    Update l'image de fond
    (Met à jour la texture appliquée sur le mesh 'screen')
    
    Input : image à appliquer en fond, scène à modifier, mesh 'screen' de la scène
    Output : scène modifiée
    '''
    #image sous la forme PIL
    img = Image.fromarray(img)
    
    uv= [ # uv mapping (pour cette vue)
         [0.0, 0.0],
         [0.0, 1.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [1.0, 1.0],
         [0.0, 0.0],  
         [1.0, 0.0],
    ]
    #Enlever l'ecran precedent
    node_scr = scene.get_nodes(name='screen')
    for i in node_scr:
        screen = i.mesh.primitives[0]
        scene.remove_node(i)
    
    
    
    #Ajouter un nouvel ecran avec la nouvelle texture
    text = pyrender.Texture(name = 'image de fond',source = img,source_channels = 'RGBA' )
    
    
    material = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    roughnessFactor=1.0,
    baseColorTexture=text
    )
    
    color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=img)
    screen2=trimesh.Trimesh(vertices=screen.positions, faces=screen.indices, visual=color_visuals, validate=True, process=False)
    screen2 = pyrender.Mesh.from_trimesh(screen2,material=material)
    pose_mesh = np.eye(4)
    pose_mesh[:3,:3]=[[1,0,0],
                      [0,1,0],
                      [0,0,1]]
    pose_mesh[:3,3]=[resolution[0]/2,resolution[1]/2,0] #aligner les coins de video des prediction et du rendering
    n_scr = pyrender.Node(name='screen',mesh=screen2,matrix = pose_mesh)
    scene.add_node(n_scr)
    
    return(scene)

def update_masque(scene,masque):
    '''
    Update la position du masque pour l'aligner sur le visage dans la vidéo
    a partir du masque (trimesh) crée à partir de la vidéo
    
    Input : scène à modifier , masque à intégrer dans la scène (sous la forme Pyvista)
    Output : scène modifiée (avec le nouveau masque)
    '''
    
    for i in scene.get_nodes(name='masque'): 
        scene.remove_node(i)
    
    #Test si le masque est un objet pyvista et le transforme en trimesh
    if isinstance(masque,pyvista.core.pointset.PolyData):
        masque = pyvistaToTrimesh(masque) #pyrender fonctionne avec trimesh
    masque = pyrender.Mesh.from_trimesh(masque,smooth =False)
    
    #Nouveau masque
    pose_masque = np.eye(4)
    pose_masque[:3,3] = [0,0,0]
    pose_masque[:3,:3]=[[1,0,0],
                        [0,1,0],
                        [0,0,1]] 
    
    n_msq = pyrender.Node(name = 'masque',mesh= masque,matrix=pose_masque)
    scene.add_node(n_msq)
    return(scene)
    


if __name__ == '__main__':
    tm= trimesh.load(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA-V3\nez_cible_colore.obj")
    img = Image.open(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\image_fond.png")
    img = np.asarray(img).copy()
    
    nose_ = pyrender.Mesh.from_trimesh(tm)
    
    resolution = (1920,1080)
    
    mat_nose = np.eye(4)
    mat_nose[:3,3] = [resolution[0]/2,resolution[1]/2,0]
    
    r = create_renderer(resolution)
    scene = create_scene(img,resolution)
    scene.add(nose_,pose = mat_nose)
    pyrender.Viewer(scene)
    color, depth = r.render(scene)
    
    color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
    cv2.imshow('sans masque',color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
        
    
    