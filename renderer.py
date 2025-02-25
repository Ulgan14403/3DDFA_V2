import pyrender
import trimesh 
import numpy as np


'''
def render(tm):
    m = pyrender.Mesh.from_trimesh(tm)
    
    
    #Per Face / Per vertex coloration #TODO ajuster avec les indices adaptés
    #tm.visual.vertex_colors = np.random.uniform(size=tm.vertices.shape)
    #tm.visual.face_colors = np.random.uniform(size=tm.faces.shape)
    mesh = pyrender.Mesh.from_trimesh(tm)
    print(mesh.primitives)
    
    #Creating lights
    pl = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    #sl = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=2.0,innerConeAngle=0.05, outerConeAngle=0.5)
    dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    
    #Creating camera
    oc = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    
    #Create scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3],bg_color=[1.0, 1.0, 1.0])
    
    #Create node
    pose_mesh = np.eye(4)
    pose_mesh[:3,3]=[2,2,2]
   
    
    pose_camera = np.eye(4)
    pose_camera[:3,3]=[0,0,100]
    print(pose_camera)
    npl = pyrender.Node(light=pl,matrix = pose_camera)
    nm = pyrender.Node(mesh=mesh, matrix=pose_mesh)
    nl = pyrender.Node(light=dl, matrix=pose_camera )
    nc = pyrender.Node(camera=oc, matrix=pose_camera)
    
    #Add_nodes
    scene.add_node(nm)
    scene.add_node(nl)
    scene.add_node(nc)
    
    pyrender.Viewer(scene,face_normals=True)
    return(0)
    
    '''

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import cv2
import imageio
import pyrealsense2 as rs

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
    camMat = np.eye(3)
    camMat[0,0] = intr.fx
    camMat[1,1] = intr.fy
    camMat[0,2] = intr.ppx
    camMat[1,2] = intr.ppy
    return(camMat)
    
def render(img):
    
    
    # Initialiser l'application GUI
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    # Créer un visualiseur
    vis = o3d.visualization.O3DVisualizer("Mon Visualiseur", 1024, 768)
    # Définir le fond avec une couleur par défaut et une image
    o3d.visualization.gui.Application.instance.add_window(vis)
    # Charger l'image en arrière-plan
    bg_image = o3d.io.read_image(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\image_fond.png")
    if bg_image is None:
        print("Erreur : Impossible de lire l'image 'background.jpg'. Assure-toi que le fichier existe et est au bon format.")
    else:

        bg_color = np.array([0.0, 1.0, 1.0, 0.5], dtype=np.float32)  # Format correct (RGBA)
        
        vis.set_background(bg_color, None)
        
        # Créer un maillage (ex: sphère)
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([1, 0, 0])  # Rouge

        # Ajouter la sphère au visualiseur
        vis.add_geometry("sphère", mesh)

        # Centrer la caméra sur l'objet
        vis.reset_camera_to_default()

        # Lancer la fenêtre et s'assurer qu'elle se ferme proprement
    try:
        app.run()
    finally:
        app.quit()  # Assure la fermeture correcte pour éviter le freeze

    pass
    
    intr = param_intrinsics()
    camMat = Mat_intr(intr)
    
    
        
    # Pick a background colour of the rendered image, I set it as black (default is light gray)
    
    
    open3d_image = o3d.geometry.Image(img)
    
    bg = np.asarray([0.0, 0.0, 0.0, 1.0]).astype(np.float32)
    render.set_background(bg,image=None)  # RGBA
    return(0)
    # setup camera intrinsic values
    pinhole = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    # now create your mesh
    mesh = o3d.geometry.TriangleMesh()
    armadillo_mesh = o3d.data.ArmadilloMesh()
    mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
    mesh.paint_uniform_color([1.0, 0.0, 0.0]) # set Red color for mesh 

    # Define a simple unlit Material.
    # (The base color does not replace the mesh's own colors.)
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"

    # add mesh to the scene
    render.add_geometry("MyMeshModel", mesh, mtl)

    # render the scene with respect to the camera
    render.setup_camera(camMat, 0.1, 1.0, 640, 480)
    img_o3d = render.render_to_image()

    # we can now save the rendered image right at this point 
    #o3d.io.write_image("output.png", img_o3d, 9)

    # Optionally, we can convert the image to OpenCV format and play around.
    # For my use case I mapped it onto the original image to check quality of 
    # segmentations and to create masks.
    # (Note: OpenCV expects the color in BGR format, so swap red and blue.)
    img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGR)
    cv2.imshow("Preview window", img_cv2)
    cv2.waitKey()
    
    
    
if __name__ == '__main__':
    
    #tm= trimesh.load(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\masque.stl")
    #tm = trimesh.creation.cylinder(5,10)
    #print(tm)
    img = imageio.imread(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\image_fond.PNG")
    img = np.asarray(img)
    render(img)