import pyrender
import trimesh 
import numpy as np



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
    
    

    
    
    
if __name__ == '__main__':
    
    tm= trimesh.load(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\masque.stl")
    #tm = trimesh.creation.cylinder(5,10)
    print(tm)
    render(tm)