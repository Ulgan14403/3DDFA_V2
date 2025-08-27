import trimesh
import pyvista as pv
import numpy as np
import copy
from alignment import compute_pca_transform
import open3d

def pyvistaToTrimesh(pyMesh):
    '''
    Transform a mesh using pyvista data type to a mesh using trimesh data type

    Arguments : pyvista mesh
                
    Return : trimesh mesh
    '''
    pyMesh = pyMesh.extract_surface().triangulate()
    faces_as_array = pyMesh.faces.reshape((pyMesh.n_faces, 4))[:, 1:] 
    tmesh = trimesh.Trimesh(pyMesh.points, faces_as_array) 

    return tmesh

def trimeshToPyvista(triMesh):
    '''
    Transform a mesh using trimesh data type to a mesh using pyvista data type

    Arguments : trimesh mesh9.2
                
    Return : pyvista mesh
    '''
    pymesh = pv.wrap(triMesh)

    return pymesh

def plotMeshes(listOfMeshes): 
    '''
    Displays meshes in the list. Not used in the script

    Arguments : list of mesh to display
    Return : nothing
    '''
    pl = pv.Plotter()
    for mesh in listOfMeshes:
        actor = pl.add_mesh(mesh)
    pl.add_axes()
    pl.show()

def extract_surface(pyMesh):
    pyMesh = pyMesh.extract_surface().triangulate()
    faces_as_array = pyMesh.faces.reshape((pyMesh.n_cells, 4))[:, 1:]
    faces_as_array =  faces_as_array.astype(np.int32)
    return (np.array(pyMesh.points).T, np.array(faces_as_array))

def align_nose_y_axis_PCA(nose_mesh,visu=False):
    '''
    Input : nose mesh
    output : nose mesh aligned with axis at origin
    
    On utilise une  pca pour obtenir le centre du nez ainsi que ses directions principales
    '''
    
    # Copie pour ne pas effectuer les operations sur le mesh d'entrée
    nose_mesh_copy = copy.deepcopy(nose_mesh)
    
    # Creation du nuage de point pour la PCA
    nose_pcd = open3d.geometry.PointCloud()
    nose_pcd.points = open3d.utility.Vector3dVector(np.ascontiguousarray(nose_mesh_copy.vertices.astype(np.float64)))
    
    #PCA
    centroid,rota = compute_pca_transform(nose_pcd)
    
    #Appliquer les transformations au nez
    mat_trans = np.eye(4)
    mat_trans[:3,3] = -centroid.T
    mat_rota = np.eye(4)
    mat_rota[:3,:3] = rota.T 
    nose_mesh_copy = nose_mesh_copy.apply_transform(mat_trans)
    nose_mesh_copy = nose_mesh_copy.apply_transform(mat_rota)
    
    # Transformer de l'axe x vers l'axe y (rotation de 90deg autour de z)
    mat_rota_z = np.eye(4)
    mat_rota_z[:3,:3] = [[0,-1,0],
                         [1,0,0],
                         [0,0,1]]
    nose_mesh_copy = nose_mesh_copy.apply_transform(mat_rota_z)
    
    
    # Visualisation
    if visu == True :
        origine = trimesh.creation.axis(5)
        plotMeshes([origine,nose_mesh_copy,nose_mesh])
    
    return(nose_mesh_copy)


def align_nose_y_axis (nose_mesh,visu=False):
    '''
    input : nose mesh 
    output : nose mesh aligned with y axis
    
    algorithme : on place le nez a l origine puis on fait tourner le nez et on repère où est la plus petite distance des bounds selon x (axe vertical)
    probleme : le nez peut etre a l envers 
    
    '''
    angle = 0
    delta_x = trimesh.bounds.corners(nose_mesh.bounds)[1][0] - trimesh.bounds.corners(nose_mesh.bounds)[1][0]
    plage = [i for i in range (-180,0)]
    nose = copy.deepcopy(nose_mesh) # copie du nez pour ne pas prendre en compte la rotation precedente
    nose_pv = trimeshToPyvista(nose)
    # place le nez a l origine
    origine_translate = [-j for j in nose_pv.center]
    trans_mat = np.eye(4)
    trans_mat[:3,3] = origine_translate
    nose = nose.apply_transform(trans_mat)
    
    nose_pv = trimeshToPyvista(nose)
    nose_final = copy.deepcopy(nose)
    
    for k in plage :
        
        nose_centered = copy.deepcopy(nose)
        #rotate le mesh 
        
        
        rot_mat = trimesh.transformations.rotation_matrix(angle = k, direction = [0,0,1], point = nose_pv.center)
        nose_centered = nose_centered.apply_transform(rot_mat)
        delta = trimesh.bounds.corners(nose_centered.bounds)[1][0] - trimesh.bounds.corners(nose_centered.bounds)[1][0]
        
        #save si la longueur est plus grande
        #if k%10 == 0:
        #    plotMeshes([nose,nose_centered])
        if delta < delta_x :
            
            angle = k
            nose_final = copy.deepcopy()
            delta_x = delta
    
    origine_translate = [-j for j in nose_pv.center]
    trans_mat[:3,3] = origine_translate
    nose_final = nose_final.apply_transform(trans_mat)
    
    if visu == True:
        origine = trimesh.creation.axis(5)
        plotMeshes([origine,nose_final,nose_mesh])
    
    return(nose_final,angle)



def main():
    return('video utils.py')

if __name__ == '__main__':
    nose_mesh = trimesh.load(r"E:\Antoine\OneDrive - ETS\Program_Files\PJ137\Dossier patient\patient014_nez - Copie (2).stl")
    align_nose_y_axis(nose_mesh,visu=True)
    align_nose_y_axis_PCA(nose_mesh,visu=True)
    main()
    