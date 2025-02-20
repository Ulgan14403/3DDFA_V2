import trimesh
import pyvista as pv
import numpy as np



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

    
def align_nose_y_axis (nose_mesh):
    '''
    input : nose mesh 
    output : nose mesh aligned with y axis
    
    algorithme : on place le nez a l origine puis on fait tourner le nez et on repère oè est la plus petite distance des bounds selon x (axe vertical)
    probleme : le nez peut etre a l envers 
    
    '''
    angle = 0
    delta_x = nose_mesh.bounds[1]-nose_mesh.bounds[0] # initialisation
    plage = [i for i in range (-180,0)]
    nose = nose_mesh.copy() # copie du nez pour ne pas prendre en compte la rotation precedente
    # place le nez a l origine
    origine_translate = [-j for j in nose.center]
    nose = nose.translate(origine_translate,inplace=True)
    
    nose_final = nose.copy()
    
    for k in plage :
        
        nose_centered = nose.copy()
        #rotate le mesh 
        
        nose_centered = nose_centered.rotate_z(k)
        delta = nose_centered.bounds[1]-nose_centered.bounds[0]
        
        #save si la longueur est plus grande
        #if k%10 == 0:
        #    plotMeshes([nose,nose_centered])
        if delta < delta_x :
            
            angle = k
            nose_final = nose_centered.copy()
            delta_x = delta
    
    origine_translate = [-j for j in nose_final.center]
    nose_final = nose_final.translate(origine_translate,inplace=True)
    
    return(nose_final,angle)


def main():
    return('video utils.py')

if __name__ == '__main__':
    main()