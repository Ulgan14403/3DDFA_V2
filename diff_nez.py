import numpy as np
import pyvista as pv
import pymeshfix
import trimesh

#Load les meshs
masque = pv.read(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\masque - Copie.stl")
nez_masque = pv.read(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\nez_masque.stl")

#Difference
def main():
    commun = []
    
    with open('points_communs.txt','w') as f:

        for idx in range (len(nez_masque.points)):
            a = np.where(masque.points == nez_masque.points[idx])
            if len(a[0])>1:
                for i in range (len(a[0])):
                    if np.allclose(masque.points[a[0][i]],nez_masque.points[idx],atol=1e-2):
                        if a[0][i] not in commun:
                            commun.append(a[0][i])
        print(commun)
        for k in commun:
            f.write(str(k)+'\n')
        f.close()
    return(0)

def viz():
    lst_pts = []
    with open('points_communs.txt','r') as f:
        lines = f.readlines() 
        for line in lines :
            lst_pts.append(int(line))
        f.close()
    masque.remove_points(lst_pts,inplace=True)
    plotter = pv.Plotter()
    plotter.add_mesh(masque)
    plotter.show()


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

def extract_nez(masque):
    lst_pts = []
    with open('points_communs.txt','r') as f:
        lines = f.readlines() 
        for line in lines :
            lst_pts.append(int(line))
        f.close()
    lst_not_pts = [k for k in range(len(masque.points)) ]
    for k in lst_pts:
        lst_not_pts.remove(k)
    masque.remove_points(lst_not_pts,inplace=True)
    masque = pyvistaToTrimesh(masque)
    meshfix = pymeshfix.MeshFix(masque)
    meshfix.repair()
    meshfix.write('nez_masque.ply')


if __name__ == '__main__':
    #print(main())
    #viz()
    
    #Load les meshs

    extract_nez(masque)