import pyvista as pv
import numpy as np
from video_utils import plotMeshes
from alignment import register_via_correspondences
import open3d as o3d
import copy

#indices des différentes parties du masque prédit par 3ddfav2 
# /!\ Si on effectue une transformation (changer le nez), ces indices ne sont plus valables /!\ 
idx_droite = [i for i in range (16367,22939)]
idx_gauche = [i for i in range (22939,29388)]
idx_bas =    [i for i in range (33661,38365)]
idx_haut =   [i for i in range (29388,33661)]
idx_face =   [i for i in range (0,16367)]
liste_position = ['face','droite','gauche','haut','bas']


def color_indx(mesh,idx):
    colors = np.ones((mesh.n_points,3))
    colors[idx] = [1, 0, 0] 
    mesh.point_data["colors"] = colors
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="colors", rgb=True)  # rgb=True pour afficher les couleurs
    #plotter.show()

def color_tot_idx(mesh):
    colors = np.ones((mesh.n_points,3))
    colors[idx_droite] = [1,0,0] 
    colors[idx_gauche] = [0,1,0] 
    colors[idx_bas]    = [0,0,1] 
    colors[idx_haut]   = [1,0,1] 
    colors[idx_face]   = [1,1,0] 
    mesh.point_data["colors"] = colors
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="colors", rgb=True)  # rgb=True pour afficher les couleurs
    #plotter.show()
    
def fusion_masque(masque):
    
    masque_face,masque_droite,masque_gauche,masque_haut,masque_bas = masque[0],masque[1],masque[2],masque[3],masque[4]
    fusion = np.zeros_like(masque_face.points)
    
    for k in idx_droite :
        fusion[k]= (3*masque_gauche.points[k] + 7 * masque_face.points[k] ) /10
    
    for k in idx_gauche :
        fusion[k]= (3*masque_droite.points[k] + 7 * masque_face.points[k]  ) /10
    
    for k in idx_haut :
        fusion[k]=  (masque_face.points[k] + 4*masque_gauche.points[k] +  4*masque_droite.points[k] )/9
    
    for k in idx_bas :
        fusion[k]=   (masque_face.points[k] +  4*masque_gauche.points[k] + 4* masque_droite.points[k] )/9
        
    for k in idx_face :
        fusion[k]= (2*masque_face.points[k] +6 * masque_gauche.points[k] + 6 * masque_droite.points[k])/14
    
    old_masque = copy.deepcopy(masque_face)
    masque_face.points = fusion
    #nouveau_masque = pv.PolyData(fusion)
    
    #plotMeshes([masque_face])
    return(masque_face)


def recup_masque(points,tri,nom):
    masque = pv.PolyData.from_regular_faces(points,tri)
    masque.save(f'masque_{nom}.stl')
    return(masque)


def registration_pls_masques(liste_masque):
    
    #Registration via correspondance pour aligner les masques
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.ascontiguousarray(liste_masque[0].points.astype(np.float64)))
    
    target_points = [i*10 for i in range (len(target.points)//10)]
    source_points = target_points
    for k in range(1,5):
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(np.ascontiguousarray(liste_masque[k].points.astype(np.float64)))
        result =register_via_correspondences(source,target,target_points,source_points)
        liste_masque[k]=liste_masque[k].transform(result)
    #plotMeshes(liste_masque)
    return(liste_masque)

def main():
    return('masque robuste')

if __name__ == '__main__':
    liste_masque = []
    for k in liste_position:
        liste_masque.append(pv.read(fr"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\masque_{k}.stl"))
    #color_tot_idx(liste_masque[0])
    liste_masque = registration_pls_masques(liste_masque)
    masque_face = fusion_masque(liste_masque)
        
    