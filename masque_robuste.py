import pyvista as pv
import numpy as np

#indices des différentes parties du masque prédit par 3ddfav2 
# /!\ Si on effectue une transformation (changer le nez), ces indices ne sont plus valables /!\ 
idx_droite = [i for i in range (16367,22939)]
idx_gauche = [i for i in range (22939,29388)]
idx_bas =    [i for i in range (33661,38365)]
idx_haut =   [i for i in range (29388,33661)]
idx_face =   [i for i in range (0,16367)]

def color_indx(mesh,idx):
    colors = np.ones((mesh.n_points,3))
    colors[idx] = [1, 0, 0] 
    mesh.point_data["colors"] = colors
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="colors", rgb=True)  # rgb=True pour afficher les couleurs
    plotter.show()

def fusion_masque(masque):
    masque_face,masque_droite,masque_gauche,masque_haut,masque_bas = masque[0],masque[1],masque[2],masque[3],masque[4]
    fusion = np.zeros_like(masque_face)
    #TODO Vérifier que les masques sont bien à position neutre
    
    for k in idx_droite :
        fusion[k]= (0.7*masque_droite[k] + 0.4 * masque_face[k] + 0.2 * masque_bas[k] + 0.2 * masque_haut[k]) /4
    
    for k in idx_gauche :
        fusion[k]= (0.7*masque_gauche[k] + 0.4 * masque_face[k] + 0.2 * masque_bas[k] + 0.2 * masque_haut[k] ) /4
    
    for k in idx_haut :
        fusion[k]= (0.7*masque_haut[k] + 0.4 * masque_face[k] + 0.2 * masque_gauche[k] + 0.2 * masque_droite[k]) /4
    
    for k in idx_bas :
        fusion[k]= (0.7*masque_bas[k] + 0.4 * masque_face[k] + 0.2 * masque_gauche[k] + 0.2 * masque_droite[k])/4
        
    for k in idx_face :
        fusion[k]= (0.7*masque_face[k] + 0.2 * masque_haut[k] + 0.2 * masque_bas[k] + 0.2 * masque_gauche[k] + 0.2 * masque_droite[k])/4
    
    return(fusion)

def main():
    return('masque robuste')

if __name__ == '__main__':
    main()