#Méthodes d'évaluation

import numpy as np
import trimesh as tm

#-------- Jitter : Ecriture code v ; Implémentation vidéo x

def jitter(rot_vec,trans_vec,freq):
    #A executer en fin de programme, une fois que toutes les predictions ont été obtenues et filtrées (rot_vec = totalité des prédictions)
    '''
    rot_vec : toutes les prédictions de rotation, sous la forme w = [ w1 w2 w3 ] (algèbre de Lie)(array numpy)
    trans_vec : toutes les prédictions de translation sous la forme [ x y z]
    freq : fps de la vidéo
    '''
    n = len(rot_vec)
    
    rot_deriv = np.zeros_like(rot_vec)
    trans_deriv = np.zeros_like(trans_vec)
    dt = 1/freq
    
    #Calcul des dérivées secondes
    for k in range (2,n):
        rot_deriv = (rot_vec[k]-2*rot_vec[k-1]+rot_vec[k-2])/ (dt**2)
        trans_deriv = (rot_vec[k]-2*rot_vec[k-1]+rot_vec[k-2])/ (dt**2)
    
    #Calcul de la métrique
    Jr,Jt = 0,0
    for k in range(n):
        Jr += (rot_deriv[k][0])**2 + (rot_deriv[k][1])**2 +(rot_deriv[k][2])**2
        Jt += (trans_deriv[k][0])**2 + (trans_deriv[k][1])**2 +(trans_deriv[k][2])**2
    Jr = (3*n)**(-1) * Jr
    Jt = (3*n)**(-1) * Jt
    
    return(Jr,Jt)

#-------- Lag : Ecriture code v ; Implémentation vidéo x

def lag(trans_pred,trans_gt,rot_pred,rot_gt):
    #A executer en fin de programme, une fois que toutes les predictions ont été obtenues et filtrées (Trans_pred = totalité des prédictions)
    '''
    rot_pred : toute les prédictions de rotations sous la forme w = [ w1 w2 w3 ] (algèbre de Lie)(array numpy)
    rot_gt : tous les ground truth des rotations sous la forme w = [ w1 w2 w3 ]
    trans_pred : toutes les prédictions de translation sous la forme [ x y z]
    trans_gt : toutes les ground truths de translation sous la forme [ x y z]
    
    ex : 
    rot_pred = np.array([[1,2,3],
                [4,5,6]
                ...
                [10,11,12]])
    '''
    
    n = len(trans_pred)
    
    #Calcul de la métrique
    Lr,Lt=0
    for k in range(n):
        Lr += np.abs(rot_pred[k,0]-rot_gt[k,0]) + np.abs(rot_pred[k,1]-rot_gt[k,1]) + np.abs(rot_pred[k,2]-rot_gt[k,2])
        Lt += np.abs(trans_pred[k,0]-trans_gt[k,0]) + np.abs(trans_pred[k,1]-trans_gt[k,1]) + np.abs(trans_pred[k,2]-trans_gt[k,2])
    Lr = Lr*(1/(3*n))
    Lt = Lt*(1/(3*n))
    return(Lr,Lt)


#-------- Calcul fps : Écriture code v ; Implémentation vidéo x

def fps(temps,frame):
    return(frame/temps)


#-------- Robustesse : Écriture code ~ ; Implémentation vidéo x
# Créer une boite englobante pour le nez, à partir des landmarks des yeux et de la bouche, puis vérifier si le nez est bien dans la boîte englobante (ne fonctionne que de face) 

# og = oeil gauche, od = oeil droit
# b = bouche

#Comparer les boites englobantes
def Robustesse(pos_og,pos_od,pos_b,nez_trimesh):
    
    # Création boite englobante visage
    
    bot_left = [pos_og[0],pos_b[1],pos_b[2]]
    top_right = [pos_od[0],pos_od[1],np.abs(pos_og[0]-pos_od[0])]
    
    # Tester si le nez est compris dans la boite
    result = tm.bounds.contains((bot_left,top_right),nez_trimesh.vertices)
    
    #Calcul du taux de points dans la boite #todo verifier que la valeur en z est suffisante 
    #fix on peut pas calculer la profondeur du nez pcq il est bien plus en avant (ajouter une projection du nez et de la boite englobante ?)
    taux = result.sum()/len(result)
    
    return(taux) # max = 1,min = 0
    
    