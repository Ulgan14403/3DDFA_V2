# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque
from crop_V import crop

from utils.pose import simple_viz_pose,eular2matrix
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
#from utils.render_ctypes import render
from utils.functions import cv_draw_landmark, get_suffix
import trimesh
import pyvista as pv
import open3d as o3d
import cv2
import time
import matplotlib.pyplot as plt


from alignment import prepare_dataset,execute_global_registration,execute_fast_global_registration,register_via_correspondences,draw_registration_result,scale_pcd,aligne_boite,custom_draw_geometry,aligne_boite_origine
from alignment import align_and_center_pcds
import video_utils

#test pour github
#pourquoi ne fonctionnes tu pas ?
def fonction(fonction):
    return fonction
def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    nose_mesh = pv.read(r"E:\Antoine\Downloads\21515_Nose_V1.stl")
    
    # définition des variables
    temps_par_frame_float32 = [0.5913968086242676, 0.6262953281402588, 0.5685248374938965, 0.5904214382171631, 0.5994799137115479, 0.5794503688812256, 0.5574865341186523, 0.5564992427825928, 0.5913956165313721, 0.6402740478515625, 0.668220043182373, 0.5695273876190186, 0.5794277191162109, 0.5804247856140137, 0.5814955234527588, 0.5784523487091064, 0.5827915668487549, 0.5555908679962158, 0.5575094223022461, 0.568528413772583, 0.5435500144958496, 0.5681324005126953, 0.5704741477966309, 0.5705208778381348, 0.5654866695404053, 0.5705225467681885, 0.5754618644714355, 0.542548418045044, 0.578510046005249, 0.556511402130127]
    temps_par_frame_float64 = [0.1117556095123291, 0.13164734840393066, 0.15955138206481934, 0.13463759422302246, 0.13763117790222168, 0.12463831901550293, 0.14261889457702637, 0.15657997131347656, 0.12267208099365234, 0.14361572265625, 0.1326148509979248, 0.14957785606384277, 0.1396045684814453, 0.11666560173034668, 0.14059185981750488, 0.1445913314819336, 0.12963151931762695, 0.1256422996520996, 0.11267638206481934, 0.11865234375, 0.1346120834350586, 0.10569548606872559, 0.1186826229095459, 0.11666202545166016, 0.15059733390808105, 0.13860654830932617, 0.10469746589660645, 0.14159941673278809, 0.11965751647949219, 0.12364077568054199]
    temps_par_frame_voxel_80 = [0.15059685707092285, 0.09574413299560547, 0.03690171241760254, 0.14960002899169922, 0.05385613441467285, 0.09973335266113281, 0.04787111282348633, 0.11469221115112305, 0.07779264450073242, 0.03989362716674805, 0.024933576583862305, 0.03191399574279785, 0.14860177040100098, 0.026927947998046875, 0.06682014465332031, 0.059838056564331055, 0.1296532154083252, 0.10870957374572754, 0.12167549133300781, 0.07081103324890137, 0.02892279624938965, 0.056848764419555664, 0.09275150299072266, 0.025930166244506836, 0.021941423416137695, 0.05385613441467285, 0.1077125072479248, 0.10970735549926758, 0.027924776077270508, 0.050864219665527344]
    temps_par_frame_onnx_gpu = [0.06183457374572754, 0.02590203285217285, 0.07914400100708008, 0.09571480751037598, 0.02692699432373047, 0.045877933502197266, 0.018950223922729492, 0.039870500564575195, 0.04587745666503906, 0.023936033248901367, 0.042885541915893555, 0.019917726516723633, 0.05285763740539551, 0.01795172691345215, 0.024933576583862305, 0.07477164268493652, 0.059839487075805664, 0.08075356483459473, 0.04787039756774902, 0.08275151252746582, 0.04787135124206543, 0.05083584785461426, 0.051859140396118164, 0.017950057983398438, 0.01795172691345215, 0.04485034942626953, 0.03388047218322754, 0.07579708099365234, 0.052858829498291016, 0.05582284927368164]
    masque_bounds = 662.5723876953125, 963.724365234375, 153.64328002929688, 527.9197998046875, 0.0, 252.28634643554688
    masque_centre = (masque_bounds[1]-masque_bounds[0])/2,(masque_bounds[3]-masque_bounds[2])/2,(masque_bounds[5]-masque_bounds[4])/2
    
    threshold_smoothing = 1 # seuil de tolérance pour déterminer que 2 prédictions sont égales
    
    idx_nez = []
    with open('points_communs.txt','r') as f:
        lines = f.readlines() 
        for line in lines :
            idx_nez.append(int(line))
        f.close()
    
    
    
    
    nose_mesh_ori,angle = video_utils.align_nose_y_axis(nose_mesh)
    nose_mesh=nose_mesh_ori
    #nose_mesh = nose_mesh.scale(2,2,2)
    #ver_ave2 = extract_surface(nose_mesh)
    
    
    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a video path
    fn = args.video_fp.split('/')[-1]
    reader = imageio.get_reader(args.video_fp)

    fps = reader.get_meta_data()['fps']
    suffix = get_suffix(args.video_fp)
    video_wfp = f'examples/results/videos/{fn.replace(suffix, "")}_{args.opt}_smooth.mp4'
    writer = imageio.get_writer(video_wfp, fps=fps)

    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()
    pose = [0,0,0]
    R = np.array(([0.0,0.0,0.0],
                 [0.0,0.0,0.0],
                 [0.0,0.0,0.0]))
    R= R.astype(np.float32)
    
    n=0
    frame_presente = 1
    temps_par_frame = []
    nombre_de_repetition = 5
    compteur = 0
    
    
    # run
    dense_flag = args.opt in ('2d_dense', '3d',)
    pre_ver = None
    for i, frame in tqdm(enumerate(reader)):
        if args.start > 0 and i < args.start:
            continue
        if args.end > 0 and i > args.end:
            break

        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i == 0:
            # detect
            #boxes = crop(frame_bgr)
            boxes = face_boxes(frame_bgr)

            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # padding queue
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
            queue_ver.append(ver.copy())

            for _ in range(n_pre):
                queue_frame.append(frame_bgr.copy())
            queue_frame.append(frame_bgr.copy())

        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

            roi_box = roi_box_lst[0]
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 5000000: #### ligne a modifier pour detecter a chaque frame ou effectuer un suivi #### 500000 pour detection  continue, 2020 pour detection unique
                boxes,thresh = crop(frame_bgr,True)
                #print(thresh)
                if thresh[0] <0.7:
                    continue
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            queue_ver.append(ver.copy())
            queue_frame.append(frame_bgr.copy())

        pre_ver = ver  # for tracking

        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)
            tri_copy = tddfa.tri
            if args.opt == '2d_sparse':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
            elif args.opt == '2d_dense':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
            elif args.opt == '3d':
                
                #recuperation du masque
                masque = pv.PolyData.from_regular_faces(ver_ave.T,tddfa.tri)
                
                
                #masque.save('masque.stl')
                #masque = pv.read(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\masque.stl")
            
                
                # nez_point_cloud = np.ndarray((1,3))
                # for k in idx_nez :
                #     nez_point_cloud = np.insert(nez_point_cloud,0,masque.points[k],axis=0)
                # nez_point_cloud = np.delete(nez_point_cloud,-1,axis=0)
                # nez_pcd = o3d.geometry.PointCloud()
                # nez_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(nez_point_cloud.astype(np.float64)))
                # nez_pcd.estimate_normals()
                # distances = nez_pcd.compute_nearest_neighbor_distance()
                # avg_dist = np.mean(distances)
                # radius = 1.5 * avg_dist
                # nez_masque_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(nez_pcd,o3d.utility.DoubleVector([radius, radius * 2]))
                # tri_mesh = trimesh.Trimesh(np.asarray(nez_masque_mesh.vertices), np.asarray(nez_masque_mesh.triangles),vertex_normals=np.asarray(nez_masque_mesh.vertex_normals))
                # py_vista = video_utils.trimeshToPyvista(tri_mesh)
                
                
                
                
                
                
                #Scaling #complètement obsolète 
                ''' 
                bounding_nez_up = ver_ave[1][tddfa.keypoints[27]]
                bounding_nez_down = ver_ave[1][tddfa.keypoints[33]]
                bounding_nez_right = ver_ave[0][tddfa.keypoints[35]]
                bounding_nez_left = ver_ave[0][tddfa.keypoints[31]]
                
                bound_mesh = nose_mesh.bounds
                Scale_nez = max(((abs((bounding_nez_left-bounding_nez_right)/(bound_mesh[1]-bound_mesh[0]))),(abs((bounding_nez_up-bounding_nez_down)/(bound_mesh[3]-bound_mesh[2])))))
                nose_mesh = nose_mesh.scale(Scale_nez*1.2,inplace=False) #[droite/gauche,up/down,depth] 
                '''
                if frame_presente == 1 :
                    #Todo mettre en place une fonction pour récuperer les masques sous différents angles (passer par une boucle avec un compteur de masque)
                    
                    # Recupération du nez a partir du masque, sous la forme de nuage de point
                    nez_point_cloud = np.ndarray((1,3))
                    for k in idx_nez :
                        nez_point_cloud = np.insert(nez_point_cloud,0,masque.points[k],axis=0)
                    nez_point_cloud = np.delete(nez_point_cloud,-1,axis=0)
                    
                   
                    
                    transformation_nez = np.zeros((nombre_de_repetition,4,4))
                    '''
                    rotations = np.zeros((nombre_de_repetition,3,3))
                    translations = np.zeros((nombre_de_repetition,3))
                    '''
                
                    #Plusieurs répétitions pour dégager une médiane
                    for k in range(nombre_de_repetition):
                        #Global Registration
                        
                        '''
                        Aligne le nez du patient sur le masque de la prédiction et remplace le nez du masque i.e. moins précis qu'effectuer une 'vraie' 
                        global registration avec la totalité du masque
                        '''
                        target = o3d.geometry.PointCloud()
                        target.points = o3d.utility.Vector3dVector(np.ascontiguousarray(nez_point_cloud.astype(np.float64)))
                        
                        source = o3d.geometry.PointCloud()
                        source.points = o3d.utility.Vector3dVector(np.ascontiguousarray(nose_mesh.points.astype(np.float64)))
                        
                        source,facteur = scale_pcd(source,target)
                        
                    
                        # source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source,target,frame_presente,1)
                        # result_ransac = execute_global_registration(source_down, target_down,
                        #                             source_fpfh, target_fpfh,
                        #                             1)
                        ''''
                        rotations[k] = result_ransac.transformation[:3,:3]
                        translations[k] = result_ransac.transformation[:3,3]
                        '''
                        
                        #transformation_nez[k] = result_ransac.transformation
                        #draw_registration_result(source,target,result_ransac.transformation)
               
                   
                    # translation = np.reshape(np.median(translations,axis=0),(3,1))
                    # rotation = np.median(rotations,axis=0)
                    # homogene = np.reshape(np.array([0,0,0,1]),(1,4)) #besoin de cette matrice pour avoir une matrice de rotation
                    # result_ransac.transformation = np.append(np.append(rotation,translation,1),homogene,0)
                    # result_ransac.transformation = np.median(transformation_nez,0)
                    
                    # source = source.transform(result_ransac.transformation)
                    # custom_draw_geometry([source,source.get_minimal_oriented_bounding_box(),target,target.get_minimal_oriented_bounding_box()])
                    # nose_mesh = nose_mesh.transform(result_ransac.transformation)
                    
                    # #Creation des nuages de points
                    # target = o3d.geometry.PointCloud()
                    # target.points = o3d.utility.Vector3dVector(np.ascontiguousarray(nez_point_cloud.astype(np.float64)))
                        
                    # source = o3d.geometry.PointCloud()
                    # source.points = o3d.utility.Vector3dVector(np.ascontiguousarray(nose_mesh.points.astype(np.float64)))
                    
                    # #Scale le nez 
                    # source,facteur = scale_pcd(source,target)
                    
                    #recuperer les transformations
                    R_align,centroid_tgt,centroid_src,icp_result = align_and_center_pcds(source,target)
                    
                    #créer les matrices de transformation
                    centre_src = np.eye(4)
                    centre_src[:3,3] = -centroid_src.T
                    centre_tgt = np.eye(4)
                    centre_tgt[:3,3] = centroid_tgt.T
                    centre_tgt_inv = np.eye(4)
                    centre_tgt_inv[:3,3] = -centroid_tgt.T
                    rota = np.eye(4)
                    rota[:3,:3] = R_align
                    scale_fact = np.eye(4)
                    for k in range(3):
                        scale_fact[k,k] = facteur
                    print(centre_tgt)
                    
                    
                    #appliquer les transformation
                    nose_mesh = nose_mesh.transform(centre_src)
                    nose_mesh = nose_mesh.transform(scale_fact)
                    nose_mesh = nose_mesh.transform(rota)
                    nose_mesh = nose_mesh.transform(centre_tgt)
                    nose_mesh = nose_mesh.transform(icp_result)
                    
                    
                    # #   Aligner les deux nez
                    # #Recuperer les rotations
                    # rot_mat,deplacement =aligne_boite_origine(source,target)
                    
                    
                    # #placer a l origine
                    # origine_translate = [-j for j in nose_mesh.center]
                    # nose_mesh = nose_mesh.translate(origine_translate,inplace=True)
                    
                    
                    # #aplliquer les translations les unes apres les autres
                    # rotation = np.eye(4)
                    # for mat in rot_mat:
                    #     rotation[:3,:3] = mat
                    #     nose_mesh = nose_mesh.transform(rotation)
                    
                    # #redeplacer vers le visage
                    # nose_mesh = nose_mesh.translate(-deplacement,inplace=True)
                    
                    
                    
                    
                    
                    '''
                    print(facteur)
                    #Appliquer la transformation sur le modèle de nez
                    nose_mesh = nose_mesh.scale(facteur,inplace=False)
                    nose_mesh = nose_mesh.transform(result_ransac.transformation)
                    '''
                    
                    #Enlever la partie 'nez' du masque
                    masque.remove_points(idx_nez,inplace=True)
                    ver_ave = (masque.points).T

                    triangles = masque.faces
                    triangles = np.reshape(triangles,(int(len(triangles)/4),4))
                    triangles = np.delete(triangles,0,1)
                    tddfa.tri = triangles.astype(np.dtype(int))
                    
                    #Ajouter le nouveau nez au masque
                    masque_modified = masque + nose_mesh
                    
                    
                    
                    
                    
                    frame_presente = nombre_de_repetition
                    
                    
                
                else :
                    par_frame_start = time.time()
                    #Original
                    #Fast Global Registration
                    
                    
                    
                    target = o3d.geometry.PointCloud()
                    target.points = o3d.utility.Vector3dVector(np.ascontiguousarray(masque.points.astype(np.float64)))
                    
                    source = o3d.geometry.PointCloud()
                    source.points = o3d.utility.Vector3dVector(np.ascontiguousarray(masque_modified.points.astype(np.float64)))
                    if frame_presente == nombre_de_repetition:
                        ''' 
                        création des listes de points de controles qui vont servir lors du recalage, l'opération est couteuse, 
                        on ne l'effectue qu'une seule fois
                        '''
                        diff = len(source.points) - len(target.points) + len(idx_nez)
    
                        target_points = [i for i in range(len(target.points)) if i not in idx_nez ]
                        source_points = [i for i in range(diff,len(source.points))]

                        
                        target_points = [target_points[i*10] for i in range (len(target_points)//10)]
                        source_points = [source_points[i*10] for i in range (len(source_points)//10)]
                        frame_presente = 70
                        
                    #Normalisation de la taille
                    '''
                    source_extent = np.max(source.get_max_bound() - source.get_min_bound())
                    target_extent = np.max(target.get_max_bound() - target.get_min_bound())
                    
                    if source_extent <= 0 or target_extent <= 0:
                        raise ValueError("L'un des nuages de points a une échelle invalide.")

                    source.scale(100 / source_extent, center=source.get_center())
                    target.scale(100 / target_extent, center=target.get_center())
                    voxel_size = 100 /source_extent
                    '''
                    voxel_size = 1
                    
                    # down sample les datasets
                    #source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source,target,frame_presente,voxel_size) #optionnel
                    try:                       
                        
                        result_ransac =register_via_correspondences(source,target,target_points,source_points)
                        #result_ransac = execute_fast_global_registration(source_down, target_down,source_fpfh, target_fpfh,1)
                        
                        #Appliquer la transformation sur le modèle de visage
                        masque_modified = masque_modified.transform(result_ransac)
                        
                    except (RuntimeError):
                        print('safeguard')
                        
                        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source,target,1,voxel_size)
                        result_ransac = execute_fast_global_registration(source_down, target_down,source_fpfh, target_fpfh,1)
                        #Appliquer la transformation sur le modèle de visage
                        masque_modified = masque_modified.transform(result_ransac.transformation)
                            
                    par_frame_end = time.time()-par_frame_start
                    temps_par_frame.append(par_frame_end)
            
                    
                    #Threshold
                    '''
                    if 'masque_prec' in locals() and np.allclose(masque.points,masque_prec.points,0,5e-02) :
                        masque = masque_prec
                    
                    else: 
                        #Fast Global Registration
                        target = o3d.geometry.PointCloud()
                        target.points = o3d.utility.Vector3dVector(np.ascontiguousarray(masque.points.astype(np.float64)))
                        
                        source = o3d.geometry.PointCloud()
                        source.points = o3d.utility.Vector3dVector(np.ascontiguousarray(masque_modified.points.astype(np.float64)))
                        
                        # down sample les datasets
                        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source,target,frame_presente)                       
                        result_ransac = execute_fast_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    1)
                        
                        #Appliquer la transformation sur le modèle de nez
                        masque_modified = masque_modified.transform(result_ransac.transformation)
                        nose_mesh = nose_mesh.transform(result_ransac.transformation)
                        par_frame_end = time.time()-par_frame_start
                        temps_par_frame.append(par_frame_end)
                        masque_prec = masque
                        '''
                        
                    #Moyenne 
                '''
                    #le nez ajoute trop de points pour les comparer (difference de taille)
                    if 'masque_prec' in locals(): 
                        print(np.shape(masque.points))
                        print(np.shape(masque_prec.points))
                        masque_modified.points =  np.mean(np.array([masque.points, masque_prec.points]), axis=0)
                    else: 
                        #Fast Global Registration
                        target = o3d.geometry.PointCloud()
                        target.points = o3d.utility.Vector3dVector(np.ascontiguousarray(masque.points.astype(np.float64)))
                        
                        source = o3d.geometry.PointCloud()
                        source.points = o3d.utility.Vector3dVector(np.ascontiguousarray(masque_modified.points.astype(np.float64)))
                        
                        # down sample les datasets
                        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source,target,frame_presente)                       
                        result_ransac = execute_fast_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    1)
                        
                        #Appliquer la transformation sur le modèle de nez
                        masque_modified = masque_modified.transform(result_ransac.transformation)
                        nose_mesh = nose_mesh.transform(result_ransac.transformation)
                        par_frame_end = time.time()-par_frame_start
                        temps_par_frame.append(par_frame_end)
                        masque_prec = masque_modified
                       ''' 

                    
                #Render le masque
                
                ver_ave = (masque_modified.points).T
                triangles = masque_modified.faces
                triangles = np.reshape(triangles,(int(len(triangles)/4),4))
                triangles = np.delete(triangles,0,1)
                
                tddfa.tri = triangles.astype(np.dtype(int))
                img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)#c35
                tddfa.tri = tri_copy
                
                #Render le nez
                '''
                ver_ave2 = extract_surface(nose_mesh)
                img_draw = render(queue_frame[n_pre], [ver_ave2[0]], ver_ave2[1], alpha=0.7)#c35
                img_draw = np.ascontiguousarray(img_draw, dtype=np.uint8)
                '''

                #img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
                #img_draw = render(queue_frame[n_pre], [ver_ave3], tddfa.tri, alpha=0.7)#c35
                
            else:
                raise ValueError(f'Unknown opt {args.opt}')

            writer.append_data(img_draw[:, :, ::-1])  # BGR->RGB

            queue_ver.popleft()
            queue_frame.popleft()
    
    # we will lost the last n_next frames, still padding
    for _ in range(n_next):
        queue_ver.append(ver.copy())
        queue_frame.append(frame_bgr.copy())  # the last frame

        ver_ave = np.mean(queue_ver, axis=0)

        if args.opt == '2d_sparse':            
            img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
        elif args.opt == '2d_dense':
            img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
        elif args.opt == '3d':
            #img_draw = render(queue_frame[n_pre], [ver_ave2[0]], ver_ave2[1], alpha=0.7)#c35
            img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)#c35
        else:
            raise ValueError(f'Unknown opt {args.opt}')

        writer.append_data(img_draw[..., ::-1])  # BGR->RGB

        queue_ver.popleft()
        queue_frame.popleft()
    #fichier.close()
    writer.close()
    print(f'Dump to {video_wfp}')
    
    #Plot le temps de calcul
    '''
    #plt.plot(temps_par_frame_float32,label='float_32')
    #plt.plot(temps_par_frame_float64,label='float_64')
    plt.plot(temps_par_frame_voxel_80,label='voxel80')
    plt.plot(temps_par_frame_onnx_gpu,label='onnx gpu')
    plt.plot(temps_par_frame,label='new')
    plt.legend()
    plt.show()
    print(temps_par_frame)
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The smooth demo of video of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--video_fp', type=str, default = r"E:/Antoine/OneDrive - ETS/Program_Files/videos test/0.Entrée/homme_cote_masque.mp4")
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('-o', '--opt', type=str, default='3d', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-s', '--start', default=-1, type=int, help='the started frames')
    parser.add_argument('-e', '--end', default=-1, type=int, help='the end frame')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
    
