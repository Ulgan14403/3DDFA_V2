# coding: utf-8

__author__ = 'cleardusk'

import renderer #custom renderer #optimize on ne peut pas ouvrir d'autres fenetres sinon ca crash
import argparse
import imageio
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque
from crop_V import crop

import cv2
import trimesh
from utils.pose import simple_viz_pose,eular2matrix
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
#from utils.render_ctypes import render
from utils.functions import cv_draw_landmark, get_suffix

import pyvista as pv
import open3d as o3d
import cv2
import time
import matplotlib.pyplot as plt
import pyrender
from PIL import Image

from alignment import prepare_dataset,execute_global_registration,execute_fast_global_registration,register_via_correspondences,draw_registration_result,scale_pcd,aligne_boite,custom_draw_geometry,aligne_boite_origine
from alignment import align_and_center_pcds
import video_utils

import numpy as np


from PIL import Image
import copy

import os
#os.environ["PYOPENGL_PLATFORM"] = "pyglet"


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    nose_mesh = trimesh.load(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA-V3\nez_cible_colore.obj")
    
    idx_nez = []
    with open('points_communs.txt','r') as f:
        lines = f.readlines() 
        for line in lines :
            idx_nez.append(int(line))
        f.close()
   
    nose_mesh_ori,angle = video_utils.align_nose_y_axis(nose_mesh)
    nose_mesh=nose_mesh_ori

    
    
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

        frame_bgr = frame#[..., ::-1]  # RGB->BGR

        if i == 0:
            #create scene for renderer
            
            print('initialisation')
            resolution = (np.shape(frame_bgr)[1],np.shape(frame_bgr)[0])
            r=renderer.create_renderer(resolution)
            scene = renderer.create_scene(frame_bgr,resolution)
            
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
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 20200: #### ligne a modifier pour detecter a chaque frame ou effectuer un suivi #### 500000 pour detection  continue, 2020 pour detection unique
                boxes,thresh = crop(frame_bgr,True)
                #print(thresh)
                if thresh[0] <0.7:
                    continue
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            #P = simple_viz_pose(param_lst,ver)
            #print(P)
            
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
            
                if frame_presente == 1 :
                    
                    
                    # Recupération du nez a partir du masque, sous la forme de nuage de point
                    nez_point_cloud = np.ndarray((1,3))
                    for k in idx_nez :
                        nez_point_cloud = np.insert(nez_point_cloud,0,masque.points[k],axis=0)
                    nez_point_cloud = np.delete(nez_point_cloud,-1,axis=0)
                    

                
                    #Plusieurs répétitions pour dégager une médiane
                    for k in range(nombre_de_repetition):
                        #Global Registration
                        
                        
                        #Aligne le nez du patient sur le masque de la prédiction et remplace le nez du masque i.e. moins précis qu'effectuer une 'vraie' 
                        #global registration avec la totalité du masque
                        
                        target = o3d.geometry.PointCloud()
                        target.points = o3d.utility.Vector3dVector(np.ascontiguousarray(nez_point_cloud.astype(np.float64)))
                        
                        source = o3d.geometry.PointCloud()
                        source.points = o3d.utility.Vector3dVector(np.ascontiguousarray(nose_mesh.vertices.astype(np.float64)))
                        
                        source,facteur = scale_pcd(source,target)
                        
                    
                    #recuperer les transformations
                    R_align,centroid_tgt,centroid_src,icp_result = align_and_center_pcds(source,target)
                    
                    nose_mesh_pv = video_utils.trimeshToPyvista(nose_mesh)
                    
                    #créer les matrices de transformation
                    centre_src = np.eye(4)
                    centre_src[:3,3] = -np.asarray(nose_mesh_pv.center).T
                    centre_tgt = np.eye(4)
                    centre_tgt[:3,3] = centroid_tgt.T
                    centre_tgt_inv = np.eye(4)
                    centre_tgt_inv[:3,3] = -centroid_tgt.T
                    rota = np.eye(4)
                    rota[:3,:3] = R_align
                    scale_fact = np.eye(4)
                    for k in range(3):
                        scale_fact[k,k] = facteur*1.1
                    
                    
                    
                    
                    #appliquer les transformation
                    nose_mesh = nose_mesh.apply_transform(centre_src)
                    nose_mesh = nose_mesh.apply_transform(scale_fact  )
                    nose_mesh = nose_mesh.apply_transform(rota)
                    nose_mesh = nose_mesh.apply_transform(centre_tgt)
                    nose_mesh = nose_mesh.apply_transform(icp_result)

                    
                    #Enlever la partie 'nez' du masque
                    masque.remove_points(idx_nez,inplace=True)
                    ver_ave = (masque.points).T

                    triangles = masque.faces
                    triangles = np.reshape(triangles,(int(len(triangles)/4),4))
                    triangles = np.delete(triangles,0,1)
                    tddfa.tri = triangles.astype(np.dtype(int))
                    
                    nose_mesh_pv = video_utils.trimeshToPyvista(nose_mesh)
                    
                    masque_modified = masque + nose_mesh_pv 
                    nose_mesh_pyr = pyrender.Mesh.from_trimesh(nose_mesh)
                    
                    nose_affichage = copy.deepcopy(nose_mesh)
                    
                    #Ajouter le nouveau nez au masque
                    frame_presente = nombre_de_repetition
                
                else :
                    target = o3d.geometry.PointCloud()
                    target.points = o3d.utility.Vector3dVector(np.ascontiguousarray(masque.points.astype(np.float64)))
                    
                    source = o3d.geometry.PointCloud()
                    source.points = o3d.utility.Vector3dVector(np.ascontiguousarray(masque_modified.points.astype(np.float64)))
                    if frame_presente == nombre_de_repetition:
                        
                        #création des listes de points de controles qui vont servir lors du recalage, l'opération est couteuse, 
                        #on ne l'effectue qu'une seule fois
                        
                        diff = len(source.points) - len(target.points) + len(idx_nez)
    
                        target_points = [i for i in range(len(target.points)) if i not in idx_nez ]
                        source_points = [i for i in range(diff,len(source.points))]

                        
                        target_points = [target_points[i*10] for i in range (len(target_points)//10)]
                        source_points = [source_points[i*10] for i in range (len(source_points)//10)]
                        frame_presente = 70
                        

                    voxel_size = 1
                    
                    try:                       
                        
                        result_ransac =register_via_correspondences(source,target,target_points,source_points)
                        #result_ransac = execute_fast_global_registration(source_down, target_down,source_fpfh, target_fpfh,1)
                        
                        #Appliquer la transformation sur le modèle de visage
                        masque_modified = masque_modified.transform(result_ransac)
                        nose_affichage = nose_affichage.transform(result_ransac)
                    except (RuntimeError):
                        print('safeguard')
                        
                        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source,target,1,voxel_size)
                        result_ransac = execute_fast_global_registration(source_down, target_down,source_fpfh, target_fpfh,1)
                        #Appliquer la transformation sur le modèle de visage
                        nose_affichage = nose_affichage.transform(result_ransac.transformation)
                            
                    
                    
                    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source,target,10,1)
                    result_ransac =register_via_correspondences(source,target,target_points,source_points) 
                    
                    
                    
                    
                    #Appliquer la transformation sur le modèle de visage
                    masque_modified = masque_modified.transform(result_ransac)
                    nose_affichage = nose_affichage.apply_transform(result_ransac)
                
                
                #Render le masque
                
                # ver_ave = (masque_modified.points).T
                # triangles = masque_modified.faces
                # triangles = np.reshape(triangles,(int(len(triangles)/4),4))
                # triangles = np.delete(triangles,0,1)
                
                # tddfa.tri = triangles.astype(np.dtype(int))
                
                # ver_ave = (nose_affichage.points).T
                # triangles = nose_affichage.faces
                # triangles = np.reshape(triangles,(int(len(triangles)/4),4))
                # triangles = np.delete(triangles,0,1)
                # tddfa.tri = triangles.astype(np.dtype(int))
                # ver_ave = ver_ave.astype(np.dtype('float32')) #/!\ different du float de numpy qui est float64
                
                
                
                

                
                #img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)#c35
                scene = renderer.update_screen(frame_bgr,scene,resolution)
                scene = renderer.update_masque(scene,nose_affichage)
                img_draw,depth = r.render(scene) # /!\ plante si une autre fenetre de visualisation a ete ouverte dans le code précédent
                tddfa.tri = tri_copy
                img_draw = cv2.cvtColor(img_draw,cv2.COLOR_RGB2BGR)
                
                
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
            #img_draw,depth = r.render(scene) #todo tester v3do pour le rendering
        else:
            raise ValueError(f'Unknown opt {args.opt}')
        
        writer.append_data(img_draw[..., ::-1])  # BGR->RGB

        queue_ver.popleft()
        queue_frame.popleft()
    #fichier.close()
    writer.close()
    r.delete()
    print(f'Dump to {video_wfp}')
    


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
    parser.add_argument('--onnx', action='store_true', default=True)

    args = parser.parse_args()
    main(args)
    
