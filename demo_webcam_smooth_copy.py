# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque
from crop_V import crop

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark

import trimesh
import pyvista as pv
import open3d as o3d
import cv2
import time

import matplotlib.pyplot as plt

from masque_robuste import recup_masque,registration_pls_masques,fusion_masque

from alignment import register_via_correspondences,scale_pcd
from alignment import align_and_center_pcds, compute_pca_transform
import video_utils
import pyrealsense2 as rs
import copy

import renderer
from minilag_filter import MinilagFilter
import minilag_filter


def main(args):
    config_mini_x = {
    'freq' : 30,
    'mincutoff' :1e-3,
    'gamma' : 1e-4,
    'dcutoff' : 1 
    }

    config_mini_y = {
    'freq' : 30,
    'mincutoff' :1e-3,
    'gamma' : 1e-4,
    'dcutoff' : 1 
    }
    config_mini_rot = {
    'freq' : 30,
    'mincutoff' :0.8,
    'gamma' : 0.6,
    'dcutoff' : 1 
    }


    MiniLag_x = MinilagFilter(**config_mini_x)
    MiniLag_y = MinilagFilter(**config_mini_y)
    MiniLag_z = MinilagFilter(**config_mini_y)
    Minilag_rot = minilag_filter.Rotation_Minilag(config_mini_rot)
    pos_x = []
    pos_y = []
    pos_z = []
    pos_xf = []
    pos_yf = []
    pos_zf = []
    compteur =0
    nose_mesh = trimesh.load(args.nez)
    idx_nez = []
    with open('points_communs.txt','r') as f:
        lines = f.readlines() 
        for line in lines :
            idx_nez.append(int(line))
        f.close()
    
    nose_mesh_ori,angle = video_utils.align_nose_y_axis_PCA(nose_mesh)
    nose_mesh=nose_mesh_ori
    
   
    video_wfp = r'E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\examples\results\videos\archives\video_live_smooth.mp4'
    writer = imageio.get_writer(video_wfp)
    
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

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

    # Given a camera
    # before run this line, make sure you have installed `imageio-ffmpeg`
    #reader = imageio.get_reader("<video0>")
    
    #Intel Real Sense
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    # Get the color
    resolution = (640,480) #/!\ Ne pas oublier de changer les fps en fonction de la resolution, en accord avec les capacitées de la caméra /!\
    fps =30
    config.enable_stream(rs.stream.color, resolution[0],resolution[1], rs.format.bgr8, fps)

    # Start streaming
    pipeline.start(config)
    
    
    
    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()

    R = np.array(([0.0,0.0,0.0],
                 [0.0,0.0,0.0],
                 [0.0,0.0,0.0]))
    R= R.astype(np.float32)
    
    n=0
    frame_presente = 1
    nombre_de_repetition = 5
    
    # run
    dense_flag = args.opt in ('2d_dense', '3d')
    pre_ver = None
    i=0
    First=True
    position =6 #todo finir la fonction pour les masques robustes
    liste_position = ['face','droite','gauche','haut','bas']
    liste_indication = ['FACE','-->','<--',r' /\ ',r' \/ '] 
    liste_masque_position = []
    
    
    try:
        while True:           
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            # Convert images to numpy arrays
            frame_bgr = np.asanyarray(color_frame.get_data()) #sous forme numpy
            if i == 0:
                #Création du renderer pour afficher le masque 
                scene = renderer.create_scene(frame_bgr.copy(),resolution)
                r = renderer.create_renderer(resolution)
                
                
                # the first frame, detect face, here we only use the first face, you can change depending on your need
                boxes = crop(frame_bgr)
                if len(boxes)==0:
                    continue
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
                
                i=1
            else:
                if len(boxes)==0:
                    i=0
                    continue
                param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')
                
                roi_box = roi_box_lst[0]
                # todo: add confidence threshold to judge the tracking is failed
                if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 500000000000000000:
                    boxes,thresh = crop(frame_bgr,True)
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

                if args.opt == '2d_sparse':
                    img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
                elif args.opt == '2d_dense':
                    img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
                elif args.opt == '3d':
                    
                    
                    
                    #recuperation du masque
                    tri_copy = tddfa.tri
                    masque = pv.PolyData.from_regular_faces(ver_ave.T,tddfa.tri)
                    
                    if position <= 4:
                        
                        
                        indication = liste_indication[position]
                        
                        #Display l'image pour que l'utilisateur puisse guider le patient
                        img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
                        cv2.putText(img_draw,f'{indication}',(0,475),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,2)
                        cv2.imshow('image', img_draw)
                        queue_ver.popleft()
                        queue_frame.popleft()
                        
                        
                        
                        #Récupère le masque quand l'utilisateur appuie sur entrée
                        k = cv2.waitKey(13)
                        if k ==13:
                            liste_masque_position.append(recup_masque(ver_ave.T,tddfa.tri,liste_position[position]))
                            position+=1
                        
                        if k == ord('q'):
                            break
                        continue
                    if position == 5 :
                        
                        liste_masque = registration_pls_masques(liste_masque_position)
                        masque = fusion_masque(liste_masque)
                        position+=1
                        
                    if frame_presente == 1 :
                        
                        # Recupération du nez a partir du masque, sous la forme de nuage de point
                        nez_point_cloud = np.ndarray((1,3))
                        for k in idx_nez :
                            nez_point_cloud = np.insert(nez_point_cloud,0,masque.points[k],axis=0)
                        nez_point_cloud = np.delete(nez_point_cloud,-1,axis=0)
                    
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
                        #Ajouter le nouveau nez au masque
                        masque_modified = masque + nose_mesh_pv
                        nose_affichage = copy.deepcopy(nose_mesh)
                        nose_affichage2 = copy.deepcopy(nose_affichage)
                        frame_presente = nombre_de_repetition
                        
                    
                    else :
                        
                        
                        target = o3d.geometry.PointCloud()
                        target.points = o3d.utility.Vector3dVector(np.ascontiguousarray(masque.points.astype(np.float64)))
                        
                        source = o3d.geometry.PointCloud()
                        source.points = o3d.utility.Vector3dVector(np.ascontiguousarray(masque_modified.points.astype(np.float64)))
                        
                        if First == True :
                            
                            ''' 
                            création des listes de points de controles qui vont servir lors du recalage, l'opération est couteuse, 
                            on ne l'effectue qu'une seule fois
                            '''
                            diff = len(source.points) - len(target.points) + len(idx_nez)
        
                            target_points = [i for i in range(len(target.points)) if i not in idx_nez ]
                            source_points = [i for i in range(diff,len(source.points))]

                            
                            target_points = [target_points[i*10] for i in range (len(target_points)//10)]
                            source_points = [source_points[i*10] for i in range (len(source_points)//10)]
                            First = False
                
                        
                        #Registration
                        result_ransac = register_via_correspondences(source,target,target_points,source_points)
                        
                        #Appliquer la transformation sur le modèle de visage
                        masque_modified = masque_modified.transform(result_ransac)
                        nose_affichage  = nose_affichage.apply_transform(result_ransac)

                        #Filtrer la position du nez
                        compteur +=1
                        #todo ici
                        nose_affichage_pcd = o3d.geometry.PointCloud()
                        nose_affichage_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(nose_affichage.vertices.astype(np.float64)))
                        centroid, Mat_rot = compute_pca_transform(nose_affichage_pcd)
                    
                   
                    
                        pos_x.append(centroid[0])
                        pos_y.append(centroid[1])
                        pos_z.append(centroid[2])
                        pred=[0,0,0]
                        pred[0] = MiniLag_x(centroid[0]) 
                        pred[1] = MiniLag_y(centroid[1])
                        pred[2] = MiniLag_z(centroid[2])
                        pos_xf.append(pred[0])
                        pos_yf.append(pred[1])
                        pos_zf.append(pred[2])
                        translate_offset  = np.subtract(pred,centroid)
                        
                        #lissage des rotations
                        rot_filtered = Minilag_rot(Mat_rot)
            
                        #soustraction de 2 matrices de rotations
                        Mat_offset =  rot_filtered @ Mat_rot.T
                        
                        #Mat_transfo_offset = trimesh.transformations.transform_around(Mat_transfo_offset,pred)
                        
                        
                        nose_affichage2.vertices = nose_affichage.vertices - centroid #placer a l origine
                        nose_affichage2.vertices = nose_affichage2.vertices @ Mat_offset.T #appliquer la rotation
                        nose_affichage2.vertices = nose_affichage2.vertices + pred  #placer au bon endroit
                    
                    
                    
                    
                    
                    
                    
                    
                    # #Render le masque
                    
                    
                    # ver_ave = (masque_modified.points).T
                    # triangles = masque_modified.faces
                    # triangles = np.reshape(triangles,(int(len(triangles)/4),4))
                    # triangles = np.delete(triangles,0,1)
                    
                    # tddfa.tri = triangles.astype(np.dtype(int))
                    # img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)#c35
                    # tddfa.tri = tri_copy
                    
                    
                    #img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7) #ff0000 ajouter le nez directement ?
                else: 
                    raise ValueError(f'Unknown opt {args.opt}')

                
                
                #Render l'image
                frame_bgr = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
                
                scene = renderer.update_screen(frame_bgr,scene,resolution)
                scene = renderer.update_masque(scene,nose_affichage2)
                img_draw,depth = r.render(scene)
                
                img_draw = cv2.cvtColor(img_draw,cv2.COLOR_RGB2BGR)
                
                cv2.imshow('image', img_draw)
                k = cv2.waitKey(10)
                if (k & 0xff == ord('q')):
                    break
                
                queue_ver.popleft()
                queue_frame.popleft()
                writer.append_data(img_draw[..., ::-1])
                
    finally:
    
        # Stop streaming
        print('help')
        pipeline.stop()
        writer.close()
        print(f'Dump to {video_wfp}')
    fig, axs = plt.subplots(3,sharex=True)
    disp_frame = [i for i in range(compteur)]
    axs[0].plot(disp_frame,pos_x,label = 'raw')
    axs[1].plot(disp_frame,pos_y,label = 'raw')
    axs[2].plot(disp_frame,pos_z,label = 'raw')
    
    axs[0].plot(disp_frame,pos_xf,label= 'filtered')
    axs[1].plot(disp_frame,pos_yf,label= 'filtered')
    axs[2].plot(disp_frame,pos_zf,label= 'filtered')
    plt.legend()
    plt.show()
        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The smooth demo of webcam of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='3d', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-n_pre', default=0, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=0, type=int, help='the next frames of smoothing')
    parser.add_argument('--onnx', action='store_true', default=True)
    parser.add_argument('--nez',type=str,default =r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA-V3\nez_cible_colore.obj" )

    args = parser.parse_args()
    main(args)
