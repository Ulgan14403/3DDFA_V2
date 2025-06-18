# coding: utf-8

__author__ = 'cleardusk'

import os.path as osp
import numpy as np
import cv2
import onnxruntime

from utils.onnx import convert_to_onnx
from utils.io import _load
from utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)
from utils.tddfa_util import _parse_param, similar_transform
from bfm.bfm import BFMModel
from bfm.bfm_onnx import convert_bfm_to_onnx

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
from OneEuroFilter import OneEuroFilter

config_euro = {
    'freq': 30,       # Hz
    'mincutoff': 1,  # Hz
    'beta': 0.0,       
    'dcutoff': 5.0    
    }

one_euro_filter_x = OneEuroFilter(**config_euro)
one_euro_filter_y = OneEuroFilter(**config_euro)
one_euro_filter_z = OneEuroFilter(**config_euro)

class TDDFA_ONNX(object):
    """TDDFA_ONNX: the ONNX version of Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        # torch.set_grad_enabled(False)

        # load onnx version of BFM
        bfm_fp = kvs.get('bfm_fp', make_abs_path('configs/bfm_noneck_v3.pkl'))
        bfm_onnx_fp = bfm_fp.replace('.pkl', '.onnx')
        if not osp.exists(bfm_onnx_fp):
            convert_bfm_to_onnx(
                bfm_onnx_fp,
                shape_dim=kvs.get('shape_dim', 40),
                exp_dim=kvs.get('exp_dim', 10)
            )
            
        self.bfm_session = onnxruntime.InferenceSession(bfm_onnx_fp,None,providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
] )
       
        
        # load for optimization
        bfm = BFMModel(bfm_fp, shape_dim=kvs.get('shape_dim', 40), exp_dim=kvs.get('exp_dim', 10))
        self.tri = bfm.tri
        self.u_base, self.w_shp_base, self.w_exp_base = bfm.u_base, bfm.w_shp_base, bfm.w_exp_base
        self.keypoints = bfm.keypoints
        #print(self.w_shp_base) # reste constant => contient tous les indices de coordonnées (x,y,z) a la suite (pour les keypoints)
        
        list_keypoints = []
        for i in range (len(bfm.keypoints)):
            if i%3 == 0 :
                list_keypoints.append(int(bfm.keypoints[i]/3))
        #print(list_keypoints)            
        self.keypoints = list_keypoints  
        
        
        # config
        self.gpu_mode = kvs.get('gpu_mode', True)
        self.gpu_id = kvs.get('gpu_id', 0)
        self.size = kvs.get('size', 120)

        param_mean_std_fp = kvs.get(
            'param_mean_std_fp', make_abs_path(f'configs/param_mean_std_62d_{self.size}x{self.size}.pkl') 
        )

        onnx_fp = kvs.get('onnx_fp', kvs.get('checkpoint_fp').replace('.pth', '.onnx'))

        # convert to onnx online if not existed
        if onnx_fp is None or not osp.exists(onnx_fp):
            print(f'{onnx_fp} does not exist, try to convert the `.pth` version to `.onnx` online')
            onnx_fp = convert_to_onnx(**kvs)

        self.session = onnxruntime.InferenceSession(onnx_fp, None,providers=["CUDAExecutionProvider"])

        # params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

    def __call__(self, img_ori, objs, **kvs):
        # Crop image, forward to get the param
        param_lst = []
        roi_box_lst = []

        crop_policy = kvs.get('crop_policy', 'box')
        for obj in objs:
            if crop_policy == 'box':
                # by face box    
                roi_box = parse_roi_box_from_bbox(obj)
                #print('roi box (tddfa onnx)')
                #print(roi_box)
            elif crop_policy == 'landmark':
                # by landmarks
                roi_box = parse_roi_box_from_landmark(obj)
                
            else:
                raise ValueError(f'Unknown crop policy {crop_policy}')

            roi_box_lst.append(roi_box)
            
            '''
            img_ori = np.ascontiguousarray(img_ori, dtype = np.uint8)
            
            cv2.drawMarker(img_ori,(int(roi_box[0]),int(roi_box[1])),1)
            cv2.drawMarker(img_ori,(int(roi_box[0]),int(roi_box[1]+10)),(0,255,0))
            
            #cv2.drawMarker(img_ori,(int(roi_box[2]),int(roi_box[3])),2)
            #cv2.drawMarker(img_ori,(int(roi_box[2]),int(roi_box[3]-20)),2)
            
            cv2.imshow('img',img_ori)
            cv2.waitKey(0)
            '''
            
            img = crop_img(img_ori, roi_box)
            img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
            img = (img - 127.5) / 128.

            inp_dct = {'input': img}

            param = self.session.run(None, inp_dct)[0]
            param = param.flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale 
            param_lst.append(param)
        
        return param_lst, roi_box_lst

    def recon_vers(self, param_lst, roi_box_lst, **kvs):
        dense_flag = kvs.get('dense_flag', False)
        size = self.size
        
        ver_lst = []
        for param, roi_box in zip(param_lst, roi_box_lst):
            R, offset, alpha_shp, alpha_exp = _parse_param(param)
           
            '''alpha_shp = [[-1.66670250e+05],
                         [-6.58709062e+04],
                         [ 2.73075312e+05],
                         [ 1.93103141e+05],
                         [ 1.70099375e+05],
                         [-2.48961906e+05],
                         [ 3.27879688e+04],
                         [ 2.50869180e+04],
                         [-3.71162617e+04],
                         [ 3.83556602e+04],
                         [-7.81622422e+04],
                         [-1.20918828e+04],
                         [ 5.06083008e+03],
                         [ 1.24135693e+04],
                         [-1.07722129e+04],
                         [ 1.01803105e+04],
                         [-2.39407422e+03],
                         [ 1.92025500e+03],
                         [-1.47800479e+04],
                         [ 1.07233242e+04],
                         [-1.27134863e+04],
                         [-1.35050303e+04],
                         [ 3.03227246e+03],
                         [-2.82783281e+04],
                         [ 4.03933911e+03],
                         [ 4.40725586e+02],
                         [-1.28661494e+04],
                         [ 1.94874670e+03],
                         [-4.26649658e+03],
                         [ 1.24085625e+04],
                         [-2.68236328e+03],
                         [-3.08830029e+03],
                         [-4.59871436e+03],
                         [-1.95311768e+02],
                         [-4.42659180e+03],
                         [-1.13439102e+04],
                         [ 7.96788208e+02],
                         [ 6.20482422e+03],
                         [-2.91427588e+03],
                         [ 1.15754382e+03]]'''
            #alpha_exp = np.zeros_like(alpha_exp)
            
            #alpha_shp = np.ones((40,1)).astype(np.float32)
            if dense_flag:
                #print(offset)
                offset[0] = one_euro_filter_x(offset[0])
                offset[1] = one_euro_filter_y(offset[1])
                offset[2] = one_euro_filter_z(offset[2])
                #print(offset)
                inp_dct = {
                    'R': R, 'offset': offset, 'alpha_shp': alpha_shp, 'alpha_exp': alpha_exp
                }
                pts3d = self.bfm_session.run(None, inp_dct)[0]
                pts3d = similar_transform(pts3d, roi_box, size)
            else:
                print('offset (tddfa onnx)')
                print(offset)
                pts3d = R @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp). \
                    reshape(3, -1, order='F') + offset
                pts3d = similar_transform(pts3d, roi_box, size)
            
            ver_lst.append(pts3d)
            
            """
            ver_lst2= []
            interieur = np.ndarray((3,36643))
            ver_lst2.append(interieur)
            
           
            print(np.shape(ver_lst2))
            mask = np.ones(np.shape(ver_lst)[2],dtype=bool)
            for k in range(self.keypoints[27],self.keypoints[35]):
                mask[k] = False
            print(self.keypoints[27])
            print(mask)
            #on enleve tous les vertex qui ont des indices compris entre les indices de ceux du nez
            ver_lst2[0][0]= ver_lst[0][0][mask]
            ver_lst2[0][1]= ver_lst[0][1][mask]
            ver_lst2[0][2]= ver_lst[0][2][mask]
            
        
           
            for k in range(self.keypoints[27],self.keypoints[35]):
                ver_lst[0][0][k]=None
                ver_lst[0][1][k]=None
                ver_lst[0][2][k]=None
                """
                
        return ver_lst
