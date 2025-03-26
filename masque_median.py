import pyrealsense2 as rs
import numpy as np
import cv2
import alignment
import open3d as o3d
import pyvista as pv
import copy 
from sklearn.decomposition import PCA

def main(masque):
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
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)
    
    #Creation des filtres pour post process les images de profondeur
    threshold_filter = rs.threshold_filter(min_dist=0.20, max_dist=2.0)  # en mètres
    temporal_filter = rs.temporal_filter(smooth_alpha =0.40, smooth_delta = 20.0, persistence_control = 3)
    
    
    
    #Creation de l'image en 3d
    pos = np.zeros((resolution[1],resolution[0],3))
    for j in range (resolution[1]) :
        for i in range (resolution[0]) :
            pos[j,i,0] = j          
            pos[j,i,1] = i
    
            
    
    try: 
        while True:
            #Recuperer les frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            #Process pour enlever le fond
            filtered_depth = threshold_filter.process(depth_frame)
            # Convert images to numpy arrays
            frame = np.asanyarray(filtered_depth.get_data()) #sous forme numpy
            
           
            #Ajoute la frame dans les donnees de profondeur de l'image 3D
            pos[:,:,2] = frame
            pos_flat = np.reshape(pos,(resolution[1]*resolution[0],3))
            
            #Enlever le fond
            supp_idx = []
            for k in range(np.shape(pos_flat)[0]):
                if pos_flat[k,2] <= 1:
                    supp_idx.append(k)
            pos_flat = np.delete(pos_flat,supp_idx,axis=0)
           
            #Convertir vers open3D
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pos_flat.astype(np.float64)))
            
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(np.ascontiguousarray(masque.points.astype(np.float64)))

            
            #Aligner les 2 mesh sur le meme axe
            origine = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 100) #visualiser origine et les axes
            centroid,R = alignment.compute_pca_transform(source)
            source = source.rotate(np.linalg.inv(R))
            R = np.array(([0,-1,0],[1,0,0],[0,0,1]))
            source = source.rotate(R)
            o3d.visualization.draw_geometries([target,source,origine])

            
            #Prepare les pcd pour la global registration et l'icp
            source, target, source_down, target_down, source_fpfh, target_fpfh = alignment.prepare_dataset(source,target,0.1,110) # global registration
            source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
            #Global registration
            result_global = alignment.execute_global_registration(source_down,target_down,source_fpfh,target_fpfh,10)
            
            #ICP
            criteria_icp = o3d.pipelines.registration.ICPConvergenceCriteria()
            criteria_icp.max_iteration = 500
            criteria_icp.relative_rmse = 1e-04
            criteria_icp.relative_fitness = 1e-05
            result_icp = o3d.pipelines.registration.registration_icp(source,target,max_correspondence_distance = 0.7,
                                                    init = result_global.transformation,
                                                    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                    criteria = criteria_icp                                                   
                                                    )
            
            #source = source.transform(result_global.transformation)
            alignment.draw_registration_result(source,target,result_global.transformation)
            alignment.draw_registration_result(source,target,result_icp.transformation)
            return(0)
            
            
            #optimize global registration into icp 
            
            
            depth_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            print(depth_normalized)
            #Display l'image
            img_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_RAINBOW)
            cv2.imshow('image',img_colormap)
            cv2.waitKey(5)
            k = cv2.waitKey(10)
            if (k & 0xff == ord('q')):
                break
            
            
    finally:
        # Stop streaming
        pipeline.stop()

    


if __name__ == '__main__':
    masque_face = pv.read(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\masque_face.stl")
    masque_face.compute_normals(inplace=True)
    main(masque_face)
