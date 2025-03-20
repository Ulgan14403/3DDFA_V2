import open3d as o3d
import numpy as np
import copy
import time
from sklearn.decomposition import PCA




def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482 ,0.1556])
    
    

def preprocess_point_cloud(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 20
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))
    return pcd_down, pcd_fpfh


def prepare_dataset(source,target,init,voxel_size):
    if init == 1:
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size *5) #3
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size  )
    else:
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size * init) #3
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size * init )
    return source, target, source_down, target_down, source_fpfh, target_fpfh

#voxel_size = 1 # means 5cm for this dataset



def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1 #todo ajuster les nouveaux paramètres à la nouvelle taille
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching( 
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999999))
    return result

def refine_registration(source, target, voxel_size,result_ransac):
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    init_source_to_target = result_ransac.transformation
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.000001,relative_rmse=0.000001,max_iteration=100)
    voxel_size = voxel_size *0.5
    max_correspondence_distance = 0.1
    reg_point_to_plane = o3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance,init_source_to_target, estimation, criteria)
    #draw_registration_result(source,target,reg_point_to_plane.transformation)
    return(reg_point_to_plane)

# Fast global registration

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size *0.5
  
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(division_factor = 1.4, use_absolute_scale= False, 
                                                                     decrease_mu = False, maximum_correspondence_distance = 0.025,
            iteration_number= 128, tuple_scale = 0.95, maximum_tuple_count= 1000, tuple_test= True)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, option)
    #draw_registration_result(source_down,target_down,result.transformation)
    return result

def register_via_correspondences(source,target,target_points,source_points) :
    corr = np.zeros((len(source_points), 2))
    corr[:, 0] = source_points
    corr[:, 1] = target_points
    correspondence_set = o3d.utility.Vector2iVector(corr)

    # Appliquer ICP avec correspondances connues
    result_correspondence = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True).compute_transformation(source, target, correspondence_set)
    return(result_correspondence)

def scale_pcd(source,target):
    boite_source = source.get_oriented_bounding_box()
    boite_source.color=[1.0,0.0,0.0]
    boite_target = target.get_oriented_bounding_box()
    boite_target.color=[0.0,0.0,1.0]
    points_source = boite_source.get_box_points()
    points_target = boite_target.get_box_points()
    
    delta_x_source = np.linalg.norm(points_source[7]-points_source[1])
    delta_y_source = np.linalg.norm(points_source[6]-points_source[1])
    delta_z_source = np.linalg.norm(points_source[0]-points_source[1])
    
    delta_x_target = np.linalg.norm(points_target[7]-points_target[1])
    delta_y_target = np.linalg.norm(points_target[6]-points_target[1])
    delta_z_target = np.linalg.norm(points_target[0]-points_target[1])
    
    facteur = min(delta_x_target/delta_x_source,delta_y_target/delta_y_source,delta_z_target/delta_z_source)
    
    source.scale(facteur,boite_source.get_center())
    return(source,facteur)


def custom_draw_geometry(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    for k in pcd:
        vis.add_geometry(k)
    vis.run()
    vis.destroy_window()

def aligne_boite(source,target):
    boite_source = source.get_minimal_oriented_bounding_box()
    boite_target = target.get_minimal_oriented_bounding_box()
    boite_target.color=[0.0,0.0,1.0]
    deplacement = np.subtract(boite_target.get_center(),boite_source.get_center())
    source = copy.deepcopy(source).translate(deplacement)
    
    #custom_draw_geometry([source,source.get_minimal_oriented_bounding_box(),target,target.get_minimal_oriented_bounding_box()])
    
    
    boite_source = source.get_oriented_bounding_box()
    boite_target = target.get_oriented_bounding_box()
    rotation_source = boite_source.R
    rotation_target = boite_target.R
    
    source = source.rotate(rotation_source.T,center=boite_source.center)
    source = source.rotate(rotation_target,center=boite_source.center)
    
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh2 = o3d.geometry.TriangleMesh.create_coordinate_frame()
    
    mesh.rotate(rotation_source)
    mesh.rotate(rotation_target)
    custom_draw_geometry([mesh,mesh2])   
    
    mesh = mesh.translate(boite_source.get_center())
    mesh.scale(30.0,mesh.get_center())
    mesh2 = copy.deepcopy(mesh).rotate(rotation_source)
    mesh2.scale(2.0,mesh.get_center())
    
    rotation_matrix =rotation_source.T*rotation_target
    custom_draw_geometry([boite_source,mesh,mesh2])
    
    
    
    
    custom_draw_geometry([boite_source,source,boite_target,target])
    return(rotation_source,rotation_target,deplacement)
    
    
    
    
    # boite_source.color=[1.0,0.0,0.0]
    
    # target_points=[0,1,2,3,4,5,6,7]
    # source_points=[0,1,2,3,4,5,6,7]
    
    # boite_source_pcd = o3d.geometry.PointCloud() 
    # boite_target_pcd = o3d.geometry.PointCloud()
    
    # '''
    # moitie_source =np.subtract(boite_source.get_box_points()[5],boite_source.get_box_points()[3])/2
    # moitie_target = np.subtract(boite_target.get_box_points()[5],boite_target.get_box_points()[3])/2
    
    # point_source_devant = np.subtract(boite_source.get_box_points()[5],moitie_source)
    # point_target_devant = np.subtract(boite_target.get_box_points()[5],moitie_target)
    
    # point_source_devant_haut = np.subtract(boite_source.get_box_points()[4],moitie_source)
    # point_target_devant_haut = np.subtract(boite_target.get_box_points()[4],moitie_target)
    
    # point_source_devant = np.reshape(point_source_devant,(1,3))
    # point_source_devant_haut = np.reshape(point_source_devant_haut,(1,3))
    # point_source_devant = np.append(point_source_devant,point_source_devant_haut,0)
    
    # point_target_devant = np.reshape(point_target_devant,(1,3))
    # point_target_devant_haut = np.reshape(point_target_devant_haut,(1,3))
    # point_target_devant = np.append(point_target_devant,point_target_devant_haut,0)
    
    
    # point_source = np.append(np.asarray(boite_source.get_box_points()),point_source_devant,0)
    # point_target = np.append(np.asarray(boite_target.get_box_points()),point_target_devant,0)
    
    # '''
    
    # point_source = np.asarray(boite_source.get_box_points())
    # point_target = np.asarray(boite_target.get_box_points())
    
    # boite_source_pcd.points = o3d.utility.Vector3dVector(point_source)
    # boite_target_pcd.points = o3d.utility.Vector3dVector(point_target)
    
    
    # #todo ajouter un point pour indiquer l'avant et l'arriere pb : ?? le mesh se retourne ??
    
    # result = register_via_correspondences(boite_source_pcd,boite_target_pcd,target_points,source_points)
    
    
    # custom_draw_geometry([source.transform(result),target])
    # #custom_draw_geometry([source.transform(result),source.transform(result).get_minimal_oriented_bounding_box(),target,target.get_minimal_oriented_bounding_box()])
    
    
    # flip = input("Flip ? y/n"+'\n')
    # if flip == 'y':
            
    #         target_points=[0,1,2,3,4,5,6,7]
    #         source_points=[5,4,3,2,1,0,7,6]
    #         result = register_via_correspondences(boite_source_pcd,boite_target_pcd,target_points,source_points)
    #         custom_draw_geometry([source.transform(result).get_minimal_oriented_bounding_box(),source.transform(result),target.get_minimal_oriented_bounding_box(),target])
         
    # return result
    
def aligne_boite_origine(source,target):
    boite_source = source.get_minimal_oriented_bounding_box()
    boite_target = target.get_minimal_oriented_bounding_box()
    deplacement_source = source.get_center()
    deplacement_target = target.get_center()
    
    target.translate(-deplacement_target)
    source.translate(-deplacement_source)
    print(source.get_center())
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(100) # origine
    custom_draw_geometry([source,target,mesh])
    
    boite_source = source.get_minimal_oriented_bounding_box()
    rotation_source = boite_source.R
    
    boite_target = target.get_minimal_oriented_bounding_box()
    rotation_target = boite_target.R
    
    print(rotation_source)
    source = source.rotate(rotation_source.T) 
    target = target.rotate(rotation_target.T)   
    custom_draw_geometry([source,target,mesh])
    rot_mat=np.eye(1,3,3)
    
    flip = input('Rotate ? rouge/bleu/vert/n'+'\n')
    while flip != 'n' :
        if flip == 'rouge' : #vert
            mat = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi/2,0,0])
        if flip == 'bleu' : 
            mat = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,np.pi/2])
        if flip == 'vert' :
            mat = o3d.geometry.get_rotation_matrix_from_axis_angle([0,np.pi/2,0])
        source.rotate(mat)
        custom_draw_geometry([source,target,mesh])
        rot_mat = np.append(rot_mat,mat,0)
        flip = input('Rotate ? rouge/bleu/vert/n'+'\n')
        
    #custom_draw_geometry([source.rotate(rot_mat)])
    return(rot_mat,deplacement_target)
    
def compute_pca_transform(pcd):
    """ Calcule le centre et la matrice de rotation basée sur le PCA d'un nuage de points """
    points = np.asarray(pcd.points)
    
    # Centrage des points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # PCA pour obtenir les axes principaux
    pca = PCA(n_components=3)
    pca.fit(centered_points)

    # La matrice de rotation est donnée par les axes principaux
    R = pca.components_.T  

    return centroid, R

def align_and_center_pcds(source, target):
    """ Aligne le nuage source sur le nuage cible en centrant d'abord les nuages à l'origine """
    # Calcul des matrices PCA et des centres
    
    centroid_src, R_src = compute_pca_transform(source)
    centroid_tgt, R_tgt = compute_pca_transform(target)

    boite_source = source.get_minimal_oriented_bounding_box()
    boite_target = target.get_minimal_oriented_bounding_box()

    
    centroid_src = boite_source.get_center()
    centroid_tgt = boite_target.get_center()
    
    # Déplacer les nuages à l'origine
    points_src = np.asarray(source.points) - centroid_src
    points_tgt = np.asarray(target.points) - centroid_tgt
    
    
    
  

    # Trouver la rotation qui aligne les axes principaux
    R_align = R_tgt @ R_src.T  # Matrice de transformation entre les bases

    # Appliquer la rotation à source
    aligned_points = points_src @ R_align.T

    # Replacer le nuage aligné au centre du nuage cible
    aligned_points += centroid_tgt

    # Création du nuage aligné
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)
    # Appliquer ICP (Point-to-Point)
    threshold = 50 # Distance maximale de correspondance
    icp_result = o3d.pipelines.registration.registration_icp(
        aligned_pcd, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    #o3d.visualization.draw_geometries([aligned_pcd.paint_uniform_color([0,0,1]),aligned_pcd.transform(icp_result.transformation).paint_uniform_color([1, 0, 0]),
                                       #target.paint_uniform_color([0,1,0])])  
    
    return R_align,centroid_tgt,centroid_src,icp_result.transformation
    
    
    
if __name__ == '__main__':
    source =  o3d.io.read_point_cloud(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\patient014_nez.ply")
    target =  o3d.io.read_point_cloud(r"E:\Antoine\OneDrive - ETS\Program_Files\GitHubs\3DDFA_V2\nez_masque.ply")
    source,facteur = scale_pcd(source,target)
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source,target,True,1)
    #aligne_boite(source,target)
    source_aligned = align_and_center_pcds(source, target)

    
    
    
    
    '''
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    
    #o3d.visualization.draw_geometries([source, target,mesh_frame],zoom=0.4559,front=[0.6452, -0.3036, -0.7011],lookat=[1.9892, 2.0208, 1.8945],up=[-0.2779, -0.9482 ,0.1556])
    
    voxel_size = 1
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source,target,3,voxel_size)
    start = time.time()
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print("global registration took %.3f sec.\n" % (time.time() - start))
    print(result_ransac)
    draw_registration_result(source, target, result_ransac.transformation)
    
    #source = source.transform(result_ransac.transformation)
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source,target,0.5,voxel_size)  
    start = time.time()
    result_icp = refine_registration(source, target_down,voxel_size,result_ransac)
    print("global registration took %.3f sec.\n" % (time.time() - start))
    print('result icp')
    print(result_icp.transformation)
    print('result ransac')
    print(result_ransac.transformation)
    print('difference')
    print(result_ransac.transformation[0]-result_icp.transformation[0])
    draw_registration_result(source, target, result_icp.transformation)   
    ''' 
    '''
    source1 =copy.deepcopy(source.transform(result_ransac.transformation))
    source1.paint_uniform_color([0.0,0.0,1.0])
    boite = source1.get_oriented_bounding_box()
    boite.color = [1.0,0.0,1.0]
    
    o3d.visualization.draw_geometries([source1,boite],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482 ,0.1556])
    
    '''
    
    '''
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source,target,20)
        
    
    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    draw_registration_result(source_down, target_down,
                            result_fast.transformation)
    '''
    
    
    '''
    
    idx_nez = []
    with open('points_communs.txt','r') as f:
        lines = f.readlines() 
        for line in lines :
            idx_nez.append(int(line))
        f.close()
    
    
    diff = len(source.points) - len(target.points) + len(idx_nez)
    
    target_points = [i for i in range(len(target.points)) if i not in idx_nez ]
    source_points = [i for i in range(diff,len(source.points))]
    start = time.time()
    register_via_correspondences(source,target,target_points,source_points)
    print('temps ecoule')
    print(time.time()-start)
    '''
    #colorer le nez
    '''
    source.paint_uniform_color([0.0,0.0,1.0])
    # Liste des indices des points à mettre en évidence
    highlight_indices = [i for i in range(diff)]

    # Extraire les couleurs existantes
    colors = np.asarray(source.colors)

    # Appliquer une couleur différente (par exemple, rouge) aux points spécifiés
    for idx in highlight_indices:
        colors[idx] = [1.0, 0.0, 0.0]  # Rouge
    
    source.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors.astype(np.float64)))
    # Afficher le nuage de points
    o3d.visualization.draw_geometries([source])
    '''
    
    
    