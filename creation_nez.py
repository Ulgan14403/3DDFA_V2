import numpy as np
import trimesh.exchange
from DDFAV3.unwrapping import  Change_texture
from DDFAV3.demo import icp_masque
import argparse
import open3d as o3d


parser = argparse.ArgumentParser(description='3DDFA-V3')

parser.add_argument('-i', '--inputpath', default=r"E:/Antoine/OneDrive - ETS/Program_Files/videos test/0.Entrée/image/3ddfav3/", type=str,
                    help='path to the test data, should be a image folder')
parser.add_argument('-s', '--savepath', default=r'./examples/results', type=str,
                    help='path to the output directory, where results (obj, png files) will be stored.')
parser.add_argument('--device', default='cuda', type=str,
                    help='set device, cuda or cpu' )

# process test images
parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to crop input image, set false only when the test image are well cropped and resized into (224,224,3).' )
parser.add_argument('--detector', default='retinaface', type=str,
                    help='face detector for cropping image, support for mtcnn and retinaface')

# save
parser.add_argument('--ldm68', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='save and show 68 landmarks')
parser.add_argument('--ldm106', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='save and show 106 landmarks')
parser.add_argument('--ldm106_2d', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='save and show 106 landmarks, face profile is in 2d form')
parser.add_argument('--ldm134', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='save and show 134 landmarks' )
parser.add_argument('--seg', default=True, type=lambda x: x.lower() in ['true', '1'],
                    help='save and show segmentation in 2d without visible mask' )
parser.add_argument('--seg_visible', default=True, type=lambda x: x.lower() in ['true', '1'],
                    help='save and show segmentation in 2d with visible mask' )
parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                    help='save obj use texture from BFM model')
parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                    help='save obj use texture extracted from input image')

# backbone
parser.add_argument('--backbone', default='mbnetv3', type=str,
                    help='backbone for reconstruction, support for resnet50 and mbnetv3')

# video 
parser.add_argument('--video',default=False, help = 'runs the program on a video instead of an image, the path should point to the video')
#live
parser.add_argument('--live',default=False, help = 'runs the program on a webcam, will work only with --video also switched on True')

def Trimesh2Open3D(mesh_trimesh):
    trimesh.exchange.export.export_mesh(mesh_trimesh,'trimesh2open3d.obj')
    mesh_o3d = o3d.io.read_triangle_mesh('trimesh2open3d.obj')
    return(mesh_o3d)

def Open3D2Trimesh(mesh_o3d):
    o3d.io.write_triangle_mesh("o3d2trimesh.obj", mesh_o3d)
    mesh_trim = trimesh.load("o3d2trimesh.obj")
    return(mesh_trim)

def extract_textured_nose(image,nose_model):
    '''
    Fonction pour extraire le nez avec texture du masque de 3DDFAV3 et appliquer la texture sur le nez créé par le médecin
    Input :
    -image : Image dont le nez doit être extrait (numpy)
    -nose_model : modele de nez (trimesh)
    '''
    
    extracted_nose = icp_masque(image,parser.parse_args()) #renvoie le nez avec texture sous forme trimesh
    extracted_nose = Trimesh2Open3D(extracted_nose)
    nose_model = Trimesh2Open3D(nose_model)
    textured_nose = Change_texture(mesh_src=extracted_nose,mesh_tgt=nose_model,save=True,visu=False)
    vertex_colors = np.asarray(textured_nose.vertex_colors)
    textured_nose = Open3D2Trimesh(textured_nose)
    textured_nose.visual.face_colors = trimesh.visual.color.vertex_to_face_color(textured_nose.visual.vertex_colors,textured_nose.faces) # Changer couleur par sommet à couleur par face 
    
    return(textured_nose)

if __name__ == '__main__':
    import trimesh
    from PIL import Image
    nose = trimesh.load(r"E:\Antoine\OneDrive - ETS\Program_Files\PJ137\Dossier patient\patient014_nez.stl")
    
    
    image_icp = Image.open(r"E:/Antoine/OneDrive - ETS/Program_Files/videos test/0.Entrée/image/3ddfav3/photo.jpg")
    image_icp = np.array(image_icp)
    extr_nose = extract_textured_nose(image_icp,nose)
    print(extr_nose)
    
    
    