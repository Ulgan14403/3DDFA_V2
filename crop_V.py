#Transformation des images pour les prédictions

import mediapipe as mp
import torchvision.transforms as transforms
import cv2
import numpy as np

def crop (image,thresh=None):
    mp_face_detection = mp.solutions.face_detection
        
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.2) as face_detection:
            # Convertir l'image en RGB
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Détecter les visages
            results = face_detection.process(rgb_frame)
            #print(results.detections[0].score)
            

            # Parcourir les visages détectés
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                    offx, offy, w, h = bbox
                    
                    # Dessiner un rectangle autour du visage
                    '''
                    image = np.ascontiguousarray(image, dtype= np.uint8)
                    cv2.rectangle(image, (offx, offy), (offx + w, offy + h), (0, 255, 0), 2)
                    cv2.imshow('image',image)
                    cv2.waitKey(0)
                    '''
            else:
                offx,offy,w,h=10,10,10,10
    if thresh and results.detections is not None:
        return(np.array([[offx,offy,offx+w,offy+h],[0,0,0,0]]),results.detections[0].score)            
    
    return (np.array([[offx,offy,offx+w,offy+h],[0,0,0,0]]))