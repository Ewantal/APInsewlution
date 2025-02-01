from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math
import json
import os
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from bust_pattern import BustPattern
from affichage import Affichage
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from bust_pattern import BustPattern
from affichage import Affichage
import gdown

app = Flask(__name__)
CORS(app)

# Directory to save uploaded files (if needed)
UPLOAD_FOLDER = './tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/process', methods=['POST'])
def process_images():
    # Extract uploaded files
    image1 = request.files.get('image1')
    image2 = request.files.get('image2')
    if not image1 or not image2:
        return jsonify({"success": False, "error": "Two images are required"}), 400

    # Save the images temporarily
    image1_path = os.path.join(UPLOAD_FOLDER, image1.filename)
    image2_path = os.path.join(UPLOAD_FOLDER, image2.filename)
    output1_path = os.path.join(UPLOAD_FOLDER, 'Segmented_Image1.png')
    output2_path = os.path.join(UPLOAD_FOLDER, 'Segmented_Image2.png')
    output3_path = os.path.join(UPLOAD_FOLDER, 'Edge_Image1.png')
    output4_path = os.path.join(UPLOAD_FOLDER, 'Edge_Image2.png')
    output5_path = os.path.join(UPLOAD_FOLDER, 'pose_annotated_image.jpg')
    image1.save(image1_path)
    image2.save(image2_path)

    # Extract additional form data
    height = float(request.form.get('height'))
    type_ = request.form.get('type')
    fit = request.form.get('fit')

                                                                            #######    SEGEMENTATION PART      #########

     # Initialize segmenter
    segmenter = PersonSegmenter()
    
    # Process images
    image_paths = [image1_path, image2_path]
    output_paths = [output1_path, output2_path]
    
    for img_path, out_path in zip(image_paths, output_paths):
        print(f"Processing {img_path}...")
        result = segmenter.segment_person(img_path, out_path)

    time.sleep(0.1)


                                                                                #######    CONTOUR PART      #########

    # Process first image
    result1 = detect_edges(output1_path, output3_path)
    
    # Process second image
    result2 = detect_edges(output2_path, output4_path)
    

    time.sleep(0.1)

                                                                                #######    POSE AND PI PART      #########

    image_path = output1_path

    # Initialize MediaPipe Pose solution
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

    # Initialize MediaPipe Drawing solution for visualization
    mp_drawing = mp.solutions.drawing_utils

    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image '{image_path}'. Please check the file path.")
        exit()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image.shape[:2]

    # Process the image to detect pose landmarks
    results = pose.process(image_rgb)

    # Check if pose landmarks were detected
    if results.pose_landmarks:
        # Draw pose landmarks on the image
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Save the annotated image with pose landmarks
        cv2.imwrite(output5_path, annotated_image)

        # Extract landmarks for calculating lengths
        landmarks = results.pose_landmarks.landmark

        # Get the coordinates of the left ankle and the nose for height measurement
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_mouth = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value]
        right_mouth = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]

        # Get the screen resolution using pyautogui
        screen_width, screen_height = pyautogui.size()

        edge = cv2.imread(output3_path) #Ouverture de l'image 'edge'
        edge_height, edge_width, _ = edge.shape

        #translation de l'image (a corriger)
        tx, ty = -35, -35       # Décalage en pixels sur l'axe x et y
        M = np.float32([[1, 0, tx], [0, 1, ty]])    # Créer la matrice de translation
        rows, cols = edge.shape[:2]    # Obtenir les dimensions de l'image
        #edge = cv2.warpAffine(edge, M, (cols, rows))    #Appliquer la translation



        # Calculate the top of the head as an average of the nose and left ear
        top_of_head_y = find_top_of_head(edge, nose.x, nose.y, edge_height, edge_width)
        top_of_head_x = nose.x

        # Create a landmark object for the top of the head
        top_of_head = type('Landmark', (object,), {'x': top_of_head_x, 'y': top_of_head_y})
        print(top_of_head.y, nose.y)

        # Calculate the height of the person in pixels based on the distance between the ankle and the top of the head
        height_dec = left_heel.y - top_of_head.y
        height_pixels = height_dec * image_height

        # Actual height in meters
        actual_height = height/100  # meters

        # Calculate the scaling factor (meters per pixel)
        scaling_factor = actual_height / height_pixels #m/pixel


        #Recuperation des points des épaules
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        #Recuperation des points des genoux
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

        #recherche du cou
        neck_line, neck_col_1, neck_col_2, shoulder_line = research_neck(edge, right_mouth.y, left_shoulder.y, left_shoulder.x, right_shoulder.y, right_shoulder.x, edge_height, edge_width)


        #recherche de la longueur taille epaule
        L_hip_shoulder = length_hip_shoulder(edge,shoulder_line, left_hip.y, left_hip.x, left_shoulder.y, left_shoulder.x, edge_height, edge_width, scaling_factor)


        #calcule des dimensions du cou
        R_neck_pixel = (neck_col_2 - neck_col_1)
        D_neck = R_neck_pixel * scaling_factor


        #recherche des aisselles
        left_armpit_x, left_armpit_y = find_armpit(edge, left_shoulder.x, left_shoulder.y, left_hip.y, edge_height, edge_width, 'left')
        right_armpit_x, right_armpit_y = find_armpit(edge, right_shoulder.x, right_shoulder.y, left_hip.y, edge_height, edge_width, 'right')


        #calcule inter-aisselles
        d_inter_armpit_pixel = left_armpit_x - right_armpit_x
        d_inter_armpit = d_inter_armpit_pixel * scaling_factor


        #recherche angles cou
        left_syce_x, left_syce_y = find_syce(edge, int(left_shoulder.x * edge_width), shoulder_line, neck_col_2, neck_line, 'left')
        right_syce_x, right_syce_y = find_syce(edge, int(right_shoulder.x * edge_width), shoulder_line, neck_col_1, neck_line, 'right')


        #taille buste
        L_bust_pixel = left_armpit_y - shoulder_line
        L_bust = L_bust_pixel * scaling_factor


        #taille mi-dos
        half_back_pixel = left_armpit_x - int(nose.x * edge_width) + 1
        half_back = half_back_pixel * scaling_factor


        #taille bras
        wrist_line = find_wrist_line(edge, left_wrist.y, left_wrist.x, edge_height, edge_width)
        L_arm_pixel = calculate_distance_px(int(left_wrist.x * edge_width), find_wrist_line(edge, left_wrist.y, left_wrist.x, edge_height, edge_width), (left_shoulder.x * edge_width), shoulder_line)
        L_arm = L_arm_pixel * scaling_factor


        #taille épaule cou
        L_shoulder_neck_pixel = calculate_distance_px((left_shoulder.x * edge_width), shoulder_line, left_syce_x, left_syce_y)
        L_shoulder_neck = L_shoulder_neck_pixel * scaling_factor


        #longueur inter-epaule
        L_shoulder_shoulder_pixel = calculate_distance_px((left_shoulder.x * edge_width), shoulder_line, (right_shoulder.x * edge_width), shoulder_line)
        L_shoulder_shoulder = L_shoulder_shoulder_pixel * scaling_factor


        #hauteur aisselle hanche
        L_hip_armpit_pixel = calculate_distance_px((left_hip.x * edge_width), (left_hip.y * edge_height), left_armpit_x, left_armpit_y)
        L_hip_armpit = L_hip_armpit_pixel * scaling_factor
        
        
        #hauteur cou hanche
        L_hip_neck_pixel = calculate_distance_px((left_hip.x * edge_width), (left_hip.y * edge_height), left_syce_x, left_syce_y)
        L_hip_neck = L_hip_neck_pixel * scaling_factor


        #recherche de la partie la moins epaisse du "ventre"
        hip_line, hip_col_1, hip_col_2 = find_hip(edge, left_armpit_x, left_armpit_y, right_armpit_x, right_armpit_y, left_hip.y, edge_height, edge_width)
        d_inter_hipPixel = hip_col_2 - hip_col_1
        d_inter_hip = d_inter_hipPixel * scaling_factor

        image_height, image_width, _ = image.shape

        #Recherche de ration epaule-aisselle (0 si l'aisselle se trouve au niveau de l'épaule, 1 si l'aisselle se trouve au niveau du genou). Utile pour photo profil
        ratioShoulderArmpit = calculate_ratio(left_armpit_y, left_shoulder.y, left_knee.y,"Armpit", image_height)
        #print(f'Armpit ratio: {ratioShoulderArmpit:.3f}')

        ratioShoulderHip = calculate_ratio(hip_line, left_shoulder.y, left_knee.y,"Hip", image_height)
        #print(f'Hip ratio: {ratioShoulderHip:.3f}')

        armpit_width_meters, hip_width_meters = process_profile_image(
        output2_path,
        output4_path,
        ratioShoulderArmpit,
        ratioShoulderHip,
        mp_pose,
        pose,
        actual_height,
        edge,
    )

        #print(f'Armpit width: {armpit_width_meters:.2f}m')
        #print(f'Hip width: {hip_width_meters:.2f}m')

        tour_de_buste = calculElipse(armpit_width_meters/2, d_inter_armpit/2)
        tour_de_taille = calculElipse(hip_width_meters/2, d_inter_hip/2)

        print(f'Tour de buste: {tour_de_buste:.2f}m')
        print(f'Tour de taille: {tour_de_taille:.2f}m')

        print_size(D_neck, L_hip_shoulder, d_inter_armpit, L_bust, half_back, L_arm, L_shoulder_neck, L_shoulder_shoulder, L_hip_armpit, L_hip_neck)
        data = print_json(D_neck, L_hip_shoulder, d_inter_armpit, L_bust, half_back, L_arm, L_shoulder_neck, L_shoulder_shoulder, L_hip_armpit, L_hip_neck)
        draw_point(edge, annotated_image, shoulder_line, left_shoulder.x, right_shoulder.x, wrist_line, left_wrist.x, left_armpit_x, left_armpit_y, right_armpit_x, right_armpit_y, hip_line, hip_col_1, hip_col_2, edge_width, left_syce_x, left_syce_y, right_syce_x, right_syce_y, neck_col_1, neck_col_2, neck_line)


    # Release the pose model resources
    pose.close()


                                                                                #######    GENERATE MODEL PART     #########

    invoke_pattern(data, style=fit, sleeves=type_)

                                                                                #######    END  PART      #########

    # Cleanup: Delete images after processing
    os.remove(output1_path)
    os.remove(output2_path)
    os.remove(output3_path)
    os.remove(output4_path)
    os.remove(output5_path)
    cv2.destroyAllWindows()
       
    # Example: Simulating processing logic
    result = {
        "message": "Images processed successfully",
        "processed_height": height,
        "processed_type": type_,
        "processed_fit": fit,
    }

    # Return a success response
    return jsonify({"success": True, "result": result})




                                                                            #######    SEGMENTATION FUNCTIONS      #########

class PersonSegmenter:
    def __init__(self, sam_checkpoint="fichiers_model/sam_vit_h_4b8939.pth"):
        """Initialize the PersonSegmenter with SAM model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SAM
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        
        # Initialize YOLO for person detection
        self.detector = YOLO('fichiers_model/yolov8x.pt')
    
    def detect_person(self, image):
        """Detect person in the image using YOLO"""
        results = self.detector(image)
        
        # Filter for person class (class 0 in COCO)
        person_boxes = []
        person_scores = []
        
        for result in results:
            boxes = result.boxes
            for box, cls in zip(boxes.xyxy, boxes.cls):
                if int(cls) == 0:  # person class
                    person_boxes.append(box.cpu().numpy())
                    person_scores.append(boxes.conf[len(person_scores)].cpu().numpy())
        
        if not person_boxes:
            return None
        
        # Get the person detection with highest confidence
        person_boxes = np.array(person_boxes)
        person_scores = np.array(person_scores)
        best_idx = np.argmax(person_scores)
        
        return person_boxes[best_idx]
    
    def segment_person(self, image_path, output_path=None):
        """Segment person from the image"""
        # Read image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect person
        bbox = self.detect_person(image_rgb)
        if bbox is None:
            print("No person detected in the image")
            return None
        
        # Prepare image for SAM
        self.predictor.set_image(image_rgb)
        
        # Get segmentation mask using the bounding box
        masks, scores, logits = self.predictor.predict(
            box=bbox,
            multimask_output=True
        )
        
        # Select the mask with highest score
        best_mask = masks[scores.argmax()]
        
        # Create final segmented image
        segmented_image = self.create_segmented_image(image, best_mask)
        
        # Save if output path is provided
        if output_path:
            cv2.imwrite(output_path, segmented_image)
        
        return segmented_image
    
    def create_segmented_image(self, image, mask):
        """Create final image with black silhouette on white background"""
        # Convert boolean mask to uint8
        mask = mask.astype(np.uint8) * 255
        
        # Create white background and black silhouette
        white_background = np.ones_like(image) * 255
        black_silhouette = np.zeros_like(image)  # Black silhouette
        
        # Apply masks
        segmented_image = cv2.bitwise_and(black_silhouette, black_silhouette, mask=mask)
        white_background = cv2.bitwise_and(white_background, white_background, 
                                         mask=cv2.bitwise_not(mask))
        final_image = cv2.add(segmented_image, white_background)
        
        return final_image

                                                                            #######    CONTOUR FUNCTIONS      #########

def detect_edges(image_path, output_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error: Unable to load image {image_path}")

    # Create a binary mask of white pixels (255)
    white_pixels = (image == 255)

    # Define an 8-connected kernel for checking neighboring pixels
    kernel = np.array([[1,1,1], [1,0,1], [1,1,1]], dtype=np.uint8)

    # Count non-white (black/gray) neighbors using OpenCV's filter2D
    neighbor_count = cv2.filter2D((~white_pixels).astype(np.uint8), -1, kernel)

    # Edge pixels: White pixels that have at least one black/gray neighbor
    edges = white_pixels & (neighbor_count > 0)

    # Convert boolean mask to uint8 image (255 for edges, 0 otherwise)
    result = (edges * 255).astype(np.uint8)

    # Save the result
    cv2.imwrite(output_path, result)
    
    return result

                                                                            #######    POSE AND PI FUNCTIONS      #########

# Function to resize the image to fit the screen while maintaining aspect ratio
def resize_to_fit_screen(image, screen_width, screen_height):
    image_height, image_width = image.shape[:2]

    # Calculate the aspect ratio of the image and the screen
    width_ratio = screen_width / image_width
    height_ratio = screen_height / image_height
    scale_ratio = min(width_ratio, height_ratio)  # Use the smaller ratio to maintain aspect ratio

    # Resize the image using the calculated scale ratio
    new_width = int(image_width * scale_ratio)
    new_height = int(image_height * scale_ratio)

    return cv2.resize(image, (new_width, new_height))



# Function to calculate Euclidean distance between two landmarks (2D: x, y)
def calculate_distance_2d(landmark1, landmark2):
    return math.sqrt(
        (landmark2.x - landmark1.x) ** 2 +
        (landmark2.y - landmark1.y) ** 2
    )

def calculate_distance_px(x1, y1, x2, y2):
    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )

def normalize_Ycoordinate(pixel_y, image_height):
    
    return pixel_y / image_height



def find_top_of_head(edge, nose_x, nose_y, height, width):
    i = int(nose_y * height) + 1
    j = int(nose_x * width)
    top = 0
    while i > 0 and top == 0:
        if edge[i,j][1] !=0:
            top = i
        # point = (j, i)
        # color = (0, 0, 255)
        # cv2.circle(edge, point, radius=0, color=color, thickness=-1)
        i-=1
    return top/height



def research_neck(edge, mouth_y, left_shoulder_y, left_shoulder_x, right_shoulder_y, right_shoulder_x, height, width): #fonction qui recherche le cou
    y_pixel_mouth = int(mouth_y * height) + 1
    i1 = int(right_shoulder_y * height)
    i2 = int(left_shoulder_y * height)
    neck_width = int((left_shoulder_x - right_shoulder_x) * width)
    w = width
    j1 = int(right_shoulder_x * width) #on initialise à la taille des épaules puis on réduit de chaque cote
    j2 = int(left_shoulder_x * width)
    while edge[i1,j1][0] == 0: #cherche la ligne d'épaule pour commencer la recherche du cou au dessus
         i1-=1
    while edge[i2,j2][0] == 0: #cherche la ligne d'épaule pour commencer la recherche du cou au dessus
         i2-=1
    i = min(i1, i2)
    shoulder_line = i1
    M1 = 0
    M2 = 0
    I = i
    i = i-50
    print(i, y_pixel_mouth)
    while i > y_pixel_mouth: #on augmlente à chaque ligne en cherchant les colones du cou
        w = width
        j1 = int(right_shoulder_x * width) #on initialise à la taille des épaules puis on réduit de chaque cote
        j2 = int(left_shoulder_x * width)
        m1, m2 = 0, 0
        while w == width and j1 < j2:  #tant qu'on a pas trouver une nouvelle taille de cou on continue
            if m1 != 0 and m2 != 0:
                w = j2 - j1
                point = (j2, i)
                point2 = (j1, i)
                color = (0, 0, 255)
                cv2.circle(edge, point, radius=1, color=color, thickness=-1)
                cv2.circle(edge, point2, radius=1, color=color, thickness=-1) #on affiche les points du cou sur l'image
                break
            if edge[i,j1][0] != 0:
                m1 = j1
            if edge[i,j2][0] != 0:
                m2 = j2
            if m1 == 0 and m2 == 0:
                j2-=1
                j1+=1
            elif m1 != 0 :
                j2-=1
            elif m2 != 0:
                j1+=1
            else:
                print("err")
        i-=1


        if w<neck_width: #on renvoit les coordonées du cou (ligne et les deux points de colone)
            neck_width = w
            M1 = m1
            M2 = m2
            I = i
    return I, M1, M2, shoulder_line




def length_hip_shoulder(edge, shoulder_line, left_hip_y, left_hip_x, left_shoulder_y, left_shoulder_x, height, width, scaling_factor):
    l = abs((shoulder_line - left_hip_y * height) * scaling_factor)

    # point = (int(left_shoulder_x * width), int(left_hip_y * height))
    # point2 = (int(left_shoulder_x * width), shoulder_line)
    # point3 = (int(left_hip_x * width), int(left_hip_y * height))
    # point4 = (int(left_hip_x * width), shoulder_line)
    # color = (0, 0, 255)
    # cv2.circle(edge, point, radius=5, color=color, thickness=-1)
    # cv2.circle(edge, point2, radius=5, color=color, thickness=-1) #on affiche les points du cou sur l'image
    #
    # color2 = (0, 255, 0)
    # thickness = 5
    # cv2.line(edge, point, point2, color2, thickness)  # Dessiner la ligne
    # cv2.line(edge, point3, point4, color2, thickness)  # Dessiner la ligne

    return l




def find_pointA(edge, shoulder_x, shoulder_y, hip_y, height, width, side):
    i = (int(hip_y * height) + int(shoulder_y * height))//2
    j1 = int(shoulder_x * width)
    j2 = j1
    while edge[i,j1][0] == 0 and edge[i,j2][0] == 0 and j1 < width-1 and j2>1:
        j1+=1
        j2-=1
        # cv2.circle(edge, [j1, i], radius=1, color=[0,255,255], thickness=-1)
        # cv2.circle(edge, [j2, i], radius=1, color=[0,255,255], thickness=-1)
    if edge[i,j1][0] != 0:
        return i,j1
    elif edge[i,j2][0] != 0:
        return i,j2



def find_pointB(edge, xA, yA, side):
    i = yA
    if side == 'left':
        j = xA + 100
    else:
        j = xA - 100
    while edge[i,j][0] == 0: #cherche la ligne basse du bras
         i-=1
    return i,j



def find_angle(contours, side):
        # Prendre le contour le plus grand (supposant qu'il correspond à la courbe)
        contour = max(contours, key=cv2.contourArea)
        #contour = contour[len(contour)//2:]
        x = []
        y = []
        for i in range(len(contour)):
            x.append(contour[i][0][0])
            y.append(contour[i][0][1])

        x = np.array(x)
        y = np.array(y)

        #plt.plot(x,y, label = '1')

        window_size = 80  # Taille de la fenêtre (ajuster selon le lissage souhaité)
        x = moving_average(x, window_size)
        y = moving_average(y, window_size)

        #plt.plot(x,y, label = '2')
        x, y = cut_curve(x, y, side)


        # for i in range(len(x)):
        #     if side == 'left':
        #         cv2.circle(edge, (int(x[i]+xA-10), int(y[i]+yB-10)), 2, (255, 0, 255), -1)
        #     else:
        #         cv2.circle(edge, (int(x[i]+xB-10), int(y[i]+yB-10)), 2, (0, 255, 255), -1)

        plt.plot(x,y, label = '3')
        plt.legend()
        #plt.show()

        # Distance de comparaison en indices
        distance = 4

        # Calcul des vecteurs tangents espacés
        # On prend des points séparés par `distance` indices
        v1 = np.array([x[distance:] - x[:-distance], y[distance:] - y[:-distance]])

        # Décalage d'un autre `distance` indices pour obtenir v2
        v2 = np.array([x[2*distance:] - x[distance:-distance], y[2*distance:] - y[distance:-distance]])

        # Ajustement pour s'assurer que les tailles sont compatibles
        min_length = min(v1.shape[1], v2.shape[1])
        v1, v2 = v1[:, :min_length], v2[:, :min_length]

        # Calcul des produits scalaires
        dot_products = np.sum(v1 * v2, axis=0)

        # Calcul des normes
        norm1 = np.linalg.norm(v1, axis=0)
        norm2 = np.linalg.norm(v2, axis=0)

        # Calcul des cosinus des angles
        cos_theta = dot_products / (norm1 * norm2)

        # Calcul des angles en radians
        angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        # Trouver l'indice de l'angle maximal
        max_angle_index = np.argmax(angles) + distance  # +distance pour correspondre à l'indice original dans x, y

        # Récupérer les coordonnées du point correspondant
        max_x = int(x[max_angle_index])
        max_y = int(y[max_angle_index])

        return max_x, max_y



def find_angle2(contours, side):
        # Prendre le contour le plus grand (supposant qu'il correspond à la courbe)
        contour = max(contours, key=cv2.contourArea)
        #contour = contour[len(contour)//2:]
        x = []
        y = []
        for i in range(len(contour)):
            x.append(contour[i][0][0])
            y.append(contour[i][0][1])

        x = np.array(x)
        y = np.array(y)

        #plt.plot(x,y, label = '1')

        window_size = 20  # Taille de la fenêtre (ajuster selon le lissage souhaité)
        x = moving_average(x, window_size)
        y = moving_average(y, window_size)

        #plt.plot(x,y, label = '2')

        # x, y = cut_curve(x, y, side)


        # for i in range(len(x)):
        #     cv2.circle(binary, (int(x[i]), int(y[i])), 2, (255, 0, 255), -1)
        #     if side == 'left':
        #         cv2.circle(edge, (int(x[i]+xA-10), int(y[i]+yB-10)), 2, (255, 0, 255), -1)
        #     else:
        #         cv2.circle(edge, (int(x[i]+xB-10), int(y[i]+yB-10)), 2, (0, 255, 255), -1)

        #plt.plot(x,y, label = '3')
        plt.legend()
        #plt.show()

        # Distance de comparaison en indices
        distance = 4

        # Calcul des vecteurs tangents espacés
        # On prend des points séparés par `distance` indices
        v1 = np.array([x[distance:] - x[:-distance], y[distance:] - y[:-distance]])

        # Décalage d'un autre `distance` indices pour obtenir v2
        v2 = np.array([x[2*distance:] - x[distance:-distance], y[2*distance:] - y[distance:-distance]])

        # Ajustement pour s'assurer que les tailles sont compatibles
        min_length = min(v1.shape[1], v2.shape[1])
        v1, v2 = v1[:, :min_length], v2[:, :min_length]

        # Calcul des produits scalaires
        dot_products = np.sum(v1 * v2, axis=0)

        # Calcul des normes
        norm1 = np.linalg.norm(v1, axis=0)
        norm2 = np.linalg.norm(v2, axis=0)

        # Calcul des cosinus des angles
        cos_theta = dot_products / (norm1 * norm2)

        # Calcul des angles en radians
        angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        # Trouver l'indice de l'angle le plus proche de 90° (pi/2 radians)
        closest_to_90_index = np.argmin(np.abs(angles - np.pi / 2)) + distance  # +distance pour correspondre à l'indice original dans x, y

        # Récupérer les coordonnées du point correspondant
        max_x = int(x[closest_to_90_index])
        max_y = int(y[closest_to_90_index])

        return max_x, max_y



def courbure(p1, p2, p3):
    # Calcul de la courbure en fonction des distances entre les points
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    if a * b * c == 0:
        return 0
    return (4 * np.abs(np.cross(p2 - p1, p3 - p1))) / (a * b * c)



def moving_average(data, window_size):
    return savgol_filter(data, window_length=window_size, polyorder=2)



def cut_curve(x, y, side):
    X = x[0]
    Y = y[0]
    min_x = 0
    max_y = 0
    for i in range(len(x)):
        if side == 'left':
            if x[i] > X:
                X = x[i]
                min_x = i
        else:
            if x[i] < X:
                X = x[i]
                min_x = i
        if y[i] > Y:
            Y = y[i]
            max_y = i
    i1 = min(max_y, min_x)
    i2 = max(max_y, min_x)
    return x[i1+5:i2-5], y[i1+5:i2-5]



def find_armpit(edge, shoulder_x, shoulder_y, hip_y, height, width, side):
    yA, xA = find_pointA(edge, shoulder_x, shoulder_y, hip_y, height, width, side)
    yB, xB = find_pointB(edge, xA, yA, side)

    A = (xA, yA)
    B = (xB, yB)


    if side == 'left':
        binary = edge[yB-10:yA+10, xA-10:xB+10]
    else:
        binary = edge[yB-10:yA+10, xB-10:xA+10]

    binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    # color = (0, 255, 255)
    # cv2.circle(edge, A, radius=5, color=color, thickness=-1)
    # cv2.circle(edge, B, radius=5, color=color, thickness=-1)

    # Détection des contours dans la ROI
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Vérifier si des contours ont été détectés
    if contours:
        max_x, max_y = find_angle(contours, side)

        if side == 'left':
            max_x+=xA-10
            max_y+=yB-10
            #cv2.circle(edge, (max_x, max_y), 10, (255, 0, 255), -1)
        else:
            max_x+=xB-10
            max_y+=yB-10
            #cv2.circle(edge, (max_x, max_y), 10, (255, 0, 255), -1)

        # Afficher l'image avec le point le plus anguleux
        # cv2.imshow("Point le plus anguleux", binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return max_x, max_y
    else:
        print("Aucun contour détecté.")



def find_hip(edge, left_armpit_x, left_armpit_y, right_armpit_x, right_armpit_y, left_hip_y, height, width):
    i = max(left_armpit_y, right_armpit_y)
    width1 = left_armpit_x - right_armpit_x
    width2 = left_armpit_x - right_armpit_x
    I1, I2 = left_armpit_y, left_armpit_y
    while i < int(left_hip_y * height):
        w = width1
        j1 = right_armpit_x - 1 #on initialise à la taille des aisselles puis on réduit de chaque cote attention solution pour les personnes maigres
        j2 = left_armpit_x + 1
        j3 = right_armpit_x + 1 #on initialise à la taille des aisselles puis on augmente de chaque cote attention solution pour les personnes enveloppees
        j4 = left_armpit_x - 1
        m1, m2, m3, m4 = 0, 0, 0, 0
        while w == width1 and j1 < j2 and j3 < width and j4 > 0:  #tant qu'on a pas trouver une nouvelle taille de hanche on continue
            if m1 != 0 and m2 != 0:
                w = j2 - j1
                break
            if m3 != 0 and m4 != 0:
                w = j4 - j3
                break
            if edge[i,j1][0] !=0:
                m1 = j1
            if edge[i,j2][0] !=0:
                m2 = j2
            if edge[i,j3][0] !=0:
                m1 = j3
            if edge[i,j4][0] !=0:
                m2 = j4
            if m1 == 0:
                j1+=1
            if m2 == 0:
                j2-=1
            if m3 == 0:
                j3-=1
            if m4 == 0:
                j1+=1
            else:
                print("err")
        i+=1

        if w<width1 and w>10: #on renvoit les coordonées du cou (ligne et les deux points de colone)
            #print(i ,j1, j2, w)
            width1 = w
            M1 = m1
            M2 = m2
            I1 = i
            
            
        if w>width2 and w<width: #on renvoit les coordonées du cou (ligne et les deux points de colone)
            print(i ,j1, j2, w)
            width2 = w
            M3 = m1
            M4 = m2
            I2 = i
            I1 = I2
    
    if I1 != left_armpit_y and I2 != left_armpit_y:
        middle = left_hip_y * height - left_armpit_y
        if abs(I1-middle) < abs(I2-middle):
            return I1, M1, M2
        else:
            return I2, M3, M4
    
    elif I1 != left_armpit_y:
        return I1, M1, M2
    
    elif I2 != left_armpit_y:
        return I2, M3, M4

    else:
        print("err in find_hip")
    



def find_syce(edge, shoulder_x, shoulder_y, neck_x, neck_y, side):
    # A = (neck_x, neck_y)
    # B = (shoulder_x, shoulder_y)
    if side == 'left':
        binary = edge[neck_y-10:shoulder_y+10, neck_x-10:shoulder_x+10]
        binary = edge[neck_y-10:shoulder_y+10, neck_x-10:neck_x+10]
    else:
        binary = edge[neck_y-10:shoulder_y+10, shoulder_x-10:neck_x+10]
        binary = edge[neck_y-10:shoulder_y+10, neck_x-10:neck_x+10]

    binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

    # Détection des contours dans la ROI
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Vérifier si des contours ont été détectés
    if contours:
        max_x, max_y = find_angle2(contours, side)
        # cv2.circle(binary, (max_x, max_y), 10, (255, 0, 255), -1)
        # cv2.imshow(side, binary)
        # cv2.waitKey(2)
        if side == 'left':
            max_x+=neck_x-10
            max_y+=neck_y-10
            cv2.circle(edge, (max_x, max_y), 2, (0, 0, 255), -1)
        else:
            max_x+=neck_x-10
            max_y+=neck_y-10
            cv2.circle(edge, (max_x, max_y), 2, (0, 0, 255), -1)

        #Afficher l'image avec le point le plus anguleux
        # cv2.imshow("Point le plus anguleux", binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return max_x, max_y
    else:
        print("Aucun contour détecté.")


def find_wrist_line(edge, wrist_y, wrist_x, height, width):
    i = int(wrist_y * height)
    j1 = int(wrist_x * width)
    while edge[i,j1][0] == 0: #cherche la ligne d'épaule pour commencer la recherche du cou au dessus
         i-=1
    return i

def calculate_ratio(yCoordinate, shoulder_y, knee_y, name, image_height):
    y_normalized = normalize_Ycoordinate(yCoordinate, image_height)
    #print(f'{name} {y_normalized} shoulder {shoulder_y} knee {knee_y}')
    total_distance = knee_y - shoulder_y
    if total_distance == 0:
        return 0
    #print(f'total distance {total_distance}')
    distance = y_normalized - shoulder_y
    #print(f'{name} distance {distance}')
    ratio = distance / total_distance
         
    return ratio

def find_edge_points(edge_image, y_level):
    """Find leftmost and rightmost white pixels at a given y-level."""
    height, width = edge_image.shape[:2]
    if y_level < 0 or y_level >= height:
        return None, None
    
    # Convert to grayscale if needed
    if len(edge_image.shape) == 3:
        gray = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = edge_image
    
    # Get the row at y_level
    row = gray[y_level, :]
    
    # Find indices of non-zero (white) pixels
    white_pixels = np.where(row != 0)[0]
    
    if len(white_pixels) == 0:
        return None, None
    
    # Get leftmost and rightmost points
    left_x = white_pixels[0]
    right_x = white_pixels[-1]
    
    return left_x, right_x

def calculate_profile_scaling_factor(profile_image_path, mp_pose, pose, actual_height, edge):
    # Load and process the profile image
    profile_image = cv2.imread(profile_image_path)
    if profile_image is None:
        print(f"Error: Unable to load profile image '{profile_image_path}'")
        return None
        
    profile_rgb = cv2.cvtColor(profile_image, cv2.COLOR_BGR2RGB)
    profile_height, profile_width = profile_image.shape[:2]
    
    # Process the profile image to detect pose landmarks
    profile_results = pose.process(profile_rgb)
    
    if profile_results.pose_landmarks:
        landmarks = profile_results.pose_landmarks.landmark
        
        # Get the heel and nose landmarks for height calculation
        heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        
        # Calculate top of head position (can be approximated from nose and ear positions)
        top_of_head_y = find_top_of_head(edge, left_ear.x, left_ear.y, profile_height, profile_width)  # Approximate adjustment for top of head
        
        # Calculate the height in pixels
        height_dec = heel.y - top_of_head_y
        height_pixels = height_dec * profile_height
        
        # Calculate the scaling factor for profile image using the provided actual height
        profile_scaling_factor = actual_height / height_pixels
        
        return profile_scaling_factor
    else:
        print("No pose landmarks detected in profile image")
        return None


def analyze_profile_image(edge_profile, segmented_profile, armpit_ratio, hip_ratio, mp_pose, pose, profile_scaling_factor):
    # Convert the segmented image to RGB for MediaPipe
    profile_rgb = cv2.cvtColor(segmented_profile, cv2.COLOR_BGR2RGB)
    
    # Process the profile image to detect pose landmarks
    profile_results = pose.process(profile_rgb)
    
    if profile_results.pose_landmarks:
        # Get landmarks
        landmarks = profile_results.pose_landmarks.landmark
        
        # Get necessary landmarks for height calculation
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        
        # Get image dimensions
        height, width = edge_profile.shape[:2]
        
        # Calculate pixel coordinates
        shoulder_y = int(shoulder.y * height)
        knee_y = int(knee.y * height)
        
        # Calculate the positions using the ratios
        total_distance = knee_y - shoulder_y
        armpit_line_y = int(shoulder_y + (total_distance * armpit_ratio))
        hip_line_y = int(shoulder_y + (total_distance * hip_ratio))
        
        # Draw reference lines (gray)
        cv2.line(edge_profile, (0, shoulder_y), (width, shoulder_y), (128, 128, 128), 1)
        cv2.line(edge_profile, (0, knee_y), (width, knee_y), (128, 128, 128), 1)
        
        # Find and draw edge points for armpit level
        armpit_left, armpit_right = find_edge_points(edge_profile, armpit_line_y)
        armpit_width_meters = None
        if armpit_left is not None and armpit_right is not None:
            armpit_width_pixels = armpit_right - armpit_left
            armpit_width_meters = armpit_width_pixels * profile_scaling_factor
            
            cv2.line(edge_profile, (armpit_left, armpit_line_y), 
                    (armpit_right, armpit_line_y), (0, 255, 0), 2)
            cv2.circle(edge_profile, (armpit_left, armpit_line_y), 3, (0, 255, 0), -1)
            cv2.circle(edge_profile, (armpit_right, armpit_line_y), 3, (0, 255, 0), -1)
        
        # Find and draw edge points for hip level
        hip_left, hip_right = find_edge_points(edge_profile, hip_line_y)
        hip_width_meters = None
        if hip_left is not None and hip_right is not None:
            hip_width_pixels = hip_right - hip_left
            hip_width_meters = hip_width_pixels * profile_scaling_factor
            
            cv2.line(edge_profile, (hip_left, hip_line_y), 
                    (hip_right, hip_line_y), (255, 0, 0), 2)
            cv2.circle(edge_profile, (hip_left, hip_line_y), 3, (255, 0, 0), -1)
            cv2.circle(edge_profile, (hip_right, hip_line_y), 3, (255, 0, 0), -1)
        
        # Add labels with metric measurements
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        cv2.putText(edge_profile, "Shoulder", (10, shoulder_y - 10), font, font_scale, (128, 128, 128), 1)
        if armpit_width_meters is not None:
            cv2.putText(edge_profile, f"Armpit Width: {armpit_width_meters:.2f}m", 
                       (10, armpit_line_y - 10), font, font_scale, (0, 255, 0), 1)
        if hip_width_meters is not None:
            cv2.putText(edge_profile, f"Hip Width: {hip_width_meters:.2f}m", 
                       (10, hip_line_y - 10), font, font_scale, (255, 0, 0), 1)
        cv2.putText(edge_profile, "Knee", (10, knee_y - 10), font, font_scale, (128, 128, 128), 1)
        
        return edge_profile, armpit_width_meters, hip_width_meters
    else:
        print("No pose landmarks detected in profile image")
        return None
    
def process_profile_image(profile_path, edge_profile_path, armpit_ratio, hip_ratio, mp_pose, pose, actual_height, edge):
    # Load the profile images
    segmented_profile = cv2.imread(profile_path)
    edge_profile = cv2.imread(edge_profile_path)
    
    if segmented_profile is None or edge_profile is None:
        print("Error loading profile images")
        return
    
    # Calculate the scaling factor specific to the profile image
    profile_scaling_factor = calculate_profile_scaling_factor(profile_path, mp_pose, pose, actual_height, edge)
    
    if profile_scaling_factor is None:
        print("Error calculating profile scaling factor")
        return
    
    # Process the profile image with the new scaling factor
    result, armpit_width_meters, hip_width_meters = analyze_profile_image(
        edge_profile, 
        segmented_profile, 
        armpit_ratio, 
        hip_ratio, 
        mp_pose, 
        pose, 
        profile_scaling_factor
    )
        
    return armpit_width_meters, hip_width_meters

def calculElipse(a, b):
    formula1 = math.pi * math.sqrt(2 * (a**2 + b**2))
    formula2 = math.pi * ((3/2)*(a + b) - math.sqrt(a*b))
    perimeter = (formula1 + formula2) / 2
    
    print(f"A : {a}, B : {b}")
    return (formula1 - 0.10)  #Normalement formule 1 rend un résultat + petit que la vérité, et formule 2 plus grand, donc on fait moyenne. Par contre, on a résultat trop grand, donc on retourne formule 1 ici

def draw_point(edge, annotated_image, shoulder_line, left_shoulder_x, right_shoulder_x, wrist_line, left_wrist_x, left_armpit_x, left_armpit_y, right_armpit_x, right_armpit_y, hip_line, hip_col_1, hip_col_2, width, left_syce_x, left_syce_y, right_syce_x, right_syce_y, neck_col_1, neck_col_2, neck_line):
    color = [128, 128, 128]
    color2 = [0, 255, 0]
    R = 5
    thickness = 2
    cv2.circle(edge, [int(left_shoulder_x * width), shoulder_line], radius=R, color=color, thickness=-1)
    cv2.circle(edge, [int(right_shoulder_x * width), shoulder_line], radius=R, color=color, thickness=-1)
    cv2.circle(edge, [int(left_wrist_x * width), wrist_line], radius=R, color=color, thickness=-1)
    cv2.circle(edge, [left_armpit_x, left_armpit_y], radius=R, color=color, thickness=-1)
    cv2.circle(edge, [right_armpit_x, right_armpit_y], radius=R, color=color, thickness=-1)
    cv2.circle(edge, [hip_col_1, hip_line], radius=R, color=color2, thickness=-1)
    cv2.circle(edge, [hip_col_2, hip_line], radius=R, color=color2, thickness=-1)
    cv2.circle(edge, [left_syce_x, left_syce_y], radius=R, color=color, thickness=-1)
    cv2.circle(edge, [right_syce_x, right_syce_y], radius=R, color=color, thickness=-1)
    cv2.circle(edge, [neck_col_1, neck_line], radius=R, color=color, thickness=-1)
    cv2.circle(edge, [neck_col_2, neck_line], radius=R, color=color, thickness=-1)
    cv2.line(edge, (neck_col_1, neck_line), (neck_col_2, neck_line), color2, thickness)
    cv2.line(edge, (hip_col_1, hip_line), (hip_col_2, hip_line), color2, thickness)

    cv2.circle(annotated_image, [int(left_shoulder_x * width), shoulder_line], radius=R, color=color, thickness=-1)
    cv2.circle(annotated_image, [int(right_shoulder_x * width), shoulder_line], radius=R, color=color, thickness=-1)
    cv2.circle(annotated_image, [int(left_wrist_x * width), wrist_line], radius=R, color=color, thickness=-1)
    cv2.circle(annotated_image, [left_armpit_x, left_armpit_y], radius=R, color=color, thickness=-1)
    cv2.circle(annotated_image, [right_armpit_x, right_armpit_y], radius=R, color=color, thickness=-1)
    cv2.circle(annotated_image, [hip_col_1, hip_line], radius=R, color=color2, thickness=-1)
    cv2.circle(annotated_image, [hip_col_2, hip_line], radius=R, color=color2, thickness=-1)
    cv2.circle(annotated_image, [left_syce_x, left_syce_y], radius=R, color=color, thickness=-1)
    cv2.circle(annotated_image, [right_syce_x, right_syce_y], radius=R, color=color, thickness=-1)
    cv2.circle(annotated_image, [neck_col_1, neck_line], radius=R, color=color, thickness=-1)
    cv2.circle(annotated_image, [neck_col_2, neck_line], radius=R, color=color, thickness=-1)
    cv2.line(annotated_image, (neck_col_1, neck_line), (neck_col_2, neck_line), color2, thickness)
    cv2.line(annotated_image, (hip_col_1, hip_line), (hip_col_2, hip_line), color2, thickness)



def print_size(D_neck, L_hip_shoulder, d_inter_armpit, L_bust, half_back, L_arm, L_shoulder_neck, L_shoulder_shoulder, L_hip_armpit, L_hip_neck):
    print('Diamètre du cou: ', D_neck, 'm')
    print('taille bras: ', L_arm, 'm')
    print('longueur inter-epaules: ', L_shoulder_shoulder, 'm')
    print('longueur inter-aisselles: ', d_inter_armpit, 'm')
    print('longueur epaule-cou: ', L_shoulder_neck, 'm')
    print('longueur epaule hanche: ', L_hip_shoulder, 'm')
    print('longueur cou hanche: ', L_hip_neck, 'm')
    print('longueur aisselle-hanche: ', L_hip_armpit, 'm')
    print('hauteur buste: ', L_bust, 'm')
    print('mi dos: ', half_back, 'm')



def print_json(D_neck, L_hip_shoulder, d_inter_armpit, L_bust, half_back, L_arm, L_shoulder_neck, L_shoulder_shoulder, L_hip_armpit, L_hip_neck):
    data = {
        "cou": D_neck,
        "poignet_epaule": L_arm,
        "inter_epaules": L_shoulder_shoulder,
        "inter_aisselles": d_inter_armpit,
        "epaule_cou": L_shoulder_neck,
        "epaule_hanche": L_hip_shoulder,
        "hauteur_cou": L_hip_neck,
        "hauteur_aisselle": L_hip_armpit,
        "tour_de_taille":0,
        "tour_de_buste":0,
        "taille":0
    }

    return data

                                                                            #######    SEGEMENTATION FUNCTIONS      #########


# Disable all Qt logging
os.environ["QT_LOGGING_RULES"] = "*=false"
os.environ["QT_DEBUG_PLUGINS"] = "0"
os.environ["QT_VERBOSE"] = "0"


def convert_dict_files(distances_init) :
    distances_final = {}

    for i in distances_init :
        distances_init[i] = int(distances_init[i]*100)

    # lateral distances are divided by 2 to make a half pattern
    distances_final["cou"] = int(distances_init.get("cou")) / 2
    distances_final["inter_epaules"] = int(distances_init.get("inter_epaules")) / 2
    distances_final["inter_aisselles"] = int(distances_init.get("inter_aisselles")) / 2
    distances_final["tour_de_taille"] = 40 / 2 # change to actual value when json is ready

    distances_final["cou_epaule"] = distances_init.get("epaule_cou")
    distances_final["hauteur_buste"] = distances_init.get("epaule_hanche")
    distances_final["hauteur_cou"] = distances_init.get("hauteur_cou")
    distances_final["hauteur_aisselles"] = distances_init.get("hauteur_aisselle")* 0.8
    distances_final["hauteur_manches"] = distances_init.get("poignet_epaule")

    print(distances_final)

    # ensure all values in the dict are ints
    for value in distances_final:
        distances_final[value] = int(distances_final[value]) * 10

    distances_final["largeur_epaules"] = distances_init.get("poignet_epaule") # this value is random
    distances_final["offset"] = 60
    distances_final["largeur_manches"] = 150

    return distances_final

def invoke_pattern(data, style, sleeves) : 

    distances = convert_dict_files(data)
    
    distances["f-g"] = abs(distances.get("hauteur_buste") - distances.get("hauteur_cou"))
    distances["f-i"] = distances.get("hauteur_buste") - distances.get("hauteur_aisselles")
    distances["f-e"] = distances.get("inter_aisselles") - distances.get("cou")
    distances["f-k"] = abs(distances.get("inter_aisselles") - distances.get("tour_de_taille"))
    distances["f-l"] = abs(distances.get("inter_aisselles") - int(distances.get("tour_de_taille") * 1.4))
    
    print("DEBUG: Final distances ->", distances)

    global patron1, a1  # Use global vars to track instances
    try:
        del patron1, a1  # Delete previous instances if they exist
        print("DEBUG: Deleted previous pattern and affichage instances")
    except NameError:
        pass  # No previous instance exists

    # Create fresh instances
    patron1 = BustPattern(style=style, sleeves=sleeves)
    a1 = Affichage(distances=distances, pattern=patron1, name='pattern')

    print("DEBUG: Created fresh BustPattern and Affichage instances")

    a1.print_pattern()
    a1.save_pattern_image()


if __name__ == '__main__':
    MODEL_PATH = "fichiers_model/sam_vit_h_4b8939.pth"
    GDRIVE_URL = "https://drive.google.com/file/d/1o_GWlR2pruRMCBKmID7sX1zUQM4rVeZj/view?usp=sharing"

    # Check if model exists, if not, download it
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        print("Download complete!")

    # Run the Flask app
    app.run(host='0.0.0.0', port=8080)