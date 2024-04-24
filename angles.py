import cv2
from facenet_pytorch import MTCNN
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook

train_dir = 'C:/Users/abhij/Masters/Spring 2024/Capstone_Project/dataset_faces/train/'

def get_device():
    """Select CUDA if available, else CPU."""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def initialize_mtcnn(device):
    """Initialize MTCNN with predefined settings."""
    return MTCNN(image_size=160,
                 margin=0,
                 min_face_size=20,
                 thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds
                 factor=0.709,
                 post_process=True,
                 device=device)

def np_angle(a, b, c):
    """Calculate the angle between three points."""
    ba, bc = a - b, c - b
    cosine_angle = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def train_images_angles(mtcnn, device, path):
    angR_list_Frontal, angL_list_Not_Frontal = [], []
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"Processing class: {class_name}")

        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(class_path, file)
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Could not read image {file_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)

                if landmarks is not None and len(landmarks[0]) == 5:
                    for landmark in landmarks:
                        angR = np_angle(landmark[0], landmark[1], landmark[2])
                        angL = np_angle(landmark[1], landmark[0], landmark[2])
                        if 35 <= angR <= 65 and 35 <= angL <= 65:
                            angR_list_Frontal.append(angR)
                            angL_list_Not_Frontal.append(None)  # Append None for non-frontal
                        else:
                            angL_list_Not_Frontal.append(angL)
                            angR_list_Frontal.append(None)  # Append None for frontal
                else:
                    print(f"No landmarks found for image {file_path}")
                    angR_list_Frontal.append(None)
                    angL_list_Not_Frontal.append(None)

    # Make sure both lists are of the same length by appending None
    max_length = max(len(angR_list_Frontal), len(angL_list_Not_Frontal))
    angR_list_Frontal.extend([None] * (max_length - len(angR_list_Frontal)))
    angL_list_Not_Frontal.extend([None] * (max_length - len(angL_list_Not_Frontal)))

    df = pd.DataFrame({
        'Angles_Frontal': angR_list_Frontal,
        'Angles_NonFrontal': angL_list_Not_Frontal
    })

    # Save the DataFrame to an Excel file
    df.to_excel('output.xlsx', index=False, engine='openpyxl')



def main():
    device = get_device()
    mtcnn = initialize_mtcnn(device)
    train_images_angles(mtcnn, device, train_dir)

if __name__ == "__main__":
    main()
