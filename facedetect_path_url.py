import argparse
import math
import numpy as np
import requests
from PIL import Image
import torch
from facenet_pytorch import MTCNN
import cv2

def get_device():
    """Select CUDA if available, else CPU."""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def initialize_mtcnn(device):
    """Initialize MTCNN with predefined settings."""
    return MTCNN(image_size=160, margin=0, min_face_size=20,
                 thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

def load_image(path=None, url=None):
    """Load an image from a file path or URL."""
    if path:
        image = Image.open(path)
    elif url:
        response = requests.get(url, stream=True)
        image = Image.open(response.raw)
    else:
        raise ValueError("Either path or url must be provided")
    if image.mode != "RGB":
        image = image.convert('RGB')
    return image

def np_angle(a, b, c):
    """Calculate the angle between three points."""
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

def predict_face_pose(mtcnn, image):
    bbox, prob, landmarks = mtcnn.detect(image, landmarks=True)
    results = {'landmarks': [], 'angle_R': [], 'angle_L': [], 'predLabel': []}
    if bbox is not None:
        for b, lm, p in zip(bbox, landmarks, prob):
            if p > 0.9:
                angR = np_angle(lm[0], lm[1], lm[2])
                angL = np_angle(lm[1], lm[0], lm[2])
                if 35 <= angR <= 57 and 35 <= angL <= 58:
                    predLabel = 'Frontal'
                elif angR < angL:
                    predLabel = 'Left Profile'
                else:
                    predLabel = 'Right Profile'
                results['landmarks'].append(lm)
                results['angle_R'].append(angR)
                results['angle_L'].append(angL)
                results['predLabel'].append(predLabel)
            else:
                print('Detected face below confidence threshold.')
    else:
        print('No face detected in the image.')
    return results

def main():
    parser = argparse.ArgumentParser(description="Face pose detection for one face")
    parser.add_argument("-p", "--path", help="Image path", type=str)
    parser.add_argument("-u", "--url", help="Image URL", type=str)
    args = parser.parse_args()

    device = get_device()
    print(f'Running on device: {device}')
    mtcnn = initialize_mtcnn(device)

    if args.path or args.url:
        try:
            image = load_image(path=args.path, url=args.url)
            results = predict_face_pose(mtcnn, image)
            # Print results to console
            for i, label in enumerate(results['predLabel']):
                print(f"Face {i+1}:")
                print(f"  Pose: {label}")
                print(f"  Right Eye Angle: {results['angle_R'][i]:.2f}")
                print(f"  Left Eye Angle: {results['angle_L'][i]:.2f}")
                print(f"  Landmarks: {results['landmarks'][i]}")
            print('Done detecting face pose.')
        except Exception as e:
            print(f"Error loading image: {e}")
    else:
        print("No image source provided. Please specify either a path or URL.")


if __name__ == "__main__":
    main()