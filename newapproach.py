import cv2
from facenet_pytorch import MTCNN
import torch
import numpy as np
from torchvision import models, transforms

def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def initialize_mtcnn(device):
    return MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

def np_angle(a, b, c):
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

def load_classification_model(device):
    model = models.resnet50(pretrained=True)  # Load a pre-trained ResNet model
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image_for_classification(image, device):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension and send to device
    return image

def classify_image(model, image, device):
    with torch.no_grad():  # No need to compute gradients
        image = preprocess_image_for_classification(image, device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
        return predicted.item()

def detect_and_display(mtcnn, classification_model, device):
    cap = cv2.VideoCapture(0)
    # Add other initializations here...

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
        # Add your existing face detection and pose estimation code here...

        # For demonstration, classifying the whole frame for each detection:
        if boxes is not None:
            class_id = classify_image(classification_model, frame_rgb, device)
            # Display the classification result on the frame
            cv2.putText(frame, f"Class ID: {class_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the frame
        cv2.imshow('Face Pose Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

def main():
    device = get_device()
    mtcnn = initialize_mtcnn(device)
    classification_model = load_classification_model(device)
    detect_and_display(mtcnn, classification_model, device)

if __name__ == "__main__":
    main()
