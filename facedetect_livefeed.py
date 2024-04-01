import cv2
from facenet_pytorch import MTCNN
import torch
import numpy as np

def get_device():
    """Select CUDA if available, else CPU."""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def initialize_mtcnn(device):
    """Initialize MTCNN with predefined settings."""
    return MTCNN(image_size=160,
              margin=0,
              min_face_size=20,
              thresholds=[0.6, 0.7, 0.7], # MTCNN thresholds
              factor=0.709,
              post_process=True,
              device=device # If you don't have GPU
        )

def np_angle(a, b, c):
    """Calculate the angle between three points."""
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

def detect_and_display(mtcnn, device):
    cap = cv2.VideoCapture(0)  # Initialize the webcam

    while True:
        ret, frame = cap.read()  # Read frame from the webcam
        if not ret:
            break

        # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces in the frame
        boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
        if boxes is not None:
            for box, landmark in zip(boxes, landmarks):
                # Draw the bounding box and landmarks on the frame
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                for point in landmark:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1)

                

                # Calculate angles to determine the pose
                angR = np_angle(landmark[0], landmark[1], landmark[2])
                angL = np_angle(landmark[1], landmark[0], landmark[2])
                #if 35 <= angR <= 57 and 35 <= angL <= 58:
                if 35 <= angR <= 55 and 35 <= angL <= 55:
                    predLabel = 'Frontal'
                elif angR < angL:
                    predLabel = 'Left Profile'
                else:
                    predLabel = 'Right Profile'

                # Display the pose prediction on the frame
                cv2.putText(frame, predLabel, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

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
    detect_and_display(mtcnn, device)

if __name__ == "__main__":
    main()
