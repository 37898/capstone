import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from facenet_pytorch import MTCNN
import torch
import cv2
from PIL import Image
import numpy as np


def get_device():
    #Select CUDA if available, else CPU.
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Initializing the MTCNN with all the initial setups
def initialize_mtcnn(device):
    return MTCNN(image_size=160, margin=0, min_face_size=20,
                 thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

def extract_features_labels(directory, mtcnn):
    features = []
    labels = []
    for label_folder in os.listdir(directory):
        folder_path = os.path.join(directory, label_folder)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # MTCNN detection
            _, probs, landmarks = mtcnn.detect(image, landmarks=True)
            
            # Check if landmarks were detected and the detection is confident
            if landmarks is not None and np.any(probs > 0.9):
                # Assuming consistent detection, all images should have landmarks flattened to the same length
                landmarks_flat = landmarks[0].flatten()  # Use only the first detected face for simplicity
                #print(landmarks_flat)
                features.append(landmarks_flat)
                labels.append(label_folder)
            else:
                print(f"No confident landmarks detected for {image_name}, skipped.")
    
    return np.array(features, dtype=object), np.array(labels)


def train_classifier(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    return classifier

device = get_device()
mtcnn = initialize_mtcnn(device)

# Paths to your dataset folders
train_dir = r'C:\Users\abhij\Masters\Spring 2024\Capstone_Project\dataset_faces\train'
test_dir = r'C:\Users\abhij\Masters\Spring 2024\Capstone_Project\dataset_faces\test'

# Extract features and labels
train_features, train_labels = extract_features_labels(train_dir, mtcnn)

# Train the classifier
classifier = train_classifier(train_features, train_labels)


# Function to predict labels of the test dataset and evaluate the model
def predict_and_evaluate(test_dir, classifier, mtcnn):
    # Extract features and labels from the test dataset
    test_features, test_labels = extract_features_labels(test_dir, mtcnn)
    
    # Predict using the trained classifier
    test_predictions = classifier.predict(test_features)
    
    # Evaluate the predictions
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test Accuracy: {accuracy}")
    
    # Detailed classification report
    print("Classification Report:")
    print(classification_report(test_labels, test_predictions))

# Predict and evaluate on the test set
predict_and_evaluate(test_dir, classifier, mtcnn)


#Individual Image test
#Assuming 'train_labels' contains folder names.
unique_labels = np.unique(train_labels)
label_encodings = {label: index for index, label in enumerate(unique_labels)}



def classify_single_image(image_path, classifier, mtcnn, reverse_label_encodings):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Extract landmarks
    _, probs, landmarks = mtcnn.detect(image, landmarks=True)
    if landmarks is not None and np.any(probs > 0.9):
        landmarks_flat = landmarks[0].flatten()
        print(landmarks_flat)
        # Predict the class of the image
        prediction_index = classifier.predict([landmarks_flat])[0]  # Predict expects a list of samples
        print(prediction_index)
        #predicted_label = reverse_label_encodings[prediction_index]
        print(f"The image {os.path.basename(image_path)} is classified as: {prediction_index}")
    else:
        print(f"No confident landmarks detected for {os.path.basename(image_path)}, unable to classify.")



#path for the file
image_path_to_classify = r'C:\Users\abhij\OneDrive\Pictures\Camera Roll\WIN_20240401_01_43_55_Pro.jpg'
#Calling the function with reverser label.
classify_single_image(image_path_to_classify, classifier, mtcnn, label_encodings)



def live_classification(classifier, mtcnn, reverse_label_encodings):
    cap = cv2.VideoCapture(0)  # Start video capture

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break
        
        # Convert frame to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect face
        _, probs, landmarks = mtcnn.detect(img, landmarks=True)
        
        # If a face is detected with high confidence
        if landmarks is not None and np.any(probs > 0.9):
            landmarks_flat = landmarks[0].flatten()
            prediction_index = classifier.predict([landmarks_flat])[0]
            #predicted_label = reverse_label_encodings.get(prediction_index, "Unknown")
            
            # Display the classification result on the frame
            cv2.putText(frame, prediction_index, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show the frame
        cv2.imshow('Live Classification', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start live classification
live_classification(classifier, mtcnn, label_encodings)
