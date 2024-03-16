import cv2
import os

# Initialize the Haar Cascade face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

def create_dataset_for_class(class_name, num_images=30, train_ratio=0.8):
    cap = cv2.VideoCapture(0)
    count = 0
    train_count = int(num_images * train_ratio)
    
    train_dir = f'dataset_faces/train/{class_name}'
    test_dir = f'dataset_faces/test/{class_name}'
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            if count <= train_count:
                file_path = f'{train_dir}/{class_name}_{count}.jpg'
            else:
                file_path = f'{test_dir}/{class_name}_{count - train_count}.jpg'
                
            cv2.imwrite(file_path, face)
            cv2.putText(face, f'{count}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Capturer', face)

        if cv2.waitKey(1) == 13 or count == num_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Dataset collection for class {class_name} complete!")

# Example usage
class_names = ['Jan','Abhi','Bala'] # Add more class names as needed
num_images_per_class = 150 # Adjust as per your requirement

for class_name in class_names:
    print(f"Collecting images for {class_name}. Please get ready!")
    create_dataset_for_class(class_name, num_images=num_images_per_class)
    print(f"Collection for {class_name} completed.")
