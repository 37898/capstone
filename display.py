from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Ensure the model path is correct and use a raw string if necessary
model_path = r"C:\Users\abhij\Masters\Spring 2024\Capstone_Project\final_model.h5"
model = load_model(model_path)

# Ensure the cascade path is correct and use a raw string if necessary
cascade_path = r"C:\Users\abhij\Masters\Spring 2024\Capstone_Project\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x,y,w,h) in faces:
        # Adding rectangle around your face
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()

    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        
        # Converting to a format suitable for the model
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)

        name = "None matching"

        # Update these conditions based on how your model predicts
        if pred[0][0] > 0.6:
            name = 'Abhi'
        if pred[0][1] > 0.6:
            name = 'Jan'
            
        if pred[0][2] > 0.6:
            name = 'Bala'  
        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
