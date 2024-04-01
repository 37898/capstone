import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt

# Optional: Suppress oneDNN optimization warnings if desired (may affect performance)
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define image size
IMAGE_SIZE = [224, 224]

# Paths to your training and validation (test) data
train_path = r"C:\Users\abhij\Masters\Spring 2024\Capstone_Project\dataset_faces\train"
valid_path = r"C:\Users\abhij\Masters\Spring 2024\Capstone_Project\dataset_faces\test"

# Initialize VGG16 with ImageNet weights
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze the pretrained layers to prevent them from being updated during training
for layer in vgg.layers:
    layer.trainable = False

# Count the number of classes by counting folders in the training path
folders = glob(train_path + '/*')

# Add custom layers on top of VGG
x = Flatten()(vgg.output)
# The prediction layer's units should equal the number of folders/classes
prediction = Dense(len(folders), activation='softmax')(x)

# Create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Define data augmentation configuration with ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Prepare data loaders
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

# Train the model
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,  # Adjust number of epochs according to your needs
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Plot training and validation loss and accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy')
plt.legend()

plt.savefig('training_results.png')
plt.show()

# Save the model
model.save('final_model.h5')
