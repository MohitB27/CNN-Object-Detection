import os
import App
import numpy as np
import cv2 as cv
import PIL
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image

path=r"C:\Users\mkbho\PycharmProjects\pythonProject1\dataset\Train"

batch_size = 32
img_height = 180
img_width = 180


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder, filename), target_size=(32, 32))
        img = img_to_array(img)
        images.append(img)
    return np.array(images)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(32, 32))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    return img


class Model:

    def __init__(self):
        self.model = tf.keras.models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
            # Binary classification, so output is 1 neuron with sigmoid activation
        ])

    def train_model(self, counters, dataset=r"C:\Users\mkbho\PycharmProjects\pythonProject1\dataset"):
        # Load images from two folders
        folder1 = r'C:\Users\mkbho\PycharmProjects\pythonProject1\dataset\1'
        folder2 = r'C:\Users\mkbho\PycharmProjects\pythonProject1\dataset\2'

        images1 = load_images_from_folder(folder1)
        images2 = load_images_from_folder(folder2)

        # Create labels for the images
        labels1 = np.zeros(len(images1))  # Assuming all images in folder1 belong to class 0
        labels2 = np.ones(len(images2))  # Assuming all images in folder2 belong to class 1

# Concatenate images and labels from both folders
        all_images = np.concatenate((images1, images2), axis=0)
        all_labels = np.concatenate((labels1, labels2), axis=0)

# Shuffle the data
        indices = np.arange(all_images.shape[0])
        np.random.shuffle(indices)
        all_images = all_images[indices]
        all_labels = all_labels[indices]

# Normalize pixel values to be between 0 and 1
        all_images = all_images / 255.0

        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',  # For binary classification
                      metrics=['accuracy'])

        self.model.fit(
            all_images, all_labels,
            epochs=10,
            batch_size=64,
            validation_split=0.2)
        print('Model Trained Successfully!!')

    def predict(self, frame):

        frame = frame[1]
        cv.imwrite('frame.jpg', cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open('frame.jpg')
        #img.save('frame.jpg', target_size=(32, 32))

        img = image.load_img(r'C:\Users\mkbho\PycharmProjects\pythonProject1\frame.jpg', target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize pixel values
        prediction = self.model.predict(img_array)
        predicted_class = "Class 1" if prediction[0][0] > 0.5 else "Class 0"
        confidence = prediction[0][0] if predicted_class == "Class 1" else 1 - prediction[0][0]
        print(predicted_class)
        print(confidence)
        return predicted_class
        #return prediction


