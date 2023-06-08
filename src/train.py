import os
import platform
import sys

import numpy as np
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

gpu_devices = tf.config.list_physical_devices('GPU')
if len(gpu_devices) > 0:
    tf.config.experimental.set_visible_devices(gpu_devices[0], 'GPU')
    # If you have multiple GPUs and want to use a specific one, you can specify its index instead of `gpu_devices[0]`
else:
    print("No GPU devices found.")

# Set the path to the data directory
data_dir = "../data"

# Set the image and annotation file paths
image_dir = os.path.join(data_dir, "images")
annotation_file = os.path.join(data_dir, "encoded_pixels.csv")

# Load the annotations from the CSV file
annotations = pd.read_csv(annotation_file)

# Use only part of the annotations to train the model
annotations = annotations[:500]

# Split the data into training and validation sets
train_annotations, val_annotations = train_test_split(annotations, test_size=0.2)

# Define the UNet model architecture
inputs = keras.Input(shape=(256, 256, 3))
conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv1)
pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(pool1)
conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv2)
pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = layers.Conv2D(256, 3, activation="relu", padding="same")(pool2)
conv3 = layers.Conv2D(256, 3, activation="relu", padding="same")(conv3)
pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = layers.Conv2D(512, 3, activation="relu", padding="same")(pool3)
conv4 = layers.Conv2D(512, 3, activation="relu", padding="same")(conv4)
pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = layers.Conv2D(1024, 3, activation="relu", padding="same")(pool4)
conv5 = layers.Conv2D(1024, 3, activation="relu", padding="same")(conv5)

up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding="same")(conv5)
concat6 = layers.concatenate([conv4, up6], axis=3)
conv6 = layers.Conv2D(512, 3, activation="relu", padding="same")(concat6)
conv6 = layers.Conv2D(512, 3, activation="relu", padding="same")(conv6)

up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding="same")(conv6)
concat7 = layers.concatenate([conv3, up7], axis=3)
conv7 = layers.Conv2D(256, 3, activation="relu", padding="same")(concat7)
conv7 = layers.Conv2D(256, 3, activation="relu", padding="same")(conv7)

up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding="same")(conv7)
concat8 = layers.concatenate([conv2, up8], axis=3)
conv8 = layers.Conv2D(128, 3, activation="relu", padding="same")(concat8)
conv8 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv8)

up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding="same")(conv8)
concat9 = layers.concatenate([conv1, up9], axis=3)
conv9 = layers.Conv2D(64, 3, activation="relu", padding="same")(concat9)
conv9 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv9)
conv9 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv9)
conv9 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv9)

conv10 = layers.Conv2D(1, 1, activation="sigmoid")(conv9)

# Create the model
model = keras.Model(inputs=inputs, outputs=conv10)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Define a function to load and preprocess images
def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return image

# Define a function to generate data batches
def data_generator(batch_size, annotations):
    num_samples = len(annotations)
    while True:
        batch_indices = np.random.choice(num_samples, size=batch_size, replace=False)
        batch_annotations = annotations.iloc[batch_indices]

        batch_images = []
        batch_masks = []
        for _, annotation in batch_annotations.iterrows():
            image_path = os.path.join(image_dir, annotation["ImageId"])
            mask_pixels = annotation["EncodedPixels"]

            image = load_image(image_path)
            mask = create_mask(mask_pixels)

            batch_images.append(image)
            batch_masks.append(mask)

        yield np.array(batch_images), np.array(batch_masks)

# Define a function to create masks from encoded pixels
def create_mask(encoded_pixels):
    mask = np.zeros((256, 256))  # Initialize an empty mask

    if pd.isna(encoded_pixels) or encoded_pixels == "":
        return mask  # Return an empty mask if there are no encoded pixels

    # Split the encoded pixels by space
    pixels = encoded_pixels.split(" ")

    # Iterate over pairs of pixel values (start, length)
    for i in range(0, len(pixels), 2):
        start = int(pixels[i]) - 1  # Subtract 1 to convert to 0-based indexing
        length = int(pixels[i + 1])

        # Calculate the end position
        end = start + length

        # Decode the pixel positions and set them to 1 in the mask
        mask[start:end] = 1

    return mask

# Train the model
batch_size = 10
epochs = 1
steps_per_epoch = len(train_annotations) // batch_size

train_generator = data_generator(batch_size, train_annotations)
val_generator = data_generator(batch_size, val_annotations)

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=steps_per_epoch // 5
)

# Save the trained model
model.save("unet_model.h5")