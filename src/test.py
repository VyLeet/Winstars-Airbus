import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras

# Set the path to the data directory
data_dir = "../data"

# Set the image and annotation file paths
image_dir = os.path.join(data_dir, "images")
annotation_file = os.path.join(data_dir, "encoded_pixels.csv")

# Load the annotations from the CSV file
annotations = pd.read_csv(annotation_file)

# Use only part of the annotations for testing
test_annotations = annotations[:10]

# Load the trained model
model = keras.models.load_model("unet_model.h5")

# Define a function to load and preprocess test images
def load_test_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return image

# Define a function to perform inference on test images
def test_model(model, test_annotations):
    for _, annotation in test_annotations.iterrows():
        image_path = os.path.join(image_dir, annotation["ImageId"])
        image = load_test_image(image_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Perform inference
        prediction = model.predict(image)

        # Convert prediction to binary mask
        prediction = (prediction > 0.5).astype(np.uint8)

        # Calculate dice-score
        intersection = np.sum(prediction * annotation["EncodedPixels"])
        union = np.sum(prediction) + np.sum(annotation["EncodedPixels"])
        dice_score = (2 * intersection) / (union + 1e-7)  # Add epsilon to avoid division by zero
        dice_scores.append(dice_score)

    # Print the dice scores
    for i, dice_score in enumerate(dice_scores):
        print(f"Dice score for image {i+1}: {dice_score}")

# Perform inference on test images
test_model(model, test_annotations)