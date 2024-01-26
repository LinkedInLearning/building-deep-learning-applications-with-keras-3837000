import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing import image
from keras.applications import resnet50

# Initialize the ResNet50 model pre-loaded with weights trained on ImageNet
model = resnet50.ResNet50(weights='imagenet')

# Load an image file, resizing it to 224x224 pixels (size required by ResNet50 model)
img = image.load_img("input/image/woman.jpeg", target_size=(224, 224))

# Convert the image to a numpy array which is the format Keras requires


# Add a fourth dimension (for batch size), which Keras expects. Here, we're
# adding a batch dimension of 1 since we are predicting for one image.


# Preprocess the input by subtracting the mean RGB channels of the ImageNet dataset.
# This mean is subtracted because the model was originally trained with this preprocessing.


# Run the image through the neural network to make a prediction.
# This outputs the probabilities for all ImageNet classes.


# Decode the prediction into human-readable class names with their probabilities.
# 'top=5' gives us the top 5 predictions that the model has made for the image.


# Print out the predictions.

