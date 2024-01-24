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
x = image.img_to_array(img)

# Add a fourth dimension (for batch size), which Keras expects. Here, we're
# adding a batch dimension of 1 since we are predicting for one image.
x = np.expand_dims(x, axis=0)

# Preprocess the input by subtracting the mean RGB channels of the ImageNet dataset.
# This mean is subtracted because the model was originally trained with this preprocessing.
x = resnet50.preprocess_input(x)

# Run the image through the neural network to make a prediction.
# This outputs the probabilities for all ImageNet classes.
predictions = model.predict(x)

# Decode the prediction into human-readable class names with their probabilities.
# 'top=5' gives us the top 5 predictions that the model has made for the image.
predicted_classes = resnet50.decode_predictions(predictions, top=5)[0]

# Print out the predictions.
for i, (imagenet_id, name, score) in enumerate(predicted_classes):
    print(f"{i + 1}: {name} ({score:.2f})")
