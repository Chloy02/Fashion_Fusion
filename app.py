import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Load the pre-trained MobileNetV2 model from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4"
model = hub.load(MODEL_URL)

# Load ImageNet labels to map model predictions
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", LABELS_URL)
imagenet_labels = np.array(open(labels_path).read().splitlines())

# Streamlit UI
st.title("👗 FashionFusion: Your Personal Wardrobe Stylist")
st.write("Upload an image of a clothing item, and we'll classify it for you!")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to 224x224 (MobileNetV2 input size)
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Perform classification when an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and classify the image
    processed_image = preprocess_image(image)
    predictions = model(processed_image)
    
    # Get the highest probability class
    predicted_class = np.argmax(predictions)
    clothing_item = imagenet_labels[predicted_class]

    # Display the classification result
    st.subheader(f"🛍️ Detected Clothing Item: **{clothing_item}**")
    st.write("Prediction Confidence:", round(np.max(predictions) * 100, 2), "%")

