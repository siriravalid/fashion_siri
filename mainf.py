import streamlit as st
import pickle
import os
import numpy as np
from PIL import Image
import keras

# Load the model
model_path = os.path.join(os.getcwd(), 'model.sav')
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error(f'Model file not found at path: {model_path}')
    st.stop()
except Exception as e:
    st.error(f'Error loading the model: {e}')
    st.stop()

# Preprocess the image
def preprocess_image(image):
    image = image.convert('L')  # Convert image to grayscale
    image = image.resize((28, 28))  # Resize image to 28x28
    image = np.array(image)  # Convert image to array
    image = image / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

# Streamlit UI
st.title('Fashion MNIST Image Classification')
st.write('Upload an image to classify it.')

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make prediction
    try:
        prediction = model.predict(preprocessed_image)
    except Exception as e:
        st.error(f'Error making prediction: {e}')
        st.stop()
    
    # Map prediction to class label
    class_labels = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    predicted_class = class_labels[np.argmax(prediction)]
    
    st.write(f'This image is a: {predicted_class}')
