import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Ensure the model file path is correct
model_path = 'model.sav'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

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
    prediction = model.predict(preprocessed_image)
    
    # Map prediction to class label
    class_labels = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    predicted_class = class_labels[np.argmax(prediction)]
    
    st.write(f'This image is a: {predicted_class}')
