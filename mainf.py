import streamlit as st
import pickle
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
with open('model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a function to preprocess the image
def preprocess_image(image):
    # Convert image to grayscale and resize it to 28x28
    image = image.convert('L').resize((28, 28))
    image_array = np.array(image).reshape(1, -1)
    # Normalize the image
    scaler = StandardScaler()
    image_array = scaler.fit_transform(image_array)
    return image_array

# Streamlit app
st.title("Fashion MNIST Image Classifier")

st.write("Upload an image to classify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image_array = preprocess_image(image)
    
    # Predict the class
    prediction = model.predict(image_array)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    st.write(f'This image is predicted to be a: {class_names[prediction[0]]}')
