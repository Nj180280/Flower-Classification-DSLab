# Create a new Python file for the Streamlit app
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = load_model('flower_classification_model.h5')

# Class labels (make sure these match your original classes)
class_labels = ['Lily', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

st.title('Flower Classification App')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image and make prediction
    processed_image = preprocess_image(img)
    prediction = model.predict(processed_image)
    
    # Get the predicted class
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Display bar chart of predictions
    st.bar_chart(dict(zip(class_labels, prediction[0] * 100)))