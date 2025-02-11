import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained Keras model
@st.cache_resource
def load_model():
    model_path = "C://Users//meich//OneDrive - Asia Pacific University//Machine Learning and Data Science Project//Potato Disease Classification//saved_models//1.keras"  # Make sure the model file is in the correct directory
    return tf.keras.models.load_model(model_path)

model = load_model()

# Define class labels based on training dataset (Update according to your model)
CLASS_NAMES = ["Healthy", "Early Blight", "Late Blight"]  # Adjust as per your model's classes

# Streamlit App UI
st.title("🥔 Potato Leaf Disease Detection")
st.write("Upload an image of a potato leaf to detect disease!")

# File uploader for image input
uploaded_file = st.file_uploader("📂 Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    def preprocess_image(img):
        img = img.resize((256, 256))  # Resize to match model's input size
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    # Process the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    # Display Prediction Results
    st.subheader("🔍 Prediction Result")
    st.write(f"**Detected:** {predicted_class}")

    # Display confidence scores
    st.write("📊 Confidence Scores:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")

st.info("Ensure the uploaded image is clear and properly formatted.")
