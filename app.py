import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import pytesseract
import cv2
import numpy as np
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Label mapping
ind = {0: 'handwritten', 1: 'printed'}

class CNN_MODEL:
    def __init__(self, inp_size, no_of_channels):
        self.inp_size = inp_size
        self.no_of_channels = no_of_channels
    
    # Function to enhance image
    def enhance_image(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return image, binary

    # Function to classify and extract text
    def classify_img(self, image_path, model):
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.inp_size)
        if self.no_of_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image = np.expand_dims(image, axis=-1)  # add channel dimension for grayscale images
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        prediction = model.predict(image)
        predicted_class = 'printed' if prediction[0][0] > 0.5 else "handwritten"
        return predicted_class
    
    

class TransFormers:
    def __init__(self) -> None:
        pass
    # Load the trained model
    model = ViTForImageClassification.from_pretrained('./vit-handwritten-printed')
    # Load the feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    # Function to preprocess the image for classification
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs['pixel_values']

    # Function to make predictions
    def predict(self, image_path):
        self.model.eval()
        pixel_values = self.preprocess_image(image_path)
        with torch.no_grad():
            outputs = self.model(pixel_values)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        return predicted_class_idx

# Load the models
model_tl = load_model('tl_model.h5')
model_custom = load_model('custom_modell.h5')
tl_obj = TransFormers()


st.title("Handwritten vs Printed Text Classification")
st.write(f"Custom Model Accuracy: {0.94 * 100:.2f}%")
st.write(f"TL Model Accuracy: {0.98 * 100:.2f}%")
st.write(f"Transformer Accuracy: {1 * 100:.2f}%")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_path = f"temp_{uploaded_file.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    _, binary = CNN_MODEL(None, None).enhance_image(image_path)
    pil_image = Image.fromarray(binary)
    text = pytesseract.image_to_string(pil_image, config='--psm 6')
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Please note that most accurate predictions are from transformers model")
    
    # Classify and extract text button
    if st.button("Classify and Extract Text"):

        ############################################ CNN Model (Custom)###################################
        predicted_class = CNN_MODEL(inp_size=(32, 32), no_of_channels=1).classify_img(image_path, model_custom)
        st.header("Custom CNN Model Predictions")
        if predicted_class is not None:
            st.write(f"**Predicted Class:** {predicted_class}")
            st.write("**Recognized Text**")
            st.write(text)
        else:
            st.error("Could not classify and extract text from the image.")
        
         ############################################ CNN Model (transfer Learning)###################################
        predicted_class = CNN_MODEL(inp_size=(75, 75), no_of_channels=3).classify_img(image_path, model_tl)
        st.header("Transfer Learning Model Predictions")
        if predicted_class is not None:
            st.write(f"**Predicted Class:** {predicted_class}")
            st.write("**Recognized Text**")
            st.write(text)
        else:
            st.error("Could not classify and extract text from the image.")

        ############################################Transformers #####################################################
        st.header("Transformers predictions")
        class_labels = {0: 'handwritten', 1: 'printed'}
        predicted_class_idx = tl_obj.predict(image_path)
        predicted_class = class_labels[predicted_class_idx]
        # Extract text using Tesseract
        # extracted_text = tl_obj.extract_text(image_path)
        
        if predicted_class is not None:
            st.write(f"**Predicted Class:** {predicted_class}")
            st.write("**Extracted Text:**")
            st.write(text)
        else:
            st.error("Could not classify and extract text from the image.")
