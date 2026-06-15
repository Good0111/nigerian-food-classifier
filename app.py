import streamlit as st
import numpy as np
import cv2
import os

# 1. Page Configuration
st.set_page_config(page_title="Nigerian Food Classifier", page_icon="🇳🇬", layout="centered")

CLASS_NAMES = [
    'Abacha and Ugba(african salad)', 'Akara and Eko', 'Amala and Gbegiri- Ewedu', 
    'Asaro', 'Boli(bole)', 'Chin Chin', 'Egusi Soup', 'Ewa-Agoyin', 
    'Fried Plantains (dodo)', 'Jollof Rice', 'Meat-pie', 'Moin-Moin', 
    'Nkwobi', 'Okro Soup', 'Pepper Soup', 'Puff-Puff', 'Suya', 'Vegetable Soup'
]

# Look for the file locally in your repository folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "nigerian_food_model.tflite")

if not os.path.exists(MODEL_PATH):
    st.error("Error: 'nigerian_food_model.tflite' was not found in your repository.")
    st.info("Please upload the real model file using the 'Add file -> Upload files' button on GitHub.")
else:
    try:
        # Load the local uncorrupted model file
        net = cv2.dnn.readNetFromTFLite(MODEL_PATH)

        # 2. User Interface
        st.title("🇳🇬 Nigerian Food Image Classifier")
        st.write("Upload a photo of a Nigerian dish, and MobileNetV2 will identify it.")

        uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_img = cv2.imdecode(file_bytes, 1)
            st.image(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
            
            with st.spinner("Analyzing textures and ingredients... Please wait..."):
                blob = cv2.dnn.blobFromImage(
                    opencv_img, 
                    scalefactor=1.0/127.5, 
                    size=(224, 224), 
                    mean=(127.5, 127.5, 127.5), 
                    swapRB=True, 
                    crop=False
                )
                net.setInput(blob)
                predictions = net.forward()
                
                exp_preds = np.exp(predictions - np.max(predictions))
                probabilities = exp_preds / exp_preds.sum()
                
                max_idx = np.argmax(probabilities)
                predicted_class = CLASS_NAMES[max_idx]
                confidence = probabilities[max_idx] * 100

            if confidence > 40.0:
                st.success(f"### Result: **{predicted_class}**")
                st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")
            else:
                st.warning("⚠️ The model is uncertain about this image.")
                st.write(f"Closest guess: **{predicted_class}** ({confidence:.1f}% confidence)")
                
    except Exception as err:
        st.error(f"An unexpected framework error occurred while initializing OpenCV: {err}")
