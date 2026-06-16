            import streamlit as st
import numpy as np
import cv2
import os

# 1. Page Configuration
st.set_page_config(page_title="Nigerian Food Classifier", page_icon="🇳🇬", layout="centered")

# Your exact 18 food classes
CLASS_NAMES = [
    'Abacha and Ugba(african salad)', 'Akara and Eko', 'Amala and Gbegiri- Ewedu', 
    'Asaro', 'Boli(bole)', 'Chin Chin', 'Egusi Soup', 'Ewa-Agoyin', 
    'Fried Plantains (dodo)', 'Jollof Rice', 'Meat-pie', 'Moin-Moin', 
    'Nkwobi', 'Okro Soup', 'Pepper Soup', 'Puff-Puff', 'Suya', 'Vegetable Soup'
]

# DIRECT ROOT PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "nigerian_food_model.tflite")

if not os.path.exists(MODEL_PATH):
    st.error("Error: 'nigerian_food_model.tflite' was not found in your repository.")
    st.info("Please ensure your model file is uploaded directly into your repository's main folder.")
else:
    try:
        # Load TFLite Model via OpenCV natively
        net = cv2.dnn.readNetFromTFLite(MODEL_PATH)

        # 2. User Interface
        st.title("🇳🇬 Nigerian Food Image Classifier")
        st.write("Upload a photo of a Nigerian dish, and MobileNetV2 will identify it.")

        uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

        # 3. Processing & Prediction Pipeline
        if uploaded_file is not None:
            # Read image directly into OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_img = cv2.imdecode(file_bytes, 1)
            
            # Display uploaded image safely converting BGR to RGB layout
            st.image(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
            
            with st.spinner("Analyzing textures and ingredients... Please wait..."):
                # OpenCV handles rescaling (-1 to 1) and size transformations using blobFromImage
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
                
                # FIXED: Flatten the prediction layout to handle OpenCV's shape output safely
                flat_preds = predictions.flatten()
                
                # Normalise output layers using standard Softmax
                exp_preds = np.exp(flat_preds - np.max(flat_preds))
                probabilities = exp_preds / exp_preds.sum()
                
                # Extract the class mapping index directly from the flattened array
                max_idx = np.argmax(probabilities)
                predicted_class = CLASS_NAMES[max_idx]
                confidence = probabilities[max_idx] * 100

            # 4. Display Output
            if confidence > 40.0:
                st.success(f"### Result: **{predicted_class}**")
                st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")
            else:
                st.warning("⚠️ The model is uncertain about this image.")
                st.write(f"Closest guess: **{predicted_class}** ({confidence:.1f}% confidence)")
                
    except Exception as err:
        st.error(f"An unexpected framework error occurred while initializing OpenCV: {err}")
