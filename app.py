import streamlit as st
import numpy as np
from PIL import Image
import os

# Safe import for TFLite runtime to avoid TensorFlow installation crashes
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

# 1. Page Configuration
st.set_page_config(page_title="Nigerian Food Classifier", page_icon="🇳🇬", layout="centered")

# Your exact 18 food classes in alphabetical order
CLASS_NAMES = [
    'Abacha and Ugba(african salad)', 'Akara and Eko', 'Amala and Gbegiri- Ewedu', 
    'Asaro', 'Boli(bole)', 'Chin Chin', 'Egusi Soup', 'Ewa-Agoyin', 
    'Fried Plantains (dodo)', 'Jollof Rice', 'Meat-pie', 'Moin-Moin', 
    'Nkwobi', 'Okro Soup', 'Pepper Soup', 'Puff-Puff', 'Suya', 'Vegetable Soup'
]

MODEL_PATH = "nigerian_food_model.tflite"

if not os.path.exists(MODEL_PATH):
    st.error(f"Error: '{MODEL_PATH}' not found! Place the file in the same folder.")
else:
    # Load TFLite Model
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. User Interface
    st.title("🇳🇬 Nigerian Food Image Classifier")
    st.write("Upload a photo of a Nigerian dish, and MobileNetV2 will identify it.")

    uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

    # 3. Processing & Prediction Pipeline
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Analyzing textures and ingredients... Please wait..."):
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Match the standard MobileNetV2 scaling (-1 to 1)
            img_array = (img_array / 127.5) - 1.0
            
            interpreter.set_tensor(input_details['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details['index'])
            
            # Normalise scores
            exp_preds = np.exp(predictions - np.max(predictions))
            probabilities = exp_preds / exp_preds.sum()
            
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

