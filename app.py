import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Pro Digit Recognizer", layout="centered")

# --- 1. LOAD CNN MODEL ---
@st.cache_resource
def load_cnn_model():
    # This must match the CNN structure from your Colab exactly
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Path to your new CNN weights file
    weights_path = 'cnn_digit_weights.weights.h5'
    
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            return model
        except Exception as e:
            st.error(f"Error loading weights: {e}")
            return None
    return None

model = load_cnn_model()

# --- 2. USER INTERFACE ---
st.title("🖊️ Advanced Digit Recognizer")
st.markdown("This model uses **CNN** for better accuracy. Draw clearly in the center.")

st.sidebar.title("Live Preview")

if model is not None:
    # Drawing Canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    # --- 3. PREDICTION LOGIC ---
    if st.button("Analyze Drawing"):
        if canvas_result.image_data is not None:
            # Step A: Convert to Grayscale
            raw_img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
            
            # Step B: Auto-Centering (Bounding Box)
            bbox = raw_img.getbbox()
            if bbox:
                # Crop to the digit and add padding to mimic MNIST style
                cropped_img = raw_img.crop(bbox)
                img = ImageOps.expand(cropped_img, border=40, fill=0)
            else:
                img = raw_img

            # Step C: Resize to 28x28
            img = img.resize((28, 28))
            
            # Step D: Show Preview in Sidebar
            st.sidebar.image(img, caption="Processed Image", width=150)
            
            # Step E: Normalize and Reshape for CNN (Batch, H, W, Channel)
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            # Step F: Predict
            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)

            # Show Result
            st.success(f"## Prediction: {predicted_digit}")
            st.write(f"**Confidence Score:** {confidence*100:.2f}%")
            st.bar_chart(prediction[0])
        else:
            st.warning("Please draw a digit first!")
else:
    st.error("Model weights not found! Please ensure 'cnn_digit_weights.weights.h5' is in your folder.")