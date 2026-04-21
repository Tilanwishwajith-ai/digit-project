import streamlit as st
st.title("Digit Recognizer AI")
import tensorflow as tf
import numpy as np
from PIL import Image
with st.sidebar:
    st.header("Instructions")
    st.markdown("1. Draw a digit clearly\n2. Click Predict\n3. See the magic!")

from streamlit_drawable_canvas import st_canvas
canvas_result = st_canvas(
    stroke_width=18, 
    stroke_color="#FFFFFF", 
    background_color="#000000", 
    height=280, width=280, 
    drawing_mode="freedraw"
)