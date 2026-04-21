import streamlit as st
st.title("Digit Recognizer AI")
import tensorflow as tf
import numpy as np
from PIL import Image
with st.sidebar:
    st.header("Instructions")
    st.markdown("1. Draw a digit clearly\n2. Click Predict\n3. See the magic!")