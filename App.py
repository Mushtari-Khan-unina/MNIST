import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np

# Load the model
model_new = tf.keras.models.load_model('mnist.h5')

# Streamlit app title and description
st.title("MNIST Digit Recognizer")
st.write("Draw a digit in the canvas and click 'Predict' to recognize it.")

# Canvas settings
SIZE = 192
canvas_result = st_canvas(
    fill_color="#ffffff",
    stroke_width=10,
    stroke_color='#000000',
    background_color="#000000",
    height=150,
    width=150,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Input Image')
    st.image(img_rescaling, use_column_width=True)

if st.button('Predict'):
    if canvas_result.image_data is not None:
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pred = model_new.predict(test_x.reshape(1, 28, 28, 1))
        st.write(f'Result: {np.argmax(pred[0])}')
        st.bar_chart(pred[0])
    else:
        st.warning("Please draw a digit before predicting.")
