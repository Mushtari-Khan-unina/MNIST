#!/usr/bin/env python3

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your pre-trained MNIST digit classification model
model = tf.keras.models.load_model('mnist.h5')  # Replace with the actual path to your model

# Set Streamlit app title and header
st.title("MNIST Digit Classifier")
st.header("Draw a digit and let the model classify it!")

# Create a canvas for drawing
canvas = st.canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=200,
    width=200,
)

# Function to preprocess and predict the digit
def predict_digit(image):
    # Resize the image to match the input size expected by the model (28x28 pixels)
    image = image.resize((28, 28), Image.ANTIALIAS)

    # Convert the image to grayscale and normalize pixel values
    image = np.array(image.convert('L')) / 255.0

    # Reshape the image to match the input shape expected by the model (1, 28, 28, 1)
    image = np.reshape(image, (1, 28, 28, 1))

    # Make a prediction using the model
    prediction = model.predict(image)

    # Get the predicted digit
    digit = np.argmax(prediction)

    return digit, prediction

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Get the drawn image from the canvas
    drawn_image = canvas.image

    # Check if the canvas is not empty
    if drawn_image is not None:
        # Predict the digit and get the confidence scores
        predicted_digit, prediction_scores = predict_digit(drawn_image)

        # Display the predicted digit and confidence scores
        st.write(f"Predicted Digit: {predicted_digit}")
        st.write("Prediction Confidence:")
        st.bar_chart(prediction_scores.flatten())

# Add a reset button to clear the canvas
if st.button("Clear Canvas"):
    canvas.clear()

# Add some additional information
st.sidebar.header("About")
st.sidebar.write("This app allows you to draw a digit on the canvas and uses a pre-trained model to classify it. It's based on the MNIST dataset, which contains handwritten digits (0-9).")

# Add a GitHub link to your model repository
st.sidebar.write("GitHub Repository:")
st.sidebar.markdown("[Your Model Repository](https://github.com/your_username/your_model_repo)")

# Add a footer with your name or credits
st.footer("Developed by Your Name")

# Run the Streamlit app
if __name__ == "__main__":
    st.run()
