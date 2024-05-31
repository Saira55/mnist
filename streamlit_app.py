import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np

# Title of the app
st.title("Train a Neural Network on MNIST and Test with an Uploaded Image")

# Load the MNIST data
st.write("Loading the MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to categorical one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Function to create the model
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create a model
model = create_model()

# Train the model
if st.button("Train Model"):
    st.write("Training the model...")
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=2)
    
    # Display the training accuracy
    st.write("Training Accuracy:", history.history['accuracy'][-1])
    st.write("Validation Accuracy:", history.history['val_accuracy'][-1])
    
    # Display the loss
    st.write("Training Loss:", history.history['loss'][-1])
    st.write("Validation Loss:", history.history['val_loss'][-1])

# Evaluate the model on test data
if st.button("Evaluate Model"):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    st.write(f"Test Accuracy: {accuracy}")
    st.write(f"Test Loss: {loss}")

# Upload an image to check against the trained model
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict the class of the image
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    st.write(f"Predicted Class: {predicted_class[0]}")
