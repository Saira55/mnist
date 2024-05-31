import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Save the data locally
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)

print("MNIST data has been saved locally.")

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Title of the app
st.title("Train a Neural Network on MNIST")

# Load the MNIST data from local files
st.write("Loading the MNIST dataset...")
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

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
