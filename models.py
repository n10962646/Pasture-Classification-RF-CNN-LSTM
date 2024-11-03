# model.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import LSTM, Dense

def create_random_forest_model():
    """Create a Random Forest model."""
    return RandomForestClassifier(random_state=42)

def create_cnn_model(input_shape, num_classes):
    """Create a CNN model based on MobileNetV2."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model layers
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    cnn_output = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=cnn_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_model(input_shape, num_classes):
    """Create an LSTM model."""
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
