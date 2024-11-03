# train.py

import os
import argparse
import pandas as pd
import numpy as np
import json
import rasterio
from rasterio.plot import reshape_as_image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from model import create_random_forest_model, create_cnn_model, create_lstm_model
import tensorflow as tf

def load_farm_data(base_path):
    """Load farm CSV data from the specified directory."""
    farms = []
    for filename in os.listdir(base_path):
        if filename.endswith(".csv"):
            farm_df = pd.read_csv(os.path.join(base_path, filename))
            farms.append(farm_df)
    return pd.concat(farms, ignore_index=True)

def load_raster_data(raster_dir):
    """Load and preprocess raster data."""
    raster_data = []
    for root, _, files in os.walk(raster_dir):
        for file in files:
            if file.endswith(".tif"):
                with rasterio.open(os.path.join(root, file)) as src:
                    raster = src.read()
                    raster = reshape_as_image(raster)
                    raster_data.append(raster)
    return np.array(raster_data)

def preprocess_data(farms):
    label_encoder = LabelEncoder()
    farms['PASTURE_STATE_ENCODED'] = label_encoder.fit_transform(farms['PASTURE_STATE'])

    farms['AREA_DATE'] = pd.to_datetime(farms['AREA_DATE'])
    farms['YEAR'] = farms['AREA_DATE'].dt.year
    farms['MONTH'] = farms['AREA_DATE'].dt.month
    farms['DAY'] = farms['AREA_DATE'].dt.day
    farms['DAY_OF_WEEK'] = farms['AREA_DATE'].dt.dayofweek

    feature_columns = ['LANDSIZE_HA', 'LATITUDE', 'LONGITUDE', 'NDVI', 'SAVI', 'EVI', 'ARABLELANDSIZE_HA',
                       'area', 'centroid_lat', 'centroid_lon', 'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK']

    farms['area'] = farms['GEOMETRY'].area
    farms['centroid_lat'] = farms['GEOMETRY'].centroid.y
    farms['centroid_lon'] = farms['GEOMETRY'].centroid.x

    X = farms[feature_columns]
    y = farms['PASTURE_STATE_ENCODED']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def create_sequences(X, y, seq_length=50):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

def save_history(history, output_dir, model_name):
    """Save training history as JSON."""
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    print(f"Saved {model_name} history at {history_path}")

def train_combined_model(X_tabular, y, X_image, num_classes, output_dir):
    stratified_kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    weight_rf, weight_cnn, weight_lstm = 0.3, 0.3, 0.4
    results = []

    for fold, (train_index, val_index) in enumerate(stratified_kf.split(X_tabular, y)):
        print(f"Training fold {fold + 1}...")

        X_tabular_train, X_tabular_val = X_tabular[train_index], X_tabular[val_index]
        X_image_train, X_image_val = X_image[train_index], X_image[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train Random Forest
        rf_model = create_random_forest_model()
        rf_model.fit(X_tabular_train, y_train)

        # Train CNN
        cnn_model = create_cnn_model((64, 64, 3), num_classes)
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
        
        cnn_history = cnn_model.fit(X_image_train, y_train_cat, epochs=50, batch_size=16, 
                                    validation_data=(X_image_val, y_val_cat), callbacks=[early_stopping], verbose=1)
        save_history(cnn_history, output_dir, f"cnn_fold_{fold + 1}")

        # Train LSTM
        X_lstm_train, y_lstm_train = create_sequences(X_tabular_train, y_train)
        X_lstm_val, y_lstm_val = create_sequences(X_tabular_val, y_val)
        lstm_model = create_lstm_model((X_lstm_train.shape[1], X_lstm_train.shape[2]), num_classes)
        
        lstm_history = lstm_model.fit(X_lstm_train, y_lstm_train, epochs=50, batch_size=32, 
                                      validation_data=(X_lstm_val, y_lstm_val), callbacks=[early_stopping], verbose=1)
        save_history(lstm_history, output_dir, f"lstm_fold_{fold + 1}")

        # Track results
        results.append({'fold': fold + 1, 'rf_model': rf_model, 'cnn_model': cnn_model, 'lstm_model': lstm_model})

    print("Training completed.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train combined RF, CNN, and LSTM model on farm data.")
    parser.add_argument("farm_data_dir", type=str, help="Path to the directory containing farm CSV data files.")
    parser.add_argument("raster_data_dir", type=str, help="Path to the directory containing raster data folders.")
    parser.add_argument("output_dir", type=str, help="Directory to save model history and results.")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    farms = load_farm_data(args.farm_data_dir)
    raster_data = load_raster_data(args.raster_data_dir)
    
    # Preprocess data
    X_tabular, y = preprocess_data(farms)
    num_classes = len(np.unique(y))
    X_seq, y_seq = create_sequences(X_tabular, y, seq_length=50)

    # Train combined model
    train_combined_model(X_seq, y_seq, raster_data, num_classes, args.output_dir)
