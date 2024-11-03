# test.py

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from model import create_random_forest_model, create_cnn_model, create_lstm_model
from train import preprocess_data, create_sequences, load_raster_data, load_farm_data

def load_history(file_path):
    """Load training history from a JSON file."""
    with open(file_path, 'r') as f:
        history = json.load(f)
    return history

def plot_and_save_history(history, output_dir, model_name):
    """Plot and save training accuracy and loss."""
    # Accuracy plot
    plt.figure(figsize=(12, 6))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(output_dir, f'{model_name}_Training_Accuracy.png'))
    plt.close()

    # Loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(output_dir, f'{model_name}_Training_Loss.png'))
    plt.close()

def save_confusion_matrix(y_true, y_pred, class_names, output_dir, title="Confusion Matrix"):
    """Plot and save confusion matrix as an image."""
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(os.path.join(output_dir, f'{title.replace(" ", "_")}.png'))
    plt.close()

def save_actual_vs_predicted_plot(dates, y_true, y_pred, class_names, output_dir):
    """Plot and save actual vs predicted pasture states over time."""
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual': y_true,
        'Predicted': y_pred
    })
    results_df['Actual_Label'] = results_df['Actual'].map(lambda x: class_names[x])
    results_df['Predicted_Label'] = results_df['Predicted'].map(lambda x: class_names[x])

    plt.figure(figsize=(15, 8))
    plt.scatter(results_df['Date'], results_df['Actual_Label'], color='blue', marker='o', alpha=0.6, label='Actual Pasture State')
    plt.scatter(results_df['Date'], results_df['Predicted_Label'], color='orange', marker='x', alpha=0.6, label='Classified Pasture State')
    plt.xlabel('Time Steps')
    plt.ylabel('Pasture State')
    plt.title('Actual vs Classified Pasture State Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Actual_vs_Classified_Pasture_State_Over_Time.png'))
    plt.close()

def evaluate_model(rf_model, cnn_model, lstm_model, X_tabular, X_image, X_seq, y, farms, output_dir):
    """Evaluate models and save evaluation plots to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Model weights for combined predictions
    weight_rf, weight_cnn, weight_lstm = 0.3, 0.3, 0.4

    rf_proba = rf_model.predict_proba(X_tabular)
    cnn_proba = cnn_model.predict(X_image)
    lstm_proba = lstm_model.predict(X_seq)
    
    combined_proba = (weight_rf * rf_proba) + (weight_cnn * cnn_proba) + (weight_lstm * lstm_proba)
    final_predictions = np.argmax(combined_proba, axis=1)
    
    accuracy = accuracy_score(y, final_predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Extract class names for confusion matrix
    class_names = farms['PASTURE_STATE'].unique()
    class_names.sort()  # Ensure consistent ordering

    # Save confusion matrix
    save_confusion_matrix(y, final_predictions, class_names, output_dir, title=f"Confusion Matrix: {accuracy * 100:.2f}% Accuracy")

    # Plot and save actual vs classified states over time
    dates = farms['AREA_DATE'][len(X_seq):].reset_index(drop=True)
    save_actual_vs_predicted_plot(dates, y, final_predictions, class_names, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate combined RF, CNN, and LSTM model on test data and save results.")
    parser.add_argument("farm_data_dir", type=str, help="Path to the directory containing farm CSV test data files.")
    parser.add_argument("raster_data_dir", type=str, help="Path to the directory containing raster test data folders.")
    parser.add_argument("output_dir", type=str, help="Directory to save output images and plots.")

    args = parser.parse_args()
    
    # Load test data
    farms = load_farm_data(args.farm_data_dir)
    raster_data = load_raster_data(args.raster_data_dir)
    
    # Preprocess data
    X_tabular, y = preprocess_data(farms)
    X_seq, y_seq = create_sequences(X_tabular, y, seq_length=50)

    # Load trained models
    rf_model = create_random_forest_model()
    cnn_model = create_cnn_model((64, 64, 3), num_classes=len(np.unique(y)))
    lstm_model = create_lstm_model((X_seq.shape[1], X_seq.shape[2]), num_classes=len(np.unique(y)))
    
    # Evaluate and save results
    evaluate_model(rf_model, cnn_model, lstm_model, X_tabular, raster_data, X_seq, y, farms, args.output_dir)

    # Plot and save history for each model (assuming history files are in the output_dir)
    for fold in range(1, 11):
        cnn_history = load_history(os.path.join(args.output_dir, f"cnn_fold_{fold}_history.json"))
        lstm_history = load_history(os.path.join(args.output_dir, f"lstm_fold_{fold}_history.json"))

        plot_and_save_history(cnn_history, args.output_dir, f"cnn_fold_{fold}")
        plot_and_save_history(lstm_history, args.output_dir, f"lstm_fold_{fold}")
