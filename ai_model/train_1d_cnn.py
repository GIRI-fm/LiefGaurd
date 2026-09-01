#!/usr/bin/env python3
"""
============================================================================
LiefGaurd AI Model Training Pipeline
1D-CNN for CNS Fatigue Classification from Kinematic Time-Series Data
Exports INT8-quantized TensorFlow Lite model for ESP32 deployment
============================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

DATASET_SIZE = 2000           # Number of synthetic training samples
TIMESTEPS = 128               # Sequence length (128 samples @ 50Hz = 2.56s)
N_CHANNELS = 6                # Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Model hyperparameters
CONV1_FILTERS = 16
CONV2_FILTERS = 32
KERNEL_SIZE = 5
POOL_SIZE = 2
DENSE_UNITS = 16
DROPOUT_RATE = 0.3

# Training settings
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Output paths
OUTPUT_DIR = "./models"
MODEL_H5_PATH = os.path.join(OUTPUT_DIR, "liefguard_model.h5")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "liefguard_model.tflite")
METADATA_PATH = os.path.join(OUTPUT_DIR, "model_metadata.json")

# ============================================================================
# Synthetic Dataset Generation
# ============================================================================

def generate_synthetic_kinematic_data(n_samples=DATASET_SIZE, 
                                     timesteps=TIMESTEPS, 
                                     n_channels=N_CHANNELS,
                                     seed=42):
    """
    Generate synthetic kinematic time-series data simulating IMU readings.
    
    Normal form (label=0): Low-variance, stable acceleration/gyroscope signals
    Fatigued form (label=1): High-variance with micro-tremors and velocity decay
    
    Args:
        n_samples: Number of training sequences
        timesteps: Length of each sequence (128 = 2.56s @ 50Hz)
        n_channels: Number of IMU channels (6)
        seed: Random seed for reproducibility
        
    Returns:
        X: (n_samples, timesteps, n_channels) kinematic data
        y: (n_samples, 1) binary labels [0=Normal, 1=Fatigued]
    """
    np.random.seed(seed)
    
    X = np.zeros((n_samples, timesteps, n_channels), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    
    print(f"[DATA] Generating {n_samples} synthetic kinematic sequences...")
    
    for i in range(n_samples):
        # Randomly assign class (0 or 1)
        label = np.random.randint(0, 2)
        y[i, 0] = label
        
        if label == 0:
            # Normal form: Low noise, stable signals
            noise_scale = 0.3
            tremor_amplitude = 0.05
            velocity_decay = 1.0
        else:
            # Fatigued form: High noise, micro-tremors, velocity decay
            noise_scale = 0.8
            tremor_amplitude = 0.3
            velocity_decay = 0.7
        
        # Generate base acceleration signal (simulating barbell motion)
        base_accel = np.sin(np.linspace(0, 2*np.pi, timesteps))
        
        # Generate channels
        for t in range(timesteps):
            # Accelerometer channels (X, Y, Z)
            X[i, t, 0] = base_accel[t] * velocity_decay + np.random.normal(0, noise_scale)  # accel_x
            X[i, t, 1] = 0.5 * np.cos(np.linspace(0, 2*np.pi, timesteps)[t]) + np.random.normal(0, noise_scale)  # accel_y
            X[i, t, 2] = 9.81 + np.random.normal(0, noise_scale)  # accel_z (gravity)
            
            # Gyroscope channels (X, Y, Z) - with tremor for fatigued
            tremor = tremor_amplitude * np.sin(10 * np.linspace(0, 2*np.pi, timesteps)[t])
            X[i, t, 3] = tremor + np.random.normal(0, noise_scale * 0.5)  # gyro_x
            X[i, t, 4] = tremor + np.random.normal(0, noise_scale * 0.5)  # gyro_y
            X[i, t, 5] = base_accel[t] * velocity_decay * 10 + np.random.normal(0, noise_scale)  # gyro_z
    
    print(f"[DATA] Generated dataset shape: X={X.shape}, y={y.shape}")
    print(f"[DATA] Class distribution: {np.sum(y==0):.0f} Normal, {np.sum(y==1):.0f} Fatigued")
    
    return X, y

# ============================================================================
# Data Preprocessing
# ============================================================================

def normalize_data(X_train, X_val, X_test):
    """
    Standardize kinematic data using training set statistics.
    
    Args:
        X_train, X_val, X_test: Training, validation, test splits
        
    Returns:
        Normalized arrays with mean=0, std=1
    """
    print("[PREPROCESS] Normalizing data...")
    
    # Compute statistics from training set only (prevent data leakage)
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0  # Avoid division by zero
    
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_val_norm, X_test_norm

# ============================================================================
# Model Architecture - Lightweight 1D-CNN
# ============================================================================

def build_1d_cnn_model(input_shape=(TIMESTEPS, N_CHANNELS)):
    """
    Build lightweight 1D-CNN architecture for TinyML inference on ESP32.
    
    Architecture:
    - Conv1D(16, kernel=5) → ReLU
    - MaxPooling1D(2)
    - Conv1D(32, kernel=5) → ReLU
    - GlobalAveragePooling1D
    - Dense(16) → ReLU + Dropout(0.3)
    - Dense(1) → Sigmoid (binary classification)
    
    Total parameters: ~5,000 (fits easily in ESP32 RAM)
    
    Args:
        input_shape: Tuple (timesteps, channels)
        
    Returns:
        Compiled Keras model
    """
    print(f"[MODEL] Building 1D-CNN with input shape {input_shape}...")
    
    model = models.Sequential([
        # Block 1: Conv + MaxPool
        layers.Conv1D(
            filters=CONV1_FILTERS,
            kernel_size=KERNEL_SIZE,
            strides=1,
            padding='same',
            activation='relu',
            input_shape=input_shape,
            name='conv1d_1'
        ),
        layers.MaxPooling1D(pool_size=POOL_SIZE, name='maxpool1d_1'),
        
        # Block 2: Conv + GlobalAvgPool
        layers.Conv1D(
            filters=CONV2_FILTERS,
            kernel_size=KERNEL_SIZE,
            strides=1,
            padding='same',
            activation='relu',
            name='conv1d_2'
        ),
        layers.GlobalAveragePooling1D(name='global_avg_pool'),
        
        # Dense layers with regularization
        layers.Dense(DENSE_UNITS, activation='relu', name='dense_1'),
        layers.Dropout(DROPOUT_RATE, name='dropout_1'),
        
        # Output layer (binary classification: Normal vs. Fatigued)
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile with binary cross-entropy loss
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    model.summary()
    return model

# ============================================================================
# Training Pipeline
# ============================================================================

def train_model(X, y):
    """
    Train the 1D-CNN model with early stopping and validation monitoring.
    
    Args:
        X: Input kinematic data (n_samples, timesteps, channels)
        y: Binary labels (n_samples, 1)
        
    Returns:
        Trained Keras model and training history
    """
    print("[TRAIN] Splitting dataset...")
    
    # Split into train/val/test
    n_train = int(len(X) * (1 - TEST_SPLIT))
    X_train_val = X[:n_train]
    y_train_val = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    
    # Further split train/val
    n_train_split = int(len(X_train_val) * (1 - VALIDATION_SPLIT))
    X_train = X_train_val[:n_train_split]
    y_train = y_train_val[:n_train_split]
    X_val = X_train_val[n_train_split:]
    y_val = y_train_val[n_train_split:]
    
    print(f"[TRAIN] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Normalize data
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
    
    # Build model
    model = build_1d_cnn_model()
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train
    print(f"[TRAIN] Training for up to {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate on test set
    print("[EVAL] Evaluating on test set...")
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[EVAL] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    
    return model, history, X_test, y_test

# ============================================================================
# Model Quantization & Export
# ============================================================================

def quantize_to_tflite(keras_model, quantization_representative_data=None):
    """
    Convert Keras model to INT8-quantized TensorFlow Lite format for ESP32.
    
    Args:
        keras_model: Trained Keras model
        quantization_representative_data: Optional calibration data for post-training quantization
        
    Returns:
        TFLite model bytes
    """
    print("[QUANTIZE] Converting Keras model to TensorFlow Lite...")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    # Enable post-training dynamic range quantization (INT8)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # If representative data provided, use full integer quantization
    if quantization_representative_data is not None:
        def representative_dataset():
            for data in quantization_representative_data:
                yield [data[np.newaxis, :, :].astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    print(f"[QUANTIZE] TFLite model size: {len(tflite_model) / 1024:.1f} KB")
    return tflite_model

def export_model(keras_model, tflite_model):
    """
    Save Keras model and quantized TFLite model to disk.
    
    Args:
        keras_model: Trained Keras model
        tflite_model: Quantized TFLite model bytes
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save Keras model (float32)
    print(f"[EXPORT] Saving Keras model to {MODEL_H5_PATH}...")
    keras_model.save(MODEL_H5_PATH)
    
    # Save TFLite model (INT8 quantized)
    print(f"[EXPORT] Saving TFLite model to {TFLITE_PATH}...")
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    # Save metadata
    metadata = {
        "model_name": "liefguard_1d_cnn",
        "timestamp": datetime.now().isoformat(),
        "input_shape": [TIMESTEPS, N_CHANNELS],
        "input_type": "float32",
        "output_type": "float32",
        "quantization": "INT8",
        "sample_rate_hz": 50,
        "inference_window_ms": TIMESTEPS * 20,  # 20ms per sample
        "fatigue_threshold": 0.75,
        "tensorflow_version": tf.__version__,
        "model_size_kb": len(tflite_model) / 1024
    }
    
    print(f"[EXPORT] Saving metadata to {METADATA_PATH}...")
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("[EXPORT] Export complete!")
    return metadata

# ============================================================================
# Visualization
# ============================================================================

def plot_training_history(history, output_dir=OUTPUT_DIR):
    """
    Plot training and validation curves.
    
    Args:
        history: Keras training history object
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Binary Crossentropy Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Classification Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=100)
    print(f"[PLOT] Saved training curves to {plot_path}")
    plt.close()

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print("="*80)
    print("LiefGaurd 1D-CNN Training Pipeline")
    print("="*80)
    
    # 1. Generate synthetic dataset
    X, y = generate_synthetic_kinematic_data(
        n_samples=DATASET_SIZE,
        timesteps=TIMESTEPS,
        n_channels=N_CHANNELS
    )
    
    # 2. Train model
    model, history, X_test, y_test = train_model(X, y)
    
    # 3. Quantize to TFLite
    tflite_model = quantize_to_tflite(model, X_test[:10])
    
    # 4. Export models
    metadata = export_model(model, tflite_model)
    
    # 5. Plot training curves
    plot_training_history(history)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"✓ Keras model: {MODEL_H5_PATH}")
    print(f"✓ TFLite model: {TFLITE_PATH}")
    print(f"✓ Metadata: {METADATA_PATH}")
    print(f"\nModel Specs:")
    print(f"  Input: ({TIMESTEPS}, {N_CHANNELS}) samples @ 50Hz = {TIMESTEPS*20}ms window")
    print(f"  Output: Binary classification (0=Normal, 1=Fatigued)")
    print(f"  Size: {metadata['model_size_kb']:.1f} KB (fits in ESP32)")
    print(f"  Quantization: {metadata['quantization']}")
    print("="*80)

if __name__ == "__main__":
    main()
