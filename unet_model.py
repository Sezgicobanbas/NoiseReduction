# U-Net Model for Audio Denoising
# Core libraries
import os
import glob
import random
import warnings
import time
import re

# Data science and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Audio processing
import soundfile as sf
import librosa
import librosa.display

# Display utilities
from IPython.display import Audio, display

from concurrent.futures import ProcessPoolExecutor

# Signal processing
from scipy.signal import resample_poly
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Deep learning
import tensorflow as tf
from tensorflow import keras

# Import our TensorFlow configuration
try:
    from tensorflow_config import configure_tensorflow_for_audio_denoising, get_optimal_batch_size
except ImportError:
    # Fallback if tensorflow_config.py not available
    def configure_tensorflow_for_audio_denoising():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU Memory Growth enabled for {len(gpus)} GPU(s)")
                
                # Enable mixed precision for Apple Metal
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    print("✅ Mixed precision enabled")
                except Exception as e:
                    print(f"⚠️  Mixed precision not available: {e}")
                    
            except RuntimeError as e:
                print(f"❌ GPU configuration error: {e}")
        return None
    
    def get_optimal_batch_size(model_type="unet"):
        return 8  # Regular U-Net can handle larger batches than Attention U-Net
from keras import utils
from keras.utils import Sequence
from keras.models import Model
from keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D,
    Concatenate, Dropout, BatchNormalization, Add, ReLU, Multiply,
    ZeroPadding2D, Cropping2D, Dense, Reshape, LayerNormalization, MultiHeadAttention,
    Lambda, Embedding, Dropout, GlobalAveragePooling2D, AveragePooling2D,
    SeparableConv2D, Activation, Flatten,CenterCrop
)
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, Callback
from keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

import kagglehub


class TqdmProgressCallback(Callback):
    """Custom callback to show training progress with tqdm"""
    
    def __init__(self):
        super().__init__()
        self.epochs_pbar = None
        self.batch_pbar = None
        
    def on_train_begin(self, logs=None):
        self.epochs_pbar = tqdm(total=self.params['epochs'], 
                               desc="Training Progress", 
                               unit="epoch",
                               position=0,
                               leave=True)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.batch_pbar = tqdm(total=self.params['steps'], 
                              desc=f"Epoch {epoch+1}/{self.params['epochs']}", 
                              unit="batch",
                              position=1,
                              leave=False)
        
    def on_batch_end(self, batch, logs=None):
        if self.batch_pbar:
            self.batch_pbar.update(1)
            if logs:
                self.batch_pbar.set_postfix({
                    'loss': f"{logs.get('loss', 0):.4f}",
                    'mae': f"{logs.get('mae', 0):.4f}"
                })
    
    def on_epoch_end(self, epoch, logs=None):
        if self.batch_pbar:
            self.batch_pbar.close()
            
        if self.epochs_pbar:
            self.epochs_pbar.update(1)
            if logs:
                self.epochs_pbar.set_postfix({
                    'loss': f"{logs.get('loss', 0):.4f}",
                    'val_loss': f"{logs.get('val_loss', 0):.4f}",
                    'mae': f"{logs.get('mae', 0):.4f}",
                    'val_mae': f"{logs.get('val_mae', 0):.4f}"
                })
    
    def on_train_end(self, logs=None):
        if self.epochs_pbar:
            self.epochs_pbar.close()


def loss_function(y_true, y_pred):
    """Combined loss: MSE, MAE, and gradient loss for edge preservation."""
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
    grad_true_x = tf.nn.conv2d(y_true, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
    grad_true_y = tf.nn.conv2d(y_true, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
    grad_pred_x = tf.nn.conv2d(y_pred, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
    grad_pred_y = tf.nn.conv2d(y_pred, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
    grad_loss = tf.reduce_mean(tf.abs(grad_true_x - grad_pred_x)) + tf.reduce_mean(tf.abs(grad_true_y - grad_pred_y))
    return 0.7 * mse + 0.2 * mae + 0.1 * grad_loss


def evaluate_metrics(clean_specs, denoised_specs, clean_audio_paths=None, denoised_audio_paths=None):
    mse_list, mae_list, cosine_sim_list, ssim_list, psnr_list = [], [], [], [], []
    
    # Add progress bar for metrics evaluation
    with tqdm(total=len(clean_specs), desc="Evaluating Metrics", unit="sample") as pbar:
        for i, (clean_spec, den_spec) in enumerate(zip(clean_specs, denoised_specs)):
            # Squeeze singleton dimensions
            clean_spec = np.squeeze(clean_spec)
            den_spec = np.squeeze(den_spec)
            # Ensure shapes match
            if clean_spec.shape != den_spec.shape:
                min_shape = tuple(np.minimum(clean_spec.shape, den_spec.shape))
                clean_spec = clean_spec[:min_shape[0], :min_shape[1]]
                den_spec = den_spec[:min_shape[0], :min_shape[1]]
            clean_flat = clean_spec.flatten().astype(np.float32)
            den_flat = den_spec.flatten().astype(np.float32)
            mse_val = mean_squared_error(clean_flat, den_flat)
            mae_val = mean_absolute_error(clean_flat, den_flat)
            mse_list.append(mse_val)
            mae_list.append(mae_val)
            try:
                if np.all(np.isfinite(clean_flat)) and np.all(np.isfinite(den_flat)):
                    sim = 1 - cosine(clean_flat, den_flat)
                    cosine_sim_list.append(sim if np.isfinite(sim) else np.nan)
                else:
                    cosine_sim_list.append(np.nan)
            except Exception:
                cosine_sim_list.append(np.nan)
            ssim_list.append(ssim(clean_spec, den_spec, data_range=den_spec.max()-den_spec.min()))
            psnr_list.append(20 * np.log10(np.max(clean_spec) / np.sqrt(mse_val)) if mse_val > 0 else float('inf'))
            
            # Update progress bar with current metrics
            pbar.update(1)
            pbar.set_postfix({
                'MSE': f"{mse_val:.4f}",
                'MAE': f"{mae_val:.4f}",
                'SSIM': f"{ssim_list[-1]:.4f}"
            })
    
    metrics = {
        'Mean MSE': np.nanmean(mse_list),
        'Mean MAE': np.nanmean(mae_list),
        'Mean Cosine Similarity': np.nanmean(cosine_sim_list),
        'Mean SSIM': np.nanmean(ssim_list),
        'Mean PSNR': np.nanmean(psnr_list),
    }
    return metrics


def build_unet_fast_simple(input_shape):
    """ Fast U-Net with your original structure - no resize issues"""
    
    def fast_conv_block(x, filters, dropout_rate=0.0):
        """ Fast convolutional block with separable convolutions"""
        shortcut = x
        
        # Use separable convolutions for speed (reduces parameters by ~9x)
        x = SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        x = SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        
        # Residual connection - check if shapes match
        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    inputs = Input(shape=input_shape, dtype='float16')      
    # Encoder - Reduced filters for speed
    c1 = fast_conv_block(inputs, 24, dropout_rate=0.1)  # 32 -> 24
    p1 = MaxPooling2D(2)(c1)
    
    c2 = fast_conv_block(p1, 48, dropout_rate=0.1)      # 64 -> 48
    p2 = MaxPooling2D(2)(c2)
    
    c3 = fast_conv_block(p2, 96, dropout_rate=0.2)      # 128 -> 96
    p3 = MaxPooling2D(2)(c3)
    
    c4 = fast_conv_block(p3, 192, dropout_rate=0.2)     # 256 -> 192
    p4 = MaxPooling2D(2)(c4)
    
    # Bottleneck
    c5 = fast_conv_block(p4, 384, dropout_rate=0.3)     # 512 -> 384
    
    # Decoder with your original dimension handling (but optimized)
    u4 = Conv2DTranspose(192, 2, strides=2, padding='same')(c5)
    # Keep your original cropping/padding logic - it works!
    diff_height = c4.shape[1] - u4.shape[1]
    diff_width = c4.shape[2] - u4.shape[2]
    if diff_height > 0 or diff_width > 0:
        u4 = ZeroPadding2D(padding=((0, max(0, diff_height)), (0, max(0, diff_width))))(u4)
    elif diff_height < 0 or diff_width < 0:
        u4 = Cropping2D(cropping=((0, max(0, -diff_height)), (0, max(0, -diff_width))))(u4)
    u4 = Concatenate()([u4, c4])
    c6 = fast_conv_block(u4, 192, dropout_rate=0.2)
    
    # Second upsampling
    u3 = Conv2DTranspose(96, 2, strides=2, padding='same')(c6)
    diff_height = c3.shape[1] - u3.shape[1]
    diff_width = c3.shape[2] - u3.shape[2]
    if diff_height > 0 or diff_width > 0:
        u3 = ZeroPadding2D(padding=((0, max(0, diff_height)), (0, max(0, diff_width))))(u3)
    elif diff_height < 0 or diff_width < 0:
        u3 = Cropping2D(cropping=((0, max(0, -diff_height)), (0, max(0, -diff_width))))(u3)
    u3 = Concatenate()([u3, c3])
    c7 = fast_conv_block(u3, 96, dropout_rate=0.2)
    
    # Third upsampling
    u2 = Conv2DTranspose(48, 2, strides=2, padding='same')(c7)
    diff_height = c2.shape[1] - u2.shape[1]
    diff_width = c2.shape[2] - u2.shape[2]
    if diff_height > 0 or diff_width > 0:
        u2 = ZeroPadding2D(padding=((0, max(0, diff_height)), (0, max(0, diff_width))))(u2)
    elif diff_height < 0 or diff_width < 0:
        u2 = Cropping2D(cropping=((0, max(0, -diff_height)), (0, max(0, -diff_width))))(u2)
    u2 = Concatenate()([u2, c2])
    c8 = fast_conv_block(u2, 48, dropout_rate=0.1)
    
    # Fourth upsampling
    u1 = Conv2DTranspose(24, 2, strides=2, padding='same')(c8)
    diff_height = c1.shape[1] - u1.shape[1]
    diff_width = c1.shape[2] - u1.shape[2]
    if diff_height > 0 or diff_width > 0:
        u1 = ZeroPadding2D(padding=((0, max(0, diff_height)), (0, max(0, diff_width))))(u1)
    elif diff_height < 0 or diff_width < 0:
        u1 = Cropping2D(cropping=((0, max(0, -diff_height)), (0, max(0, -diff_width))))(u1)
    u1 = Concatenate()([u1, c1])
    c9 = fast_conv_block(u1, 24, dropout_rate=0.1)
    
    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid', dtype='float32', padding='same')(c9)
    
    return Model(inputs, outputs)


class SpectrogramDataGenerator(Sequence):
    def __init__(self, X, y, batch_size=8, shuffle=True, **kwargs):
        """
        Enhanced data generator with proper Keras compatibility.
        
        Args:
            X: Input data (noisy spectrograms)
            y: Target data (clean spectrograms)  
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data between epochs
            **kwargs: Additional arguments for Keras Sequence (workers, use_multiprocessing, max_queue_size)
        """
        # Call parent constructor with kwargs to fix the warning
        super().__init__(**kwargs)
        
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        """Get one batch of data"""
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Add channel dimension for CNN input
        X_batch = self.X[batch_indices][..., np.newaxis].astype(np.float32)
        y_batch = self.y[batch_indices][..., np.newaxis].astype(np.float32)
        
        return X_batch, y_batch

    def on_epoch_end(self):
        """Called at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_unet_model(X_train, y_train, X_val, y_val, config):
    """Train U-Net model and return results"""
    
    # 🚀 CONFIGURE TENSORFLOW FOR OPTIMAL PERFORMANCE
    print("🔧 Configuring TensorFlow for optimal performance...")
    strategy = configure_tensorflow_for_audio_denoising()
    
    # Get optimal batch size if not specified
    if 'BATCH_SIZE' not in config or config['BATCH_SIZE'] is None:
        config['BATCH_SIZE'] = get_optimal_batch_size("unet")
        print(f"📊 Auto-selected optimal batch size: {config['BATCH_SIZE']}")
    
    print("Training U-Net Model...")
    print(f"📊 Dataset Info:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Input shape: {X_train[0].shape}")
    print(f"  Batch size: {config['BATCH_SIZE']}")
    print(f"  Epochs: {config['EPOCHS']}")
    
    # Create data generators with optimized settings (disabled multiprocessing for Apple Metal)
    train_gen = SpectrogramDataGenerator(
        X_train, y_train, 
        batch_size=config['BATCH_SIZE'], 
        shuffle=True,
        workers=1,  # Disabled multiprocessing for Apple Metal compatibility
        use_multiprocessing=False,  # Disabled multiprocessing
        max_queue_size=10  # Reduced queue size
    )

    val_gen = SpectrogramDataGenerator(
        X_val, y_val, 
        batch_size=config['BATCH_SIZE'], 
        shuffle=False,
        workers=1,  # Disabled multiprocessing for Apple Metal compatibility
        use_multiprocessing=False,  # Disabled multiprocessing
        max_queue_size=10
    )
    
    # Build model with strategy scope if available
    print("🏗️  Building U-Net model...")
    input_shape = X_train[0].shape + (1,)
    
    if strategy:
        # Build model in strategy scope for multi-GPU
        with strategy.scope():
            model = build_unet_fast_simple(input_shape)
            
            # Compile with mixed precision compatible optimizer
            optimizer = Adam(learning_rate=config['LEARNING_RATE'])
            model.compile(optimizer=optimizer, 
                          loss=loss_function, 
                          metrics=["mae"])
    else:
        # Single GPU/CPU build
        model = build_unet_fast_simple(input_shape)
        
        # Compile with mixed precision compatible optimizer
        optimizer = Adam(learning_rate=config['LEARNING_RATE'])
        model.compile(optimizer=optimizer, 
                      loss=loss_function, 
                      metrics=["mae"])

    # Display model summary
    print("📋 Model Architecture:")
    model.summary()

    # Callbacks with improved settings
    models_dir = os.path.join(os.getcwd(), "models_dir")
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "best_unet.weights.h5")
    
    callbacks = [
        TqdmProgressCallback(),  # Add our custom progress callback
        ModelCheckpoint(
            checkpoint_path, 
            monitor="val_loss", 
            save_best_only=True, 
            save_weights_only=True,  # Faster saving
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", 
            patience=10,  # Increased patience for better training
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.7,  # Less aggressive reduction
            patience=5,  # Increased patience
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train model
    print("🚀 Starting training...")
    start_time = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config['EPOCHS'],
        callbacks=callbacks,
        verbose=0  # Set to 0 since we're using our custom progress callback
    )
    elapsed = time.time() - start_time
    print(f"\n✅ Training completed! Time: {elapsed:.2f} seconds")

    # Load best model
    print("📥 Loading best model weights...")
    model.load_weights(checkpoint_path)
    
    return {
        "model": model,
        "history": history,
        "checkpoint": checkpoint_path,
        "training_time": elapsed
    }


def predict_and_evaluate_unet(model, X_test, y_test, config):
    """Make predictions and evaluate U-Net model"""
    print("Evaluating U-Net Model...")
    print(f"📊 Test dataset: {len(X_test)} samples")
    
    # Predict with progress bar
    X_test_with_channel = X_test[..., np.newaxis]
    y_test_with_channel = y_test[..., np.newaxis]
    
    print("🔮 Making predictions...")
    # Calculate number of batches for progress tracking
    num_batches = int(np.ceil(len(X_test) / config['BATCH_SIZE']))
    
    with tqdm(total=num_batches, desc="Predicting", unit="batch") as pbar:
        def prediction_progress(batch_idx, logs=None):
            pbar.update(1)
        
        # Create a custom callback for prediction progress
        class PredictionCallback(Callback):
            def __init__(self, pbar):
                super().__init__()
                self.pbar = pbar
                self.batch_count = 0
            
            def on_predict_batch_end(self, batch, logs=None):
                self.batch_count += 1
                self.pbar.update(1)
                self.pbar.set_postfix({'Batch': f"{self.batch_count}/{num_batches}"})
        
        # Reset progress bar
        pbar.reset()
        callback = PredictionCallback(pbar)
        
        preds = model.predict(
            X_test_with_channel, 
            batch_size=config['BATCH_SIZE'], 
            verbose=0,  # Disable default verbose output
            callbacks=[callback]
        )
    
    print("📈 Calculating evaluation metrics...")
    # Evaluate metrics
    metrics = evaluate_metrics(y_test, preds)
    
    # Calculate MSE and MAE
    print("🧮 Computing overall metrics...")
    with tqdm(total=2, desc="Computing MSE & MAE") as pbar:
        mse = mean_squared_error(y_test_with_channel.flatten(), preds.flatten())
        pbar.update(1)
        pbar.set_postfix({'Metric': 'MSE'})
        
        mae = mean_absolute_error(y_test_with_channel.flatten(), preds.flatten())
        pbar.update(1)
        pbar.set_postfix({'Metric': 'MAE'})
    
    results = {
        "predictions": preds,
        "metrics": metrics,
        "mse": mse,
        "mae": mae
    }
    
    print("\n✅ U-Net Evaluation Results:")
    print("="*50)
    for k, v in metrics.items():
        print(f'  {k}: {v:.6f}')
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print("="*50)
    
    return results


if __name__ == "__main__":
    # Standalone training script for U-Net model
    import pickle
    from sklearn.model_selection import train_test_split
    
    class Config:
        """Configuration for training parameters"""
        EPOCHS = 100
        BATCH_SIZE = None  # Will be auto-determined based on hardware
        LEARNING_RATE = 5e-4  # Slightly lower for mixed precision stability

    def load_preprocessed_data():
        """Load the preprocessed spectrogram data"""
        spec_dir = os.path.join(os.getcwd(), "spec_arrays")
        
        if not os.path.exists(spec_dir):
            print("❌ Error: Preprocessed data not found!")
            print("Please run data_preprocessing.py first.")
            return None, None
        
        try:
            print("📂 Loading preprocessed spectrograms...")
            with tqdm(total=2, desc="Loading Data Files") as pbar:
                clean_stft_norm = np.load(os.path.join(spec_dir, "train_clean_stft_norm.npy"))
                pbar.update(1)
                pbar.set_postfix({'File': 'Clean spectrograms'})
                
                noisy_stft_norm = np.load(os.path.join(spec_dir, "train_noisy_stft_norm.npy"))
                pbar.update(1)
                pbar.set_postfix({'File': 'Noisy spectrograms'})
            
            print(f"✅ Loaded data shapes:")
            print(f"  Clean spectrograms: {clean_stft_norm.shape}")
            print(f"  Noisy spectrograms: {noisy_stft_norm.shape}")
            
            return clean_stft_norm, noisy_stft_norm
        
        except FileNotFoundError as e:
            print(f"❌ Error loading preprocessed data: {e}")
            print("Please run data_preprocessing.py first.")
            return None, None

    def prepare_data_splits(clean_stft_norm, noisy_stft_norm, test_size=0.15, val_size=0.15, random_state=42):
        """Split data into train, validation, and test sets"""
        print("✂️  Splitting data into train/validation/test sets...")
        
        with tqdm(total=3, desc="Data Splitting") as pbar:
            # First split off test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                noisy_stft_norm, clean_stft_norm, 
                test_size=test_size, 
                random_state=random_state
            )
            pbar.update(1)
            pbar.set_postfix({'Split': 'Test set'})
            
            # Then split temp into train/val
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, 
                test_size=val_ratio, 
                random_state=random_state
            )
            pbar.update(1)
            pbar.set_postfix({'Split': 'Train/Val sets'})
            
            # Validation step
            assert len(X_train) + len(X_val) + len(X_test) == len(noisy_stft_norm), "Data split size mismatch!"
            pbar.update(1)
            pbar.set_postfix({'Split': 'Validation complete'})
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_results(results, model_name):
        """Save results to file for later comparison"""
        results_dir = "model_results"
        os.makedirs(results_dir, exist_ok=True)
        
        print("💾 Saving results for later comparison...")
        
        with tqdm(total=3, desc="Saving Results") as pbar:
            # Prepare results without the model itself (too large)
            results_to_save = {
                'model_name': model_name,
                'history': results['history'].history,
                'checkpoint': results['checkpoint'],
                'training_time': results['training_time'],
                'test_results': results['test_results']
            }
            pbar.update(1)
            pbar.set_postfix({'Step': 'Preparing data'})
            
            # Save to file
            filename = f"{model_name.lower().replace('-', '_')}_results.pkl"
            filepath = os.path.join(results_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(results_to_save, f)
            pbar.update(1)
            pbar.set_postfix({'Step': 'Writing file'})
            
            # Verification
            if os.path.exists(filepath):
                pbar.update(1)
                pbar.set_postfix({'Step': 'Verification complete'})
            
        print(f"✅ Results saved to: {filepath}")
        
        return filepath

    # Main execution
    print("🚀 Starting U-Net Model Training...")
    print("="*80)
    
    # Initialize configuration
    config = {
        'EPOCHS': Config.EPOCHS,
        'BATCH_SIZE': Config.BATCH_SIZE,
        'LEARNING_RATE': Config.LEARNING_RATE
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load preprocessed data
    print("\n1. Loading preprocessed data...")
    clean_stft_norm, noisy_stft_norm = load_preprocessed_data()
    
    if clean_stft_norm is None or noisy_stft_norm is None:
        print("Failed to load data. Exiting...")
        exit(1)
    
    # Prepare data splits
    print("\n2. Preparing data splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(
        clean_stft_norm, noisy_stft_norm
    )
    
    print(f"Data split sizes:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples") 
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Train U-Net model
    print(f"\n3. Training U-Net Model...")
    print("="*60)
    
    try:
        # Train model
        train_results = train_unet_model(X_train, y_train, X_val, y_val, config)
        
        # Evaluate on test set
        test_results = predict_and_evaluate_unet(train_results['model'], X_test, y_test, config)
        
        # Combine results
        all_results = {
            'model': train_results['model'],
            'history': train_results['history'],
            'checkpoint': train_results['checkpoint'],
            'training_time': train_results['training_time'],
            'test_results': test_results
        }
        
        # Save results for later comparison
        save_results(all_results, "U-Net")
        
        print(f"✅ U-Net training completed successfully!")
        print(f"📁 Model saved to: {train_results['checkpoint']}")
        print(f"📊 Results saved for comparison")
        
        # Print summary
        print(f"\n📈 U-Net Results Summary:")
        print(f"  Training Time: {train_results['training_time']:.2f} seconds")
        print(f"  Test MSE: {test_results['mse']:.6f}")
        print(f"  Test MAE: {test_results['mae']:.6f}")
        for metric_name, value in test_results['metrics'].items():
            print(f"  {metric_name}: {value:.6f}")
        
        print("\n🎉 U-Net training completed!")
        print("You can now run other model training scripts or use main_comparison.py to compare results.")
        
    except Exception as e:
        print(f"❌ Error training U-Net: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)