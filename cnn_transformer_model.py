# CNN-Transformer Hybrid Model for Audio Denoising
import os
import time
import numpy as np
from tqdm import tqdm

# Machine learning
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim

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
    
    def get_optimal_batch_size(model_type="cnn_transformer"):
        return 6  # CNN-Transformer can handle slightly larger batches

from keras import utils
from keras.utils import Sequence
from keras.models import Model
from keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D,
    Concatenate, Dropout, BatchNormalization, Add, ReLU, Multiply,
    ZeroPadding2D, Cropping2D, Dense, Reshape, LayerNormalization, MultiHeadAttention,
    Lambda, Embedding, Dropout
)
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, Callback

# Remove the old mixed precision setting - it will be handled by tensorflow_config
# from keras import mixed_precision
# mixed_precision.set_global_policy('float32')
# tf.keras.backend.set_floatx('float32')


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


def transformer_encoder_patches(inputs, patch_size=8, num_heads=4, ff_dim=128):
    """Memory-efficient transformer that works on patches"""
    
    _, h, w, c = inputs.shape
    
    # Extract patches using Lambda layer
    def extract_patches(x):
        return tf.image.extract_patches(
            images=x,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'  # Changed from 'SAME' to 'VALID' to control size
        )
    
    patches = Lambda(extract_patches)(inputs)
    
    # Calculate patch dimensions with VALID padding
    patch_h = h // patch_size
    patch_w = w // patch_size
    patch_dim = patch_size * patch_size * c
    
    # Reshape to (batch, num_patches, patch_dim)
    x = Reshape((patch_h * patch_w, patch_dim))(patches)
    
    # Add positional encoding
    num_patches = patch_h * patch_w
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = Embedding(input_dim=num_patches, output_dim=patch_dim)(positions)
    x = x + position_embedding
    
    # Memory-efficient transformer with smaller dimensions
    # First, project to lower dimension to reduce computation
    projected_dim = min(256, patch_dim)  # Cap the dimension
    x_proj = Dense(projected_dim)(x)
    
    # Layer normalization
    x_norm1 = LayerNormalization(epsilon=1e-6)(x_proj)
    
    # Multi-head attention with smaller key_dim
    key_dim = max(16, projected_dim // num_heads)  # Ensure reasonable key_dim
    attn_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=key_dim,
        dropout=0.1  # Add dropout for regularization
    )(x_norm1, x_norm1)
    
    # Residual connection
    x1 = Add()([x_proj, attn_output])
    
    # Feed-forward with bottleneck
    x_norm2 = LayerNormalization(epsilon=1e-6)(x1)
    ff_output = Dense(ff_dim, activation='relu')(x_norm2)
    ff_output = Dense(projected_dim)(ff_output)
    ff_output = Dropout(0.1)(ff_output)  # Add dropout
    x2 = Add()([x1, ff_output])
    
    # Project back to original patch dimension
    x_out = Dense(patch_dim)(x2)
    
    # Reconstruct spatial dimensions
    x_out = Reshape((patch_h, patch_w, patch_dim))(x_out)
    
    # Use Conv2DTranspose for better reconstruction than simple resize
    x_out = Conv2DTranspose(c, (patch_size, patch_size), 
                           strides=(patch_size, patch_size), 
                           padding='VALID')(x_out)
    
    return x_out


def build_cnn_transformer_denoiser(input_shape):
    inputs = Input(shape=input_shape)
    
    # CNN Encoder with more aggressive downsampling
    x = Conv2D(32, (3, 3), padding='same', strides=2)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(64, (3, 3), padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(128, (3, 3), padding='same', strides=2)(x)  # Additional downsampling
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Memory-efficient transformer with appropriate patch size
    # Calculate current feature map size
    current_h = input_shape[0] // 8  # After 3 strided convs (2^3 = 8)
    current_w = input_shape[1] // 8
    
    # Use smaller patch size for the reduced feature maps
    patch_size = min(8, current_h, current_w)  # Ensure patch_size <= feature map size
    
    x = transformer_encoder_patches(x, patch_size=patch_size, num_heads=4, ff_dim=128)
    
    # CNN Decoder
    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
    
    # Ensure output matches input dimensions
    if outputs.shape[1:3] != input_shape[0:2]:
        def resize_output(x):
            return tf.image.resize(x, [input_shape[0], input_shape[1]])
        outputs = Lambda(resize_output)(outputs)
    
    return Model(inputs, outputs)


class SpectrogramDataGenerator(Sequence):
    def __init__(self, X, y, batch_size=8, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_indices][..., np.newaxis].astype(np.float32)
        y_batch = self.y[batch_indices][..., np.newaxis].astype(np.float32)
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_cnn_transformer_model(X_train, y_train, X_val, y_val, config):
    """Train CNN-Transformer model and return results"""
    
    # 🚀 CONFIGURE TENSORFLOW FOR OPTIMAL PERFORMANCE
    print("🔧 Configuring TensorFlow for optimal performance...")
    strategy = configure_tensorflow_for_audio_denoising()
    
    # Get optimal batch size if not specified
    if 'BATCH_SIZE' not in config or config['BATCH_SIZE'] is None:
        config['BATCH_SIZE'] = get_optimal_batch_size("cnn_transformer")
        print(f"📊 Auto-selected optimal batch size: {config['BATCH_SIZE']}")
    
    print("Training CNN-Transformer Model...")
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
    print("🏗️  Building CNN-Transformer model...")
    input_shape = X_train[0].shape + (1,)
    
    if strategy:
        # Build model in strategy scope for multi-GPU
        with strategy.scope():
            model = build_cnn_transformer_denoiser(input_shape)
            
            # Compile with mixed precision compatible optimizer
            optimizer = Adam(learning_rate=config['LEARNING_RATE'])
            model.compile(optimizer=optimizer, 
                          loss=loss_function, 
                          metrics=["mae"])
    else:
        # Single GPU/CPU build
        model = build_cnn_transformer_denoiser(input_shape)
        
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
    checkpoint_path = os.path.join(models_dir, "best_cnn_transformer.weights.h5")
    
    callbacks = [
        TqdmProgressCallback(),  # Add our custom progress callback
        ModelCheckpoint(
            checkpoint_path, 
            monitor="val_loss", 
            save_best_only=True, 
            save_weights_only=True,  # Faster saving
            verbose=1
        ),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
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


def predict_and_evaluate_cnn_transformer(model, X_test, y_test, config):
    """Make predictions and evaluate CNN-Transformer model"""
    print("Evaluating CNN-Transformer Model...")
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
    
    print("\n✅ CNN-Transformer Evaluation Results:")
    print("="*50)
    for k, v in metrics.items():
        print(f'  {k}: {v:.6f}')
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print("="*50)
    
    return results


if __name__ == "__main__":
    # Standalone training script for CNN-Transformer model
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
            filename = f"{model_name.lower().replace('-', '_').replace(' ', '_')}_results.pkl"
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
    print("🚀 Starting CNN-Transformer Model Training...")
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
    
    # Train CNN-Transformer model
    print(f"\n3. Training CNN-Transformer Model...")
    print("="*60)
    
    try:
        # Train model
        train_results = train_cnn_transformer_model(X_train, y_train, X_val, y_val, config)
        
        # Evaluate on test set
        test_results = predict_and_evaluate_cnn_transformer(train_results['model'], X_test, y_test, config)
        
        # Combine results
        all_results = {
            'model': train_results['model'],
            'history': train_results['history'],
            'checkpoint': train_results['checkpoint'],
            'training_time': train_results['training_time'],
            'test_results': test_results
        }
        
        # Save results for later comparison
        save_results(all_results, "CNN-Transformer")
        
        print(f"✅ CNN-Transformer training completed successfully!")
        print(f"📁 Model saved to: {train_results['checkpoint']}")
        print(f"📊 Results saved for comparison")
        
        # Print summary
        print(f"\n📈 CNN-Transformer Results Summary:")
        print(f"  Training Time: {train_results['training_time']:.2f} seconds")
        print(f"  Test MSE: {test_results['mse']:.6f}")
        print(f"  Test MAE: {test_results['mae']:.6f}")
        for metric_name, value in test_results['metrics'].items():
            print(f"  {metric_name}: {value:.6f}")
        
        print("\n🎉 CNN-Transformer training completed!")
        print("You can now run other model training scripts or use main_comparison.py to compare results.")
        
    except Exception as e:
        print(f"❌ Error training CNN-Transformer: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)