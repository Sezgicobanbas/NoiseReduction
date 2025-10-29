# TensorFlow Configuration for Audio Denoising Project
import tensorflow as tf
import os

def configure_tensorflow_for_audio_denoising():
    """
    Optimal TensorFlow configuration for audio denoising with Attention U-Net
    Returns strategy object if multiple GPUs available, None otherwise
    """
    print("🔧 Configuring TensorFlow for optimal performance...")
    
    # 1. GPU Memory Configuration (MOST IMPORTANT)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU Memory Growth enabled for {len(gpus)} GPU(s)")
            
            # Optional: Set memory limit if you have limited VRAM
            # Uncomment and adjust if you get OOM errors
            # tf.config.experimental.set_memory_limit(gpus[0], 4096)  # 4GB limit
            
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
            print("Falling back to CPU")
            return None
    else:
        print("ℹ️  No GPUs detected, using CPU")
        return None
    
    # 2. Mixed Precision for 2x faster training (if you have modern GPU)
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision enabled (2x faster training)")
    except Exception as e:
        print(f"⚠️  Mixed precision not available: {e}")
    
    # 3. Multi-GPU Strategy (if you have multiple GPUs)
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"✅ Multi-GPU strategy enabled: {strategy.num_replicas_in_sync} devices")
        return strategy
    
    # 4. XLA Compilation - disabled for Apple Metal compatibility
    # tf.config.optimizer.set_jit(True)  # Disabled for Apple Metal
    print("⚠️  XLA compilation disabled (not compatible with Apple Metal)")
    
    # 5. Threading optimization
    tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all CPU cores
    tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all CPU cores
    print("✅ Threading optimized")
    
    print("🚀 TensorFlow configuration complete!")
    return None

def get_optimal_batch_size(model_type="attention_unet"):
    """
    Get optimal batch size based on your system and model type
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        # CPU-only batch sizes (conservative)
        batch_sizes = {
            "attention_unet": 2,
            "unet": 4,
            "cnn_transformer": 3
        }
        return batch_sizes.get(model_type, 2)
    
    # GPU available - Apple M4 optimized batch sizes
    try:
        if model_type == "attention_unet":
            return 4  # Conservative for Attention U-Net (memory intensive)
        elif model_type == "unet":
            return 8  # Regular U-Net can handle larger batches
        elif model_type == "cnn_transformer":
            return 6  # CNN-Transformer hybrid - moderate batch size
        else:
            return 4  # Safe default
    except:
        return 4  # Safe fallback

def configure_data_pipeline():
    """
    Configure data pipeline for optimal performance
    """
    # Enable experimental features for better data loading
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_deterministic = False  # Faster but less reproducible
    
    return options

# Usage example for your models
if __name__ == "__main__":
    # Configure TensorFlow
    strategy = configure_tensorflow_for_audio_denoising()
    
    # Get optimal batch size
    batch_size = get_optimal_batch_size("attention_unet")
    print(f"Recommended batch size: {batch_size}")
    
    # Example usage in your training
    if strategy:
        with strategy.scope():
            # Build your model here for multi-GPU
            print("Building model with multi-GPU strategy...")
    else:
        # Single GPU or CPU
        print("Building model for single device...")