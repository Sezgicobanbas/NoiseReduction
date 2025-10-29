# Audio denoising model comparison and visualization script
# This script loads saved results from individual model training runs and provides
# comprehensive comparison, visualization, and audio generation capabilities

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import soundfile as sf
import librosa
from datetime import datetime

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_saved_results():
    """Load saved model results from pickle files"""
    results_dir = "model_results"
    
    if not os.path.exists(results_dir):
        print("Error: No saved model results found!")
        print("Please run the individual model training scripts first:")
        print("  python3 unet_model.py")
        print("  python3 attention_unet_model.py") 
        print("  python3 cnn_transformer_model.py")
        return {}
    
    saved_results = {}
    model_files = {
        'U-Net': 'u_net_results.pkl',
        'Attention U-Net': 'attention_u_net_results.pkl',
        'CNN-Transformer': 'cnn_transformer_results.pkl'
    }
    
    for model_name, filename in model_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    result = pickle.load(f)
                saved_results[model_name] = result
                print(f"✓ Loaded {model_name} results")
            except Exception as e:
                print(f"⚠️ Error loading {model_name} results: {e}")
        else:
            print(f"⚠️ {model_name} results not found at {filepath}")
    
    return saved_results


def compare_metrics(results):
    """Compare evaluation metrics across all models"""
    if not results:
        print("No results to compare!")
        return
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create comparison table
    metrics_to_compare = ['mse', 'mae', 'ssim', 'psnr', 'cosine_similarity']
    
    print(f"{'Model':<20} {'MSE':<12} {'MAE':<12} {'SSIM':<12} {'PSNR':<12} {'Cosine Sim':<12}")
    print("-" * 92)
    
    for model_name, result in results.items():
        if 'evaluation_metrics' in result:
            metrics = result['evaluation_metrics']
            print(f"{model_name:<20} "
                  f"{metrics.get('mse', 'N/A'):<12.6f} "
                  f"{metrics.get('mae', 'N/A'):<12.6f} "
                  f"{metrics.get('ssim', 'N/A'):<12.6f} "
                  f"{metrics.get('psnr', 'N/A'):<12.6f} "
                  f"{metrics.get('cosine_similarity', 'N/A'):<12.6f}")
    
    # Find best performing models for each metric
    print("\n" + "="*50)
    print("BEST PERFORMING MODELS BY METRIC")
    print("="*50)
    
    for metric in metrics_to_compare:
        best_model = None
        best_value = None
        
        for model_name, result in results.items():
            if 'evaluation_metrics' in result and metric in result['evaluation_metrics']:
                value = result['evaluation_metrics'][metric]
                
                # For MSE and MAE, lower is better; for others, higher is better
                if metric in ['mse', 'mae']:
                    if best_value is None or value < best_value:
                        best_value = value
                        best_model = model_name
                else:
                    if best_value is None or value > best_value:
                        best_value = value
                        best_model = model_name
        
        if best_model:
            print(f"{metric.upper():<20}: {best_model} ({best_value:.6f})")


def plot_training_comparison(results):
    """Plot training history comparison for all models"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot training loss
        for model_name, result in results.items():
            if 'history' in result:
                history = result['history']
                if 'loss' in history:
                    axes[0].plot(history['loss'], label=f'{model_name} (train)', linewidth=2)
                if 'val_loss' in history:
                    axes[0].plot(history['val_loss'], label=f'{model_name} (val)', 
                               linewidth=2, linestyle='--')
        
        axes[0].set_title('Training & Validation Loss Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot metrics comparison as bar chart
        model_names = []
        ssim_values = []
        psnr_values = []
        
        for model_name, result in results.items():
            if 'evaluation_metrics' in result:
                metrics = result['evaluation_metrics']
                model_names.append(model_name)
                ssim_values.append(metrics.get('ssim', 0))
                psnr_values.append(metrics.get('psnr', 0))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = axes[1].bar(x - width/2, ssim_values, width, label='SSIM', alpha=0.8)
        bars2 = axes[1].bar(x + width/2, [p/10 for p in psnr_values], width, 
                           label='PSNR (÷10)', alpha=0.8)
        
        axes[1].set_title('SSIM and PSNR Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Score')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar, psnr_val in zip(bars2, psnr_values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{psnr_val:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Training comparison plot saved as 'model_comparison.png'")
        
    except Exception as e:
        print(f"Error creating training comparison plot: {e}")


def plot_spectrogram_examples(results, num_examples=3):
    """Plot spectrogram examples from each model"""
    try:
        # Load some test data for visualization
        spec_dir = os.path.join(os.getcwd(), "spec_arrays")
        if not os.path.exists(spec_dir):
            print("⚠️ No test data available for spectrogram visualization")
            return
        
        # Try to load test data
        try:
            test_clean = np.load(os.path.join(spec_dir, "test_clean_stft_norm.npy"))
            test_noisy = np.load(os.path.join(spec_dir, "test_noisy_stft_norm.npy"))
        except:
            print("⚠️ Test data not found, using training data for visualization")
            test_clean = np.load(os.path.join(spec_dir, "train_clean_stft_norm.npy"))[:num_examples]
            test_noisy = np.load(os.path.join(spec_dir, "train_noisy_stft_norm.npy"))[:num_examples]
        
        num_models = len(results)
        if num_models == 0:
            return
        
        fig, axes = plt.subplots(num_examples, num_models + 2, 
                               figsize=(4 * (num_models + 2), 3 * num_examples))
        
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_examples, len(test_clean))):
            # Plot original noisy
            im = axes[i, 0].imshow(test_noisy[i].squeeze(), aspect='auto', cmap='viridis')
            axes[i, 0].set_title('Noisy Input' if i == 0 else '')
            axes[i, 0].set_ylabel(f'Example {i+1}')
            
            # Plot ground truth clean
            axes[i, 1].imshow(test_clean[i].squeeze(), aspect='auto', cmap='viridis')
            axes[i, 1].set_title('Clean Target' if i == 0 else '')
            
            # Plot predictions from each model (if available in results)
            col = 2
            for model_name, result in results.items():
                if 'sample_predictions' in result and i < len(result['sample_predictions']):
                    pred = result['sample_predictions'][i]
                    axes[i, col].imshow(pred.squeeze(), aspect='auto', cmap='viridis')
                    axes[i, col].set_title(f'{model_name} Output' if i == 0 else '')
                else:
                    # If no predictions available, show placeholder
                    axes[i, col].text(0.5, 0.5, 'No\nPrediction\nAvailable', 
                                    ha='center', va='center', transform=axes[i, col].transAxes)
                    axes[i, col].set_title(f'{model_name} Output' if i == 0 else '')
                col += 1
            
            # Remove axis ticks for cleaner look
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('spectrogram_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Spectrogram comparison saved as 'spectrogram_comparison.png'")
        
    except Exception as e:
        print(f"Error creating spectrogram comparison: {e}")


def generate_audio_samples(results):
    """Generate audio samples from denoised spectrograms"""
    try:
        print("\n🎵 Generating audio samples...")
        
        # Create audio output directory
        audio_dir = "generated_audio"
        os.makedirs(audio_dir, exist_ok=True)
        
        for model_name, result in results.items():
            if 'sample_predictions' in result:
                print(f"Processing {model_name} audio samples...")
                
                for i, pred_spec in enumerate(result['sample_predictions'][:3]):  # First 3 samples
                    try:
                        # Convert normalized spectrogram back to audio
                        # This is a simplified conversion - you may need to adjust based on your preprocessing
                        audio_signal = librosa.istft(pred_spec.squeeze().T)
                        
                        # Save audio file
                        filename = f"{model_name.lower().replace('-', '_').replace(' ', '_')}_sample_{i+1}.wav"
                        filepath = os.path.join(audio_dir, filename)
                        sf.write(filepath, audio_signal, 22050)
                        
                        print(f"  ✓ Saved: {filename}")
                        
                    except Exception as e:
                        print(f"  ⚠️ Error generating audio for {model_name} sample {i+1}: {e}")
        
        print(f"🎵 Audio samples saved in '{audio_dir}/' directory")
        
    except Exception as e:
        print(f"Error generating audio samples: {e}")


def main():
    """Main comparison function"""
    print("🔍 AUDIO DENOISING MODEL COMPARISON")
    print("="*50)
    
    # Load saved results from individual model runs
    results = load_saved_results()
    
    if not results:
        print("\n❌ No model results found!")
        print("Please run the individual model training scripts first:")
        print("  python3 unet_model.py")
        print("  python3 attention_unet_model.py") 
        print("  python3 cnn_transformer_model.py")
        return
    
    print(f"\n✅ Found results for {len(results)} models: {', '.join(results.keys())}")
    
    # Compare metrics
    compare_metrics(results)
    
    # Plot training comparison
    print("\n📊 Creating training comparison plots...")
    plot_training_comparison(results)
    
    # Plot spectrogram examples
    print("\n🖼️ Creating spectrogram comparison...")
    plot_spectrogram_examples(results)
    
    # Generate audio samples
    generate_audio_samples(results)
    
    print("\n✅ Comparison complete!")
    print("Generated files:")
    print("  - model_comparison.png (training history and metrics)")
    print("  - spectrogram_comparison.png (visual comparison)")
    print("  - generated_audio/ (denoised audio samples)")


if __name__ == "__main__":
    main()