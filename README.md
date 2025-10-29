# Audio Denoising Model Comparison Project

This project implements and compares three different deep learning models for audio denoising:
- **U-Net**:  Implementation with separable convolutions
- **Attention U-Net**: Enhanced U-Net with attention mechanisms  
- **CNN-Transformer**: Hybrid architecture combining CNN and Transformer

## Quick Start - NEW Workflow

### 1. Data Preprocessing
First, run the data preprocessing to download and prepare the datasets:
```bash
python3 data_preprocessing.py
cd /Users/sezgicobanbas/Desktop/Thesis && python3 data_preprocessing.py
```

### 2. Train Models Individually  
Run each model training script separately:
```bash
# Train U-Net model
python3 unet_model.py
cd /Users/sezgicobanbas/Desktop/Thesis && python3 unet_model.py

# Train Attention U-Net model  
python3 attention_unet_model.py
cd /Users/sezgicobanbas/Desktop/Thesis && python3 attention_unet_model.py

# Train CNN-Transformer model
python3 cnn_transformer_model.py
cd /Users/sezgicobanbas/Desktop/Thesis && python3 cnn_transformer_model.py
```

Each script will:
- Load and preprocess the data
- Train the model for 50 epochs with early stopping
- Evaluate performance on test data
- Save complete results to `model_results/` directory

### 3. Compare Results
After training all models, run the comparison script:
```bash
python3 main_comparison.py
```

This will:
- Load saved results from all models
- Compare performance metrics in detailed tables
- Generate training history plots
- Create spectrogram comparison visualizations
- Generate denoised audio samples

## 📁 Project Structure

```
├── data_preprocessing.py      # Data download and preprocessing pipeline
├── unet_model.py             # U-Net model implementation and training
├── attention_unet_model.py   # Attention U-Net model and training  
├── cnn_transformer_model.py  # CNN-Transformer model and training
├── main_comparison.py        # Results comparison and visualization
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── model_results/           # Saved training results (created automatically)
    ├── u_net_results.pkl
    ├── attention_u_net_results.pkl
    └── cnn_transformer_results.pkl
```

##  Generated Outputs

### Training Results (from individual model scripts)
Each model saves:
- Complete training history (loss/accuracy curves)
- Evaluation metrics (MSE, MAE, SSIM, PSNR, Cosine Similarity)
- Sample predictions for visualization
- Trained model weights (.h5 files)
- Training time and configuration

### Comparison Outputs (from main_comparison.py)
- `model_comparison.png`: Training curves and metrics comparison
- `spectrogram_comparison.png`: Visual spectrogram comparison  
- `generated_audio/`: Denoised audio samples from each model
- Detailed performance tables in terminal output

##  Technical Details

### Models
- **U-Net**: Uses separable convolutions for efficiency, ~2.7M parameters
- **Attention U-Net**: Adds attention gates for better feature selection, ~3.2M parameters  
- **CNN-Transformer**: Hybrid architecture with patch-based transformers, ~4.1M parameters

### Data Processing  
- **Datasets**: TIScode and UrbanSound8K via Kaggle
- **Audio Processing**: STFT spectrograms (1024 FFT, 256 hop length)
- **Noise Addition**: Synthetic noise overlays for training data
- **Batch Processing**: Memory-efficient 500-sample batches

### Training Configuration
- **Epochs**: 100 with early stopping (patience=10)
- **Batch Size**: 16 (adjustable based on memory)
- **Optimizer**: Adam with learning rate 5e-4
- **Mixed Precision**: Enabled for stability and speed

### Evaluation Metrics
- **MSE/MAE**: Reconstruction errors (lower is better)
- **SSIM**: Structural similarity index (higher is better)
- **PSNR**: Peak signal-to-noise ratio (higher is better)
- **Cosine Similarity**: Feature similarity measure (higher is better)

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Main dependencies:
- TensorFlow >= 2.8.0
- librosa >= 0.9.0
- soundfile >= 0.10.0
- matplotlib >= 3.5.0
- scikit-learn >= 1.0.0
- kaggle >= 1.5.0

##  Performance Comparison

The comparison script automatically identifies the best-performing model for each metric and displays results in formatted tables:

```
MODEL PERFORMANCE COMPARISON
================================================================================
Model                MSE          MAE          SSIM         PSNR         Cosine Sim  
------------------------------------------------------------------------------------
U-Net               0.001234     0.032156     0.945678     28.456789    0.967890    
Attention U-Net     0.001098     0.029876     0.956789     29.234567    0.971234    
CNN-Transformer     0.000987     0.028765     0.961234     30.123456    0.975678    

BEST PERFORMING MODELS BY METRIC
==================================================
MSE                 : CNN-Transformer (0.000987)
MAE                 : CNN-Transformer (0.028765)
SSIM                : CNN-Transformer (0.961234)
PSNR                : CNN-Transformer (30.123456)
COSINE_SIMILARITY   : CNN-Transformer (0.975678)
```

##  Audio Playback

Generated audio samples are saved in the `generated_audio/` directory:
```
generated_audio/
├── u_net_sample_1.wav
├── u_net_sample_2.wav
├── attention_u_net_sample_1.wav
├── attention_u_net_sample_2.wav
├── cnn_transformer_sample_1.wav
└── cnn_transformer_sample_2.wav
```

##  Tips for Best Results

1. **Hardware**: Use a GPU if available for faster training
2. **Memory**: Each model needs ~8-10GB RAM during training
3. **Storage**: Ensure ~15GB free space for datasets and results
4. **Time**: Each model takes 120-180 minutes to train 
5. **Kaggle Setup**: Configure API credentials for automatic dataset download

## Troubleshooting

### Common Issues

**No model results found**
```bash
 No model results found!
Please run the individual model training scripts first:
  python3 unet_model.py
  python3 attention_unet_model.py 
  python3 cnn_transformer_model.py
```
Solution: Train models individually before running comparison.

**Import/Dependency Errors**
```bash
pip install -r requirements.txt
```

**Memory Issues**
- Reduce batch size in model files (line ~20-30 in each model file)
- Close other applications to free up RAM

**Dataset Download Issues**
- Check Kaggle API credentials: `~/.kaggle/kaggle.json`
- Verify internet connection and Kaggle account access

**CUDA/GPU Issues**
- Models work on both CPU and GPU
- Check TensorFlow-GPU installation if using GPU

##  Workflow Summary

1. **Preprocessing** → Downloads data, creates spectrograms → `spec_arrays/`
2. **Individual Training** → Each model trains independently → `model_results/`  
3. **Comparison** → Loads results, compares, visualizes → `*.png` files + `generated_audio/`
