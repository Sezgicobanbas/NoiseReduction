# Complete Data Preprocessing Pipeline for Audio Denoising
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

from concurrent.futures import ProcessPoolExecutor

# Signal processing
from scipy.signal import resample_poly
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Deep learning imports (for mixed precision)
import tensorflow as tf
from keras import mixed_precision

# Set mixed precision policy to float32 for stability
mixed_precision.set_global_policy('float32')
tf.keras.backend.set_floatx('float32')

import kagglehub


class Config:
    """
    Configuration for dataset paths, audio parameters, and training hyperparameters.
    Update paths as needed for your environment.
    """
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    OUTPUT_DIR = os.path.join(current_dir, "audio_denoising_outputs")

    DATASET_PATH = kagglehub.dataset_download("manuelefavero/tiscode-dataset")
    DATASET_DIR = os.path.join(DATASET_PATH, "0-1000")

    URBANSOUND_PATH = kagglehub.dataset_download("chrisfilo/urbansound8k")
    URBANSOUND_DIR = URBANSOUND_PATH 

    CLEAN_DIR = os.path.join(OUTPUT_DIR, "CleanData")
    NOISY_DIR = os.path.join(OUTPUT_DIR, "NoisyData")

    SAMPLE_RATE = 44100
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    WIN_LENGTH = 2048
    TIME_STEPS = 436 

    EPOCHS = 100
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-4

    SNR_RANGE = (5, 15)


class UrbanNoiseLoader:
    def __init__(self, urbansound_dir, sample_rate=44100, cache_size=100):
        self.urbansound_dir = urbansound_dir
        self.sample_rate = sample_rate
        self.cache_size = cache_size
        self.noise_cache = {}
        self.noise_files = []
        self.noise_categories = {}
        self.noise_file_info = {}  # Store fold and category info for each file
        self._load_noise_files()
        print(f"UrbanSound8k Noise Loader initialized with {len(self.noise_files)} files")

    def _load_noise_files(self):
        if not os.path.exists(self.urbansound_dir):
            print(f"Error: UrbanSound8k directory not found at {self.urbansound_dir}")
            print("Cannot proceed without UrbanSound8k dataset!")
            return

        audio_extensions = ('.wav',)
        for root, dirs, files in tqdm(os.walk(self.urbansound_dir), desc="Loading UrbanSound8K noise files"):
            for file in files:
                if file.lower().endswith(audio_extensions):
                    full_path = os.path.join(root, file)
                    self.noise_files.append(full_path)
                    folder_name = os.path.basename(root).lower()
                    self.noise_file_info[full_path] = {
                        'folder': folder_name,
                        'filename': file
                    }
                    if folder_name not in self.noise_categories:
                        self.noise_categories[folder_name] = []
                    self.noise_categories[folder_name].append(full_path)

        if not self.noise_files:
            print("Error: No audio files found in UrbanSound8k directory!")
            print("Please check the dataset path and ensure it contains audio files.")
            return

        print(f"Found {len(self.noise_files)} urban noise files")
        print(f"Categories: {list(self.noise_categories.keys())}")

    def get_random_noise_with_info(self, target_length, noise_category=None):
        """Returns noise audio and information about the selected noise file"""
        if not self.noise_files:
            print("Error: No urban noise files available!")
            return None, None

        if noise_category and noise_category in self.noise_categories:
            available_files = self.noise_categories[noise_category]
        else:
            available_files = self.noise_files

        if not available_files:
            print(f"Error: No files found for category {noise_category}")
            return None, None

        noise_file = random.choice(available_files)

        # Get noise info for filename
        noise_info = self.noise_file_info[noise_file]

        if noise_file in self.noise_cache:
            noise_audio = self.noise_cache[noise_file]
        else:
            try:
                noise_audio, sr = librosa.load(noise_file, sr=self.sample_rate)

                if len(self.noise_cache) < self.cache_size:
                    self.noise_cache[noise_file] = noise_audio
                else:
                    oldest_key = next(iter(self.noise_cache))
                    del self.noise_cache[oldest_key]
                    self.noise_cache[noise_file] = noise_audio

            except Exception as e:
                print(f"Error loading {noise_file}: {e}")
                return None, None

        prepared_noise = self._prepare_noise_segment(noise_audio, target_length)
        return prepared_noise, noise_info

    def get_random_noise(self, target_length, noise_category=None):
        """Original method for backward compatibility"""
        noise, _ = self.get_random_noise_with_info(target_length, noise_category)
        return noise

    def _prepare_noise_segment(self, noise_audio, target_length):
        if len(noise_audio) == 0:
            print("Error: Loaded noise audio is empty")
            return None

        if len(noise_audio) >= target_length:
            start_idx = random.randint(0, len(noise_audio) - target_length)
            return noise_audio[start_idx:start_idx + target_length]
        else:
            if len(noise_audio) < target_length // 4:
                repeat_times = (target_length // len(noise_audio)) + 1
                extended = np.tile(noise_audio, repeat_times)
                return extended[:target_length]
            else:
                extended = np.tile(noise_audio, (target_length // len(noise_audio)) + 1)
                extended = extended[:target_length]

                fade_len = min(512, len(noise_audio) // 10)
                if fade_len > 0:
                    for i in range(len(noise_audio), len(extended), len(noise_audio)):
                        if i + fade_len < len(extended):
                            fade_out = np.linspace(1, 0, fade_len)
                            fade_in = np.linspace(0, 1, fade_len)
                            extended[i-fade_len:i] *= fade_out
                            extended[i:i+fade_len] *= fade_in

                return extended


def initialize_urban_noise_loader_working(config):
    """Working version of the urban noise loader initialization"""
    
    try:
        print(f"Initializing UrbanNoiseLoader with path: {config.URBANSOUND_DIR}")
        urban_noise_loader = UrbanNoiseLoader(
            config.URBANSOUND_DIR,
            config.SAMPLE_RATE,
            cache_size=100
        )
        
        if not urban_noise_loader.noise_files:
            print("Fatal Error: No urban noise files loaded!")
            return None
            
        print(f"✓ Successfully initialized with {len(urban_noise_loader.noise_files)} noise files")
        return urban_noise_loader
        
    except Exception as e:
        print(f"Error initializing urban noise loader: {e}")
        import traceback
        traceback.print_exc()
        return None


def add_urban_noise_with_info(audio, snr_range=None, dynamic_snr=True, noise_category=None, urban_noise_loader=None):
    """Add urban noise and return both noisy audio and noise information"""
    if snr_range is None:
        snr_range = (5, 15)  # Default SNR range

    if urban_noise_loader is None:
        print("Error: Urban noise loader not provided!")
        return audio, None

    if dynamic_snr:
        # Quieter signals get higher SNR (less noise)
        audio_rms = np.sqrt(np.mean(audio**2))
        if audio_rms < 0.1:
            snr_adjustment = 3
        elif audio_rms > 0.5:
            snr_adjustment = -2
        else:
            snr_adjustment = 0

        adjusted_range = (snr_range[0] + snr_adjustment, snr_range[1] + snr_adjustment)
    else:
        adjusted_range = snr_range

    snr_db = np.random.uniform(adjusted_range[0], adjusted_range[1])

    rms_signal = np.sqrt(np.mean(audio**2))
    if rms_signal == 0:
        return audio, None

    snr_linear = 10**(snr_db / 10)
    rms_noise = rms_signal / np.sqrt(snr_linear)

    noise, noise_info = urban_noise_loader.get_random_noise_with_info(len(audio), noise_category)

    if noise is None:
        print("Warning: Could not load urban noise, returning original audio")
        return audio, None

    current_rms = np.sqrt(np.mean(noise**2))
    if current_rms > 0:
        noise = noise * (rms_noise / current_rms)

        if random.random() < 0.3:
            noise = apply_spectral_shaping(noise, audio, 44100)
    else:
        print("Warning: Loaded noise has zero RMS, returning original audio")
        return audio, None

    return audio + noise, noise_info


def apply_spectral_shaping(noise, reference_audio, sample_rate):
    try:
        ref_mel = librosa.feature.melspectrogram(y=reference_audio, sr=sample_rate, n_mels=32)
        ref_spectrum = np.mean(ref_mel, axis=1)

        noise_mel = librosa.feature.melspectrogram(y=noise, sr=sample_rate, n_mels=32)
        noise_spectrum = np.mean(noise_mel, axis=1)

        shaping_factor = np.sqrt(ref_spectrum / (noise_spectrum + 1e-10))
        shaping_factor = np.clip(shaping_factor, 0.5, 2.0)  # Limit the shaping

        return noise * np.mean(shaping_factor)
    except:
        return noise


def preprocess_audio(audio):
    audio = audio - np.mean(audio)
    max_val = np.max(np.abs(audio)) + 1e-9
    return audio / max_val


def save_noisy_and_clean_with_multiple_urban_noises_fixed(config, versions_per_file=1):
    """
    Fixed version with proper error handling for urban noise loader
    """
    dataset_dir = config.DATASET_DIR

    # Use the working initialization function
    urban_noise_loader = initialize_urban_noise_loader_working(config)
    if urban_noise_loader is None:
        print("CRITICAL ERROR: Could not initialize urban noise loader!")
        print("This usually means:")
        print("1. UrbanSound8K dataset not found")
        print("2. No .wav files in the dataset directory") 
        print("3. Dataset download failed")
        print(f"Expected path: {config.URBANSOUND_DIR}")
        return

    os.makedirs(config.CLEAN_DIR, exist_ok=True)
    os.makedirs(config.NOISY_DIR, exist_ok=True)

    supported_exts = ('.wav', '.mp3', '.flac')
    files = sorted(f for f in os.listdir(dataset_dir) if f.lower().endswith(supported_exts))
    successful_files = 0
    total_noisy_files = 0

    print(f"Processing {len(files)} files with {versions_per_file} UrbanSound8k noises each...")

    for i, filename in enumerate(files):
        try:
            file_path = os.path.join(dataset_dir, filename)
            audio, sr = sf.read(file_path)

            audio = np.asarray(audio, dtype=np.float32)

            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            if sr != config.SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)

            audio = preprocess_audio(audio)

            # Save clean version
            clean_out_path = os.path.join(config.CLEAN_DIR, filename)
            sf.write(clean_out_path, audio, config.SAMPLE_RATE)

            # Generate multiple noisy versions
            base_name, ext = os.path.splitext(filename)

            for version in range(versions_per_file):
                noisy_audio, noise_info = add_urban_noise_with_info(
                    audio, 
                    snr_range=config.SNR_RANGE,
                    dynamic_snr=True,
                    urban_noise_loader=urban_noise_loader
                )

                if noise_info is not None:
                    fold_info = noise_info['folder']  # e.g., "fold3"
                    noisy_filename = f"{base_name}_urban{version+1}_{fold_info}{ext}"
                    noisy_out_path = os.path.join(config.NOISY_DIR, noisy_filename)
                    sf.write(noisy_out_path, noisy_audio, config.SAMPLE_RATE)
                    total_noisy_files += 1
                else:
                    print(f"Warning: Could not add urban noise to {filename} version {version+1}")

            successful_files += 1

            if (i + 1) % 100 == 0:  # Show progress every 100 files
                print(f"Processed {i + 1}/{len(files)} files... Generated {total_noisy_files} noisy files so far")

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
            continue

    print(f"\nSuccessfully processed {successful_files}/{len(files)} clean files.")
    print(f"Generated {total_noisy_files} noisy files with UrbanSound8k noise.")
    
    # Fixed: Check if urban_noise_loader exists before accessing its attributes
    if urban_noise_loader is not None:
        print(f"Urban noise cache contains {len(urban_noise_loader.noise_cache)} loaded sounds")
    else:
        print("Urban noise loader was not properly initialized")


def pair_noisy_with_clean(noisy_dir, clean_dir):
    """
    Efficiently pair each noisy file with its matching clean file.
    Returns a list of (clean_path, noisy_path) tuples.
    """
    noisy_files = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.wav', '.flac', '.mp3'))]
    clean_files = set(os.listdir(clean_dir))
    pairs = []
    pattern = re.compile(r'^(.*)_urban\d+_.*(\.\w+)$')
    for noisy_file in noisy_files:
        match = pattern.match(noisy_file)
        if match:
            base_name = match.group(1)
            ext = match.group(2)
            clean_file = base_name + ext
            if clean_file in clean_files:
                pairs.append((os.path.join(clean_dir, clean_file), os.path.join(noisy_dir, noisy_file)))
    print(f"Found {len(pairs)} pairs of clean and noisy files.")
    return pairs

def batch_files_to_stft_spectrograms(file_pairs, sample_rate=44100, n_fft=2048, hop_length=512, win_length=2048, batch_size=500):
    """
    Memory-efficient STFT conversion with batch processing to avoid memory issues.
    """
    import time
    
    total_files = len(file_pairs)
    all_clean_specs = []
    all_noisy_specs = []
    failed_files = []
    
    print(f"Starting STFT conversion for {total_files} file pairs...")
    print(f"Processing in batches of {batch_size} to conserve memory")
    
    start_time = time.time()
    
    # Process in batches to avoid memory issues
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_pairs = file_pairs[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(total_files-1)//batch_size + 1}")
        print(f"Files {batch_start+1} to {batch_end}")
        
        clean_specs = []
        noisy_specs = []
        
        # Progress bar for current batch
        progress_bar = tqdm(
            enumerate(batch_pairs), 
            total=len(batch_pairs),
            desc=f"STFT Batch {batch_start//batch_size + 1}",
            unit="pairs"
        )
        
        for i, (clean_path, noisy_path) in progress_bar:
            try:
                # Load clean audio
                clean_audio, sr_c = librosa.load(clean_path, sr=sample_rate)
                # Load noisy audio
                noisy_audio, sr_n = librosa.load(noisy_path, sr=sample_rate)
                
                # Ensure same length (pad shorter one)
                max_len = max(len(clean_audio), len(noisy_audio))
                if len(clean_audio) < max_len:
                    clean_audio = np.pad(clean_audio, (0, max_len - len(clean_audio)))
                if len(noisy_audio) < max_len:
                    noisy_audio = np.pad(noisy_audio, (0, max_len - len(noisy_audio)))
                
                # Compute STFT spectrograms
                clean_stft = librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
                noisy_stft = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
                
                clean_specs.append(clean_stft)
                noisy_specs.append(noisy_stft)
                
            except Exception as e:
                error_msg = f"Error processing {os.path.basename(clean_path)}: {str(e)}"
                failed_files.append((clean_path, noisy_path, str(e)))
                print(f"\n⚠️  {error_msg}")
                continue
        
        progress_bar.close()
        
        # Convert batch to numpy arrays and add to main list
        if clean_specs:
            batch_clean = np.array(clean_specs, dtype=np.complex64)
            batch_noisy = np.array(noisy_specs, dtype=np.complex64)
            all_clean_specs.append(batch_clean)
            all_noisy_specs.append(batch_noisy)
            
            print(f"Batch {batch_start//batch_size + 1} completed: {len(clean_specs)} spectrograms")
            print(f"Batch shape: {batch_clean.shape}")
    
    # Concatenate all batches
    print(f"\n📊 Concatenating {len(all_clean_specs)} batches...")
    if all_clean_specs:
        clean_specs = np.concatenate(all_clean_specs, axis=0)
        noisy_specs = np.concatenate(all_noisy_specs, axis=0)
    else:
        clean_specs = np.array([])
        noisy_specs = np.array([])
    
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n🎉 STFT Conversion Complete!")
    print(f"  ✅ Successfully processed: {len(clean_specs)}/{total_files} pairs")
    print(f"  ❌ Failed files: {len(failed_files)}")
    if len(clean_specs) > 0:
        print(f"  📏 Final shape: {clean_specs.shape}")
        print(f"  💾 Memory usage: {clean_specs.nbytes / 1024**2:.1f} MB per array")
    print(f"  ⏱️  Total time: {total_time/60:.2f} minutes")
    if len(clean_specs) > 0:
        print(f"  ⚡ Average: {total_time/len(clean_specs):.2f}s per file")
    
    if failed_files:
        print(f"\n⚠️  Failed files details:")
        for i, (clean_path, noisy_path, error) in enumerate(failed_files[:5]):  # Show first 5 failures
            print(f"  {i+1}. {os.path.basename(clean_path)}: {error}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more failures")
    
    return clean_specs, noisy_specs


def normalize_stft_spectrograms(stft_specs, db_min=-100.0, db_max=0.0):
    """
    Normalize a list/array of STFT spectrograms (in complex form) to [0, 1] range after converting to dB.
    Returns normalized specs and the min/max used for each.
    """
    norm_specs = []
    min_vals = []
    max_vals = []
    for stft in stft_specs:
        db_spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        norm = (db_spec - db_min) / (db_max - db_min) if (db_max - db_min) > 1e-9 else np.zeros_like(db_spec)
        norm = np.clip(norm, 0, 1)
        norm_specs.append(norm)
        min_vals.append(db_min)
        max_vals.append(db_max)
    return np.array(norm_specs), np.array(min_vals), np.array(max_vals)


def save_spectrogram_arrays(clean_stft_norm, noisy_stft_norm, prefix="train"):
    """Save spectrogram arrays as .npy files for efficient memory usage"""
    spec_dir = os.path.join(os.getcwd(), "spec_arrays")
    os.makedirs(spec_dir, exist_ok=True)
    
    clean_fp = os.path.join(spec_dir, f"{prefix}_clean_stft_norm.npy")
    noisy_fp = os.path.join(spec_dir, f"{prefix}_noisy_stft_norm.npy")
    np.save(clean_fp, clean_stft_norm.astype(np.float16))
    np.save(noisy_fp, noisy_stft_norm.astype(np.float16))
    print(f"Saved: {clean_fp}, {noisy_fp}")
    
    return clean_fp, noisy_fp


def run_complete_preprocessing_pipeline(versions_per_file=1):
    """
    Run the complete data preprocessing pipeline
    """
    print("🚀 Starting Complete Data Preprocessing Pipeline...")
    print("="*80)
    
    # Initialize configuration
    config = Config()
    
    print("Configuration loaded:")
    print(f"  Dataset path: {config.DATASET_DIR}")
    print(f"  UrbanSound path: {config.URBANSOUND_DIR}")
    print(f"  Output directory: {config.OUTPUT_DIR}")
    print(f"  Sample rate: {config.SAMPLE_RATE}")
    print(f"  Versions per file: {versions_per_file}")
    
    # Step 1: Generate noisy data
    print(f"\n📁 Step 1: Generating noisy data with {versions_per_file} versions per file...")
    save_noisy_and_clean_with_multiple_urban_noises_fixed(config, versions_per_file=versions_per_file)
    
    # Step 2: Pair clean and noisy files
    print(f"\n🔗 Step 2: Pairing clean and noisy files...")
    pairs = pair_noisy_with_clean(config.NOISY_DIR, config.CLEAN_DIR)
    
    if len(pairs) == 0:
        print("❌ No file pairs found! Check your data generation step.")
        return None
    
    # Step 3: Convert to STFT spectrograms
    print(f"\n🎵 Step 3: Converting {len(pairs)} pairs to STFT spectrograms...")
    clean_stft_specs, noisy_stft_specs = batch_files_to_stft_spectrograms(
        pairs, 
        sample_rate=config.SAMPLE_RATE, 
        n_fft=config.N_FFT, 
        hop_length=config.HOP_LENGTH, 
        win_length=config.WIN_LENGTH
    )
    
    # Step 4: Normalize spectrograms
    print(f"\n📊 Step 4: Normalizing spectrograms...")
    clean_stft_norm, clean_stft_min, clean_stft_max = normalize_stft_spectrograms(clean_stft_specs)
    noisy_stft_norm, noisy_stft_min, noisy_stft_max = normalize_stft_spectrograms(noisy_stft_specs)
    
    print(f"Normalized spectrogram shapes:")
    print(f"  Clean: {clean_stft_norm.shape}")
    print(f"  Noisy: {noisy_stft_norm.shape}")
    
    # Step 5: Save processed data
    print(f"\n💾 Step 5: Saving processed spectrograms...")
    clean_path, noisy_path = save_spectrogram_arrays(clean_stft_norm, noisy_stft_norm, prefix="train")
    
    print(f"\n✅ Complete preprocessing pipeline finished!")
    print(f"📁 Results saved to:")
    print(f"  - Clean audio: {config.CLEAN_DIR}")
    print(f"  - Noisy audio: {config.NOISY_DIR}")
    print(f"  - Clean spectrograms: {clean_path}")
    print(f"  - Noisy spectrograms: {noisy_path}")
    
    return {
        'config': config,
        'pairs': pairs,
        'clean_stft_specs': clean_stft_specs,
        'noisy_stft_specs': noisy_stft_specs,
        'clean_stft_norm': clean_stft_norm,
        'noisy_stft_norm': noisy_stft_norm,
        'clean_path': clean_path,
        'noisy_path': noisy_path
    }


if __name__ == "__main__":
    # Run the complete preprocessing pipeline
    print("Running complete data preprocessing pipeline...")
    
    # You can adjust the number of noisy versions per file here
    versions_per_file = 2
    
    results = run_complete_preprocessing_pipeline(versions_per_file=versions_per_file)
    
    if results is not None:
        print("\n🎉 Preprocessing completed successfully!")
    else:
        print("\n❌ Preprocessing failed. Please check the error messages above.")