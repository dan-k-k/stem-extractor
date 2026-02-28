# ml_pipeline/generate_plots.py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")

os.makedirs(IMAGES_DIR, exist_ok=True)

def plot_training_curve(csv_file="training_log.csv", output_img="loss_curve.png"):
    csv_path = os.path.join(SCRIPT_DIR, csv_file)
    out_path = os.path.join(IMAGES_DIR, output_img)

    if not os.path.exists(csv_path):
        print(f"Could not find {csv_path}.")
        return

    epochs, train_losses, val_losses = [], [], []

    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('L1 Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300)
    print(f"Loss curve saved to: '{out_path}'")

def plot_spectrogram_comparisons(mix_name="true_mix.wav", stem_name="vocals.wav", output_img="spectrogram_comparison.png"):
    
    mix_path = os.path.join(SCRIPT_DIR, "inference_output", mix_name)
    stem_path = os.path.join(SCRIPT_DIR, "inference_output", stem_name)
    out_path = os.path.join(IMAGES_DIR, output_img)

    if not os.path.exists(mix_path) or not os.path.exists(stem_path):
        print(f"Could not find audio files to compare.\n Mix: {mix_path}\n Stem: {stem_path}")
        return

    y_mix, sr = librosa.load(mix_path, sr=None, mono=True)
    y_stem, _ = librosa.load(stem_path, sr=sr, mono=True)

    D_mix = librosa.amplitude_to_db(np.abs(librosa.stft(y_mix)), ref=np.max)
    D_stem = librosa.amplitude_to_db(np.abs(librosa.stft(y_stem)), ref=np.max)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharey=True)

    # Plot Mix
    img1 = librosa.display.specshow(D_mix, y_axis='log', x_axis='time', sr=sr, ax=ax[0], cmap='magma')
    ax[0].set_title('Original mix', fontsize=14)
    ax[0].set_xlabel('Time (s)', fontsize=12)
    ax[0].set_ylabel('Frequency (Hz)', fontsize=12)

    # Plot Isolated Stem
    img2 = librosa.display.specshow(D_stem, y_axis='log', x_axis='time', sr=sr, ax=ax[1], cmap='magma')
    ax[1].set_title(f'Extracted {stem_name.replace(".wav", "").capitalize()}', fontsize=14)
    ax[1].set_xlabel('Time (s)', fontsize=12)

    # fig.colorbar(img1, ax=ax, format="%+2.0f dB", label="Amplitude (dB)", pad=0.05)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Spectrogram comparison saved to: '{out_path}'")

if __name__ == "__main__":
    plot_training_curve()
    plot_spectrogram_comparisons()

