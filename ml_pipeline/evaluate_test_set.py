# ml_pipeline/evaluate_test_set.py
import os
import torch
import torchaudio
import soundfile as sf
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from model import StemExtractorUNet
import matplotlib.pyplot as plt

def calculate_si_sdr(reference, estimation):
    min_len = min(len(reference), len(estimation))
    ref = reference[:min_len]
    est = estimation[:min_len]

    # Zero-mean 
    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    # Scaling factor
    ref_energy = np.sum(ref ** 2) + 1e-8
    alpha = np.sum(est * ref) / ref_energy
    
    # Separate true target from noise
    target_component = alpha * ref
    noise_component = est - target_component

    target_power = np.sum(target_component ** 2)
    noise_power = np.sum(noise_component ** 2)
    
    return 10 * np.log10((target_power + 1e-8) / (noise_power + 1e-8))

def evaluate_full_test_set():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running full test set evaluation on {device}...")

    # Load model
    model = StemExtractorUNet(num_stems=4).to(device)
    weights_path = "unet_best.pt" 
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Error: could not find {weights_path}.")
        
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval() 

    test_dir = "musdb18hq/test"
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Could not find {test_dir}.")
        
    song_folders = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
    
    sample_rate = 44100
    chunk_samples = int(21.0 * sample_rate) 
    stems = ['vocals', 'drums', 'bass', 'other']
    
    # STFT parameters
    n_fft = 1024
    hop_length = 256
    window = torch.hann_window(n_fft).to(device)
    target_time_frames = 3456

    # Dictionary to store scores
    all_scores = {stem: [] for stem in stems}

    print(f"Found {len(song_folders)} songs in the test set. Evaluating...")
    
    for song_path in tqdm(song_folders, desc="Evaluating Songs"):
        mix_path = os.path.join(song_path, "mixture.wav")
        
        info = sf.info(mix_path)
        total_frames = info.frames
        
        start_frame = (total_frames // 2) - (chunk_samples // 2)
        if start_frame < 0:
            start_frame = 0 # Fallback for very short audio

        mix_audio, sr = torchaudio.load(mix_path, frame_offset=start_frame, num_frames=chunk_samples)
        if sr != sample_rate:
            mix_audio = torchaudio.transforms.Resample(sr, sample_rate)(mix_audio)
        mix_audio = mix_audio.to(device) 

        mix_stft_2d = torch.stft(mix_audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mix_stft = mix_stft_2d.unsqueeze(0) 
        mix_mag = torch.abs(mix_stft)

        original_freqs = mix_mag.shape[-2]
        cropped_mag = mix_mag[:, :, :512, :target_time_frames]
        cropped_stft = mix_stft[:, :, :512, :target_time_frames]

        with torch.no_grad(): 
            masks = model(cropped_mag) 

        pad_freq = original_freqs - 512
        padded_masks = F.pad(masks, (0, 0, 0, pad_freq)) 
        padded_stft = F.pad(cropped_stft, (0, 0, 0, pad_freq))
        separated_stft = padded_stft.unsqueeze(1) * padded_masks

        for i, stem in enumerate(stems):
            # iSTFT to get prediction back to audio domain
            stem_stft = separated_stft[0, i] 
            out_length = (target_time_frames - 1) * hop_length
            pred_audio = torch.istft(stem_stft, n_fft=n_fft, hop_length=hop_length, window=window, length=out_length)
            pred_audio_np = pred_audio.cpu().numpy().squeeze()

            # Ground truth
            true_path = os.path.join(song_path, f"{stem}.wav")
            true_audio, true_sr = torchaudio.load(true_path, frame_offset=start_frame, num_frames=chunk_samples)
            if true_sr != sample_rate:
                true_audio = torchaudio.transforms.Resample(true_sr, sample_rate)(true_audio)
            true_audio_np = true_audio.mean(dim=0).cpu().numpy().squeeze()
            
            if len(pred_audio_np.shape) > 1:
                pred_audio_np = pred_audio_np.mean(axis=0)

            peak_amplitude = np.max(np.abs(true_audio_np))
            if peak_amplitude < 1e-3:
                continue

            score = calculate_si_sdr(true_audio_np, pred_audio_np)
            all_scores[stem].append(score)

    print(f" Macro avg SI-SDR (Across {len(song_folders)} test songs)")
    print(f" {'Stem':<10} | {'Mean (dB)':<10} | {'Median (dB)':<10}")
    
    for stem in stems:
        scores = np.array(all_scores[stem])
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        print(f" {stem.capitalize():<10} | {mean_score:>7.2f} dB | {median_score:>7.2f} dB")
    
    plot_si_sdr_distributions(all_scores)

def plot_si_sdr_distributions(all_scores, output_img="si_sdr_boxplot.png"):
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    images_dir = os.path.join(project_root, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    out_path = os.path.join(images_dir, output_img)
    
    stems = list(all_scores.keys())
    data = [all_scores[stem] for stem in stems]
    
    plt.figure(figsize=(6, 4))
    
    box = plt.boxplot(data, patch_artist=True, tick_labels=[s.capitalize() for s in stems], 
                      medianprops=dict(color='black', linewidth=1.5),
                      flierprops=dict(marker='o', color='red', alpha=0.5, markersize=5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    # 0 dB means target and noise are equal volume
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='0 dB Baseline')
    
    # plt.title('SI-SDR Score Distribution Across MUSDB18 Test Set', fontsize=14, pad=15)
    plt.ylabel('SI-SDR (dB)', fontsize=14)
    plt.xlabel('Extracted Stem', fontsize=14)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved SI-SDR box plot to: '{out_path}'")
    
if __name__ == "__main__":
    evaluate_full_test_set()

