# ml_pipeline/inference.py
import os
import random
import torch
import torchaudio
import soundfile as sf 
import torch.nn.functional as F
from model import StemExtractorUNet

# ml_pipeline/inference.py
import os
import torch
import torchaudio
import torch.nn.functional as F
from model import StemExtractorUNet

def infer():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🎸 Running Inference on {device}.")

    model = StemExtractorUNet(num_stems=4).to(device)
    weights_path = "unet_best.pt" 
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Error: Could not find {weights_path}.")
        
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval() 

    # --- UPDATED: Hardcoded target file ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mix_path = os.path.join(script_dir, "musdb18hq", "test", "Arise - Run Run Run", "mixture.wav")
    
    if not os.path.exists(mix_path):
        raise FileNotFoundError(f"❌ Could not find {mix_path}.")
        
    print(f"🎧 Loading track: {os.path.basename(mix_path)}")

    sample_rate = 44100
    chunk_samples = int(21.0 * sample_rate) 
    
    # --- UPDATED: Hardcoded start time (28 seconds) ---
    start_seconds = 28.0
    start_frame = int(start_seconds * sample_rate)

    mix_audio, sr = torchaudio.load(mix_path, frame_offset=start_frame, num_frames=chunk_samples)
    if sr != sample_rate:
        mix_audio = torchaudio.transforms.Resample(sr, sample_rate)(mix_audio)
    
    mix_audio = mix_audio.to(device) 

    # STFT 
    n_fft = 1024
    hop_length = 256
    window = torch.hann_window(n_fft).to(device)

    mix_stft_2d = torch.stft(mix_audio, n_fft=n_fft, hop_length=hop_length, 
                             window=window, return_complex=True)
    mix_stft = mix_stft_2d.unsqueeze(0) 
    mix_mag = torch.abs(mix_stft)

    original_freqs = mix_mag.shape[-2]
    
    target_time_frames = 3456
    cropped_mag = mix_mag[:, :, :512, :target_time_frames]
    cropped_stft = mix_stft[:, :, :512, :target_time_frames]

    with torch.no_grad(): 
        masks = model(cropped_mag) # Shape: [1, 4, 2, 512, 1536]

    pad_freq = original_freqs - 512
    padded_masks = F.pad(masks, (0, 0, 0, pad_freq)) 

    # Apply masks to the cropped STFT
    padded_stft = F.pad(cropped_stft, (0, 0, 0, pad_freq))
    separated_stft = padded_stft.unsqueeze(1) * padded_masks

    stems = ['vocals', 'drums', 'bass', 'other']
    output_dir = "inference_output"
    os.makedirs(output_dir, exist_ok=True)

    print("\nExporting Stems...")
    for i, stem in enumerate(stems):
        stem_stft = separated_stft[0, i] 
        
        out_length = (target_time_frames - 1) * hop_length
        stem_audio = torch.istft(stem_stft, n_fft=n_fft, hop_length=hop_length, 
                                 window=window, length=out_length)
        
        out_path = os.path.join(output_dir, f"{stem}.wav")
        torchaudio.save(out_path, stem_audio.cpu(), sample_rate)
        print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    infer()

