# ml_pipeline/inference.py
import os
import torch
import torchaudio
import torch.nn.functional as F
from model import StemExtractorUNet

def infer():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🎸 Running Inference on: {device}")

    model = StemExtractorUNet(num_stems=4).to(device)
    
    weights_path = "unet_epoch_10.pt" 
    if not os.path.exists(weights_path):
        print(f"Could not find {weights_path}. Did training finish?")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() 

    sample_rate = 44100
    chunk_samples = int(3.0 * sample_rate) 
    mix_path = "musdb18hq/SampleSong/mixture.wav"
    
    mix_audio, sr = torchaudio.load(mix_path, num_frames=chunk_samples)
    if sr != sample_rate:
        mix_audio = torchaudio.transforms.Resample(sr, sample_rate)(mix_audio)
    
    mix_audio = mix_audio.to(device) 

    n_fft = 1024
    hop_length = 256
    window = torch.hann_window(n_fft).to(device)

    mix_stft_2d = torch.stft(mix_audio, n_fft=n_fft, hop_length=hop_length, 
                             window=window, return_complex=True)
    
    mix_stft = mix_stft_2d.unsqueeze(0) 
    
    mix_mag = torch.abs(mix_stft)

    original_freqs = mix_mag.shape[-2]
    original_frames = mix_mag.shape[-1]
    
    cropped_mag = mix_mag[:, :, :512, :512]

    with torch.no_grad(): 
        masks = model(cropped_mag) # Shape: [1, 4 (stems), 2 (channels), 512, 512]

    pad_freq = original_freqs - 512
    pad_time = original_frames - 512
    
    padded_masks = F.pad(masks, (0, pad_time, 0, pad_freq)) 

    separated_stft = mix_stft.unsqueeze(1) * padded_masks

    stems = ['vocals', 'drums', 'bass', 'other']
    output_dir = "inference_output"
    os.makedirs(output_dir, exist_ok=True)

    print("\n🎧 Exporting Stems...")
    for i, stem in enumerate(stems):
        stem_stft = separated_stft[0, i] 
        
        # convert back to waveforms
        stem_audio = torch.istft(stem_stft, n_fft=n_fft, hop_length=hop_length, 
                                 window=window, length=chunk_samples)
        
        out_path = os.path.join(output_dir, f"{stem}.wav")
        torchaudio.save(out_path, stem_audio.cpu(), sample_rate)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    infer()

