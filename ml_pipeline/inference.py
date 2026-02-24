# ml_pipeline/inference.py
import os
import torch
import torchaudio
import torch.nn.functional as F
from model import StemExtractorUNet

def infer():
    # 1. Setup Device & Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🎸 Running Inference on: {device}")

    model = StemExtractorUNet(num_stems=4).to(device)
    
    # Load the absolute latest weights (change the number if you let it run past epoch 10)
    # Note: we use map_location to ensure it loads safely onto your current device
    weights_path = "unet_epoch_10.pt" 
    if not os.path.exists(weights_path):
        print(f"Could not find {weights_path}. Did training finish?")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # CRITICAL: Disables dropout and batch norm tracking

    # 2. Load the Audio 
    sample_rate = 44100
    chunk_samples = int(3.0 * sample_rate) 
    mix_path = "musdb18hq/SampleSong/mixture.wav"
    
    mix_audio, sr = torchaudio.load(mix_path, num_frames=chunk_samples)
    if sr != sample_rate:
        mix_audio = torchaudio.transforms.Resample(sr, sample_rate)(mix_audio)
    
    # Keep it 2D for the STFT! Shape: [2 (stereo), 132300]
    mix_audio = mix_audio.to(device) 

    # 3. Forward STFT (Get the Complex Numbers)
    n_fft = 1024
    hop_length = 256
    window = torch.hann_window(n_fft).to(device)

    # stft expects a 2D tensor, which mix_audio currently is
    mix_stft_2d = torch.stft(mix_audio, n_fft=n_fft, hop_length=hop_length, 
                             window=window, return_complex=True)
    
    # NOW we add the Batch dimension for the U-Net
    # Shape becomes: [1 (batch), 2 (stereo), 513 (freqs), 517 (frames)]
    mix_stft = mix_stft_2d.unsqueeze(0) 
    
    # Extract magnitude for the AI
    mix_mag = torch.abs(mix_stft)

    # 4. Match the U-Net's strict spatial dimensions (Crop to 512x512)
    original_freqs = mix_mag.shape[-2]
    original_frames = mix_mag.shape[-1]
    
    cropped_mag = mix_mag[:, :, :512, :512]

    # 5. The AI Prediction
    with torch.no_grad(): # No backprop needed here
        masks = model(cropped_mag) # Shape: [1, 4 (stems), 2 (channels), 512, 512]

    # 6. PHASE RECONSTRUCTION
    # Pad the 512x512 masks back to original dimensions (e.g., 513x517)
    pad_freq = original_freqs - 512
    pad_time = original_frames - 512
    
    # F.pad pads from the last dimension backwards: (left, right, top, bottom)
    padded_masks = F.pad(masks, (0, pad_time, 0, pad_freq)) 

    # Multiply the complex original STFT by our real-numbered AI masks
    # We unsqueeze mix_stft so it broadcasts across all 4 stems
    separated_stft = mix_stft.unsqueeze(1) * padded_masks

    # 7. Inverse STFT (Math -> Audio)
    stems = ['vocals', 'drums', 'bass', 'other']
    output_dir = "inference_output"
    os.makedirs(output_dir, exist_ok=True)

    print("\n🎧 Exporting Stems...")
    for i, stem in enumerate(stems):
        # Grab the specific complex STFT for this stem
        stem_stft = separated_stft[0, i] 
        
        # Convert back to waveforms
        stem_audio = torch.istft(stem_stft, n_fft=n_fft, hop_length=hop_length, 
                                 window=window, length=chunk_samples)
        
        # Save to disk
        out_path = os.path.join(output_dir, f"{stem}.wav")
        torchaudio.save(out_path, stem_audio.cpu(), sample_rate)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    infer()

