# ml_pipeline/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio.transforms as T
from tqdm import tqdm

from dataset import MUSDB18Dataset
from model import StemExtractorUNet

def train():
    # 1. Hardware Acceleration for Apple Silicon (M1/M2/M3)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # 2. Hyperparameters
    batch_size = 4
    epochs = 10
    learning_rate = 1e-4

    # 3. Load Dataset
    # NOTE: You will need to change this path to wherever you download MUSDB18-HQ
    print("Loading Dataset...")
    try:
        dataset = MUSDB18Dataset(root_dir="./musdb18hq", chunk_duration=3.0)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except FileNotFoundError:
        print("Dataset folder not found. Create a dummy folder for testing.")
        return

    # 4. Initialize Brain, Math, and Optimizer
    model = StemExtractorUNet(num_stems=4).to(device)
    criterion = nn.L1Loss() # Standard for comparing spectrograms
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. The Audio-to-Image Converter (STFT)
    # n_fft=1024 gives 513 frequency bins. 
    spectrogram_transform = T.Spectrogram(
        n_fft=1024,
        hop_length=256,
        power=None # Returns complex numbers so we can extract magnitude
    ).to(device)

    print("Starting training loop...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for mix_audio, target_stems_audio in progress_bar:
            mix_audio = mix_audio.to(device)
            target_stems_audio = target_stems_audio.to(device)

            optimizer.zero_grad()

            # --- PREPARE THE IMAGES ---
            # Convert raw audio waves into spectrograms
            mix_stft = spectrogram_transform(mix_audio)
            mix_mag = torch.abs(mix_stft)
            
            target_stft = spectrogram_transform(target_stems_audio)
            target_mag = torch.abs(target_stft)

            # VERY IMPORTANT: The U-Net requires dimensions divisible by 32. 
            # We crop the 513 freq bins to 512, and the 517 time frames to 512.
            mix_mag = mix_mag[:, :, :-1, :512]
            target_mag = target_mag[:, :, :, :-1, :512]

            # --- THE PREDICTION ---
            # The model guesses the filter masks (0.0 to 1.0)
            predicted_masks = model(mix_mag)

            # Apply the masks to the original mix
            # We unsqueeze the mix to broadcast it across all 4 stems
            predicted_stems_mag = predicted_masks * mix_mag.unsqueeze(1)

            # --- THE CORRECTION ---
            # Compare the prediction to the real isolated stems
            loss = criterion(predicted_stems_mag, target_mag)
            
            # Update the brain
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete - Average Loss: {avg_loss:.4f}")

        # Save the brain's new weights
        torch.save(model.state_dict(), f"unet_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()

