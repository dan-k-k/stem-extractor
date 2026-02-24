# ml_pipeline/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio.transforms as T
from tqdm import tqdm

from dataset import MUSDB18Dataset
from model import StemExtractorUNet

def process_batch(mix_audio, target_stems_audio, model, criterion, spectrogram_transform, device):
    """Helper function to process a single batch and return the loss."""
    mix_audio = mix_audio.to(device)
    target_stems_audio = target_stems_audio.to(device)

    mix_stft = spectrogram_transform(mix_audio)
    mix_mag = torch.abs(mix_stft)
    
    target_stft = spectrogram_transform(target_stems_audio)
    target_mag = torch.abs(target_stft)

    # Crop the 513 freq bins to 512, and the 517 time frames to 512.
    mix_mag = mix_mag[:, :, :-1, :512]
    target_mag = target_mag[:, :, :, :-1, :512]

    # Predict filter masks and apply to mix
    predicted_masks = model(mix_mag)
    predicted_stems_mag = predicted_masks * mix_mag.unsqueeze(1)

    loss = criterion(predicted_stems_mag, target_mag)
    return loss

def train():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🎸 Training on device: {device}")

    batch_size = 4
    epochs = 50  # Increased for full dataset
    learning_rate = 1e-4
    data_dir = "./musdb18hq"

    print("Loading Datasets...")
    try:
        train_dataset = MUSDB18Dataset(root_dir=data_dir, split="train", chunk_duration=3.0)
        val_dataset = MUSDB18Dataset(root_dir=data_dir, split="test", chunk_duration=3.0)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    except FileNotFoundError:
        print(f"Dataset folder '{data_dir}' not found. Please run setup_full_dataset.py first.")
        return

    model = StemExtractorUNet(num_stems=4).to(device)
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    spectrogram_transform = T.Spectrogram(
        n_fft=1024,
        hop_length=256,
        power=None 
    ).to(device)

    # Setup for Early Stopping & Checkpointing
    patience = 10
    start_epoch = 0
    epochs_no_improve = 0
    best_val_loss = float('inf')
    checkpoint_path = "unet_best.pt"

    if os.path.exists(checkpoint_path):
        print(f"🔄 Found checkpoint '{checkpoint_path}'. Loading state...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        epochs_no_improve = checkpoint['epochs_no_improve']
        
        print(f"✅ Resuming from Epoch {start_epoch} (Best Val Loss: {best_val_loss:.4f})")
    else:
        print("✨ No checkpoint found. Starting from scratch.")

    print("Starting training loop...")
    for epoch in range(start_epoch, epochs):
        # --- TRAINING PASS ---
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for mix_audio, target_stems_audio in train_bar:
            optimizer.zero_grad()
            loss = process_batch(mix_audio, target_stems_audio, model, criterion, spectrogram_transform, device)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PASS ---
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")

        with torch.no_grad():
            for mix_audio, target_stems_audio in val_bar:
                loss = process_batch(mix_audio, target_stems_audio, model, criterion, spectrogram_transform, device)
                val_loss += loss.item()
                val_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- EARLY STOPPING & CHECKPOINTING ---
        if avg_val_loss < best_val_loss:
            print(f"⭐ Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving checkpoint!")
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # Reset patience counter
            
            # Save the full state dictionary
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")
            
            if epochs_no_improve >= patience:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs! The best model is saved as '{checkpoint_path}'.")
                break # Exit the training loop

if __name__ == "__main__":
    train()

