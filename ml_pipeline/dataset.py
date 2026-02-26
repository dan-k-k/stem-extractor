# ml_pipeline/dataset.py
import os
import random
import torch
import torchaudio
import soundfile as sf 
from torch.utils.data import Dataset

class MUSDB18Dataset(Dataset):
    def __init__(self, root_dir, split="train", chunk_duration=3.0, sample_rate=44100):

        self.split_dir = os.path.join(root_dir, split)
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        
        self.chunk_samples = int(chunk_duration * sample_rate)

        # Train or test
        self.song_folders = [os.path.join(self.split_dir, f) for f in os.listdir(self.split_dir) 
                             if os.path.isdir(os.path.join(self.split_dir, f))]

        self.stems = ['vocals', 'drums', 'bass', 'other']

    def __len__(self):
        # Epoch is 1,000 random 3-second chunks
        return 5000

    def __getitem__(self, idx):
        # Random song from the dataset
        song_path = random.choice(self.song_folders)

        mix_path = os.path.join(song_path, "mixture.wav")
        info = sf.info(mix_path)
        total_frames = info.frames

        start_frame = random.randint(0, total_frames - self.chunk_samples)
        mix_chunk, _ = torchaudio.load(mix_path, frame_offset=start_frame, num_frames=self.chunk_samples)

        stem_chunks = []
        for stem in self.stems:
            stem_path = os.path.join(song_path, f"{stem}.wav")
            chunk, _ = torchaudio.load(stem_path, frame_offset=start_frame, num_frames=self.chunk_samples)
            stem_chunks.append(chunk)

        target_stems = torch.stack(stem_chunks)

        return mix_chunk, target_stems

