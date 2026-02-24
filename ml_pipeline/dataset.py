# ml_pipeline/dataset.py
import os
import random
import torch
import torchaudio
import soundfile as sf  # Add this line
from torch.utils.data import Dataset

class MUSDB18Dataset(Dataset):
    def __init__(self, root_dir, chunk_duration=3.0, sample_rate=44100):
        """
        root_dir: The path to your MUSDB18-HQ dataset folders (Train/Test)
        chunk_duration: How many seconds of audio to grab per batch
        """
        self.root_dir = root_dir
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        
        # how many audio samples are in the 3-second chunks
        self.chunk_samples = int(chunk_duration * sample_rate)

        # MUSDB18
        self.song_folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, f))]

        # The exact filenames expect to find in each song folder
        self.stems = ['vocals', 'drums', 'bass', 'other']

    def __len__(self):
        # epoch is 1,000 random 3-second chunks
        return 1000

    def __getitem__(self, idx):
        # 1. Pick a completely random song from the dataset
        song_path = random.choice(self.song_folders)

        # 2. Check the total length of the song so we don't pick a chunk past the end
        mix_path = os.path.join(song_path, "mixture.wav")
        info = sf.info(mix_path)
        total_frames = info.frames

        # 3. Pick a random starting point for our 3-second chunk
        start_frame = random.randint(0, total_frames - self.chunk_samples)

        # 4. Load ONLY the 3-second chunk of the mixed track
        # torchaudio is fast because it skips the rest of the file
        mix_chunk, _ = torchaudio.load(mix_path, frame_offset=start_frame, num_frames=self.chunk_samples)

        # 5. Load the EXACT SAME 3-second chunk for all the isolated stems
        stem_chunks = []
        for stem in self.stems:
            stem_path = os.path.join(song_path, f"{stem}.wav")
            chunk, _ = torchaudio.load(stem_path, frame_offset=start_frame, num_frames=self.chunk_samples)
            stem_chunks.append(chunk)

        # 6. Stack the 4 stems into a single mathematical tensor
        # Shape: [4 (stems), 2 (stereo channels), 132300 (samples)]
        target_stems = torch.stack(stem_chunks)

        return mix_chunk, target_stems

