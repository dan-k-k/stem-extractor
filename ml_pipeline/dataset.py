# ml_pipeline/dataset.py
import os
import random
import torch
import torchaudio
import soundfile as sf 
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

        # exact filenames expect to find in each song folder
        self.stems = ['vocals', 'drums', 'bass', 'other']

    def __len__(self):
        # epoch is 1,000 random 3-second chunks
        return 1000

    def __getitem__(self, idx):
        # random song from the dataset
        song_path = random.choice(self.song_folders)

        # total length of the song
        mix_path = os.path.join(song_path, "mixture.wav")
        info = sf.info(mix_path)
        total_frames = info.frames

        start_frame = random.randint(0, total_frames - self.chunk_samples)
        mix_chunk, _ = torchaudio.load(mix_path, frame_offset=start_frame, num_frames=self.chunk_samples)

        # load the EXACT 3-second chunk for all the isolated stems
        stem_chunks = []
        for stem in self.stems:
            stem_path = os.path.join(song_path, f"{stem}.wav")
            chunk, _ = torchaudio.load(stem_path, frame_offset=start_frame, num_frames=self.chunk_samples)
            stem_chunks.append(chunk)

        target_stems = torch.stack(stem_chunks)

        return mix_chunk, target_stems

