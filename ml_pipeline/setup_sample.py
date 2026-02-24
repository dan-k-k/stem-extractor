# ml_pipeline/setup_sample.py
import os
import musdb
import soundfile as sf

def setup_sample_song():
    print("Downloading official MUSDB18 7-second sample dataset...")
    mus = musdb.DB(download=True) 

    track = mus[0] 
    print(f"Loaded: {track.name}")

    out_dir = "musdb18hq/SampleSong"
    os.makedirs(out_dir, exist_ok=True)

    print("Converting and extracting stems to .wav...")
    sf.write(os.path.join(out_dir, "mixture.wav"), track.audio, track.rate)

    for name, target in track.targets.items():
        if name in ['vocals', 'drums', 'bass', 'other']:
            sf.write(os.path.join(out_dir, f"{name}.wav"), target.audio, track.rate)

    print(f"Successfully exported '{track.name}' to {out_dir}/")

if __name__ == "__main__":
    setup_sample_song()

