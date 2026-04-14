## Stem Extractor VST3

A real-time, AI-powered audio source separation plugin built with PyTorch, ONNX Runtime, and JUCE. This plugin dynamically isolates Vocals, Drums, Bass, and Other instruments from a full mix directly inside your DAW. **Currently installs only on macOS**.

__Plugin Stem Outputs Demo__

<p align="center">
  <img src="images/PluginUI.png" alt="Stem Extractor UI in Ableton" width="500">
</p>

<p align="center">
  <a href="https://youtu.be/xYLYWHUjavo">
    <img src="images/DemoThumbnail1.png" alt="Stem Extractor Demo" width="500">
  </a>
</p>

__Spectrogram Comparison__ (librosa)

<p align="center">
  <img src="images/spectrogram_comparison1.png" alt="Spectrogram Comparison" width="800">
</p>

The plugin applies STFT to mask for stems.

#### Train loss

<p align="center">
  <img src="images/loss_curve.png" alt="Training vs Validation Loss" width="400">
</p>

#### SI-SDR across Full MUSDB18-HQ Test Set

<p align="center">
    <img src="images/si_sdr_boxplot1.png" alt="SI-SDR Score Distribution" width="400">
</p>

The baseline U-Net achieves a median SI-SDR of 3.5 dB for Vocals and 2.9 dB for Drums. Output stems corresponding to true stems with no signal (< -60dBFS) are not considered, but there are still extreme negative outliers where the model mistakenly leaks audio into the prediction when there is very subtle true-stem sound.

##### AI Inference 

<p align="center">
    <img src="images/AI_inference.png" alt="AI inference timing" width="800">
</p>

