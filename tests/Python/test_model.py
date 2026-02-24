# tests/Python/test_model.py
import sys
import os
import torch
import pytest

# Add the ml_pipeline directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ml_pipeline')))
from model import StemExtractorUNet

def test_model_output_shape():
    """
    Ensures the U-Net takes in a standard stereo 512x512 spectrogram 
    and successfully outputs 4 stems with the exact same spatial dimensions.
    """
    model = StemExtractorUNet(num_stems=4)
    
    # Create a dummy batch of audio spectrograms: 
    # [1 (batch), 2 (stereo channels), 512 (freq bins), 512 (time frames)]
    dummy_input = torch.randn(1, 2, 512, 512)
    output = model(dummy_input)
    
    # Expected shape: [1 (batch), 4 (stems), 2 (channels), 512, 512]
    expected_shape = (1, 4, 2, 512, 512)
    
    assert output.shape == expected_shape, f"Failed! Expected {expected_shape}, got {output.shape}"

def test_model_mask_values():
    """
    The final layer of the model uses a Sigmoid activation. 
    This test proves that the network correctly outputs AI mask values strictly between 0.0 and 1.0.
    """
    model = StemExtractorUNet(num_stems=4)
    dummy_input = torch.randn(1, 2, 512, 512)
    output = model(dummy_input)
    
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), "Mask values must be strictly bounded between 0 and 1."

