# tests/Python/test_model.py
import sys
import os
import torch
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ml_pipeline')))
from model import StemExtractorUNet

def test_model_output_shape():
    model = StemExtractorUNet(num_stems=4)
    
    dummy_input = torch.randn(1, 2, 512, 512)
    output = model(dummy_input)
    
    expected_shape = (1, 4, 2, 512, 512)
    
    assert output.shape == expected_shape, f"Failed! Expected {expected_shape}, got {output.shape}"

def test_model_mask_values():
    model = StemExtractorUNet(num_stems=4)
    dummy_input = torch.randn(1, 2, 512, 512)
    output = model(dummy_input)
    
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), "Mask values must be strictly bounded between 0 and 1."

