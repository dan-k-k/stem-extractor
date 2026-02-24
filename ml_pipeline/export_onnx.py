# ml_pipeline/export_onnx.py
import os
import torch
from model import StemExtractorUNet

def export_to_onnx():
    print("Initializing PyTorch Model...")
    model = StemExtractorUNet(num_stems=4)
    weights_path = "unet_epoch_10.pt" 
    
    if os.path.exists(weights_path):
        print(f"Loading trained weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    else:
        print(f"Warning: '{weights_path}' not found. Exporting UNTRAINED model.")

    model.eval()
    dummy_input = torch.randn(1, 2, 512, 128)
    
    dynamic_axes = {
        'input_spectrogram': {3: 'time_frames'},
        'output_masks': {4: 'time_frames'}
    }

    shared_dir = "/Users/Shared/StemExtractor"
    os.makedirs(shared_dir, exist_ok=True) 
    onnx_filename = os.path.join(shared_dir, "stem_extractor.onnx")
    print(f"Exporting to {onnx_filename}...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_filename,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_spectrogram'],
        output_names=['output_masks'],
        dynamic_axes=dynamic_axes
    )

    print(f"\nExport successful! The model is securely waiting at: '{onnx_filename}'")

if __name__ == "__main__":
    export_to_onnx()

