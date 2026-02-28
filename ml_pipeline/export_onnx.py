# ml_pipeline/export_onnx.py
import os
import torch
from model import StemExtractorUNet
import onnx

def export_to_onnx():
    print("Initialising PyTorch model...")
    model = StemExtractorUNet(num_stems=4)
    
    weights_path = "unet_best.pt" 
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Error: trained weights '{weights_path}' not found. Run train.py first.")

    print(f"Loading trained weights from {weights_path}...")
    
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    
    dummy_input = torch.randn(1, 2, 512, 512)

    onnx_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plugin", "stem_extractor.onnx"))
    print(f"Exporting to {onnx_filename}...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_filename,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_spectrogram'],
        output_names=['output_masks']
    )
    
    onnx_model = onnx.load(onnx_filename)
    
    onnx.save_model(
        onnx_model, 
        onnx_filename, 
        save_as_external_data=False, 
        all_tensors_to_one_file=True
    )
    
    data_filename = onnx_filename + ".data"
    if os.path.exists(data_filename):
        os.remove(data_filename)
        
    print(f"Exported to: '{onnx_filename}'")

if __name__ == "__main__":
    export_to_onnx()

