# ml_pipeline/export_onnx.py
import os
import torch
from model import StemExtractorUNet
import onnx

def export_to_onnx():
    print("Initializing PyTorch Model...")
    model = StemExtractorUNet(num_stems=4)
    
    weights_path = "unet_best.pt" 
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ CRITICAL ERROR: Trained weights '{weights_path}' not found! Run train.py first.")

    print(f"Loading trained weights from {weights_path}...")
    
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    
    # --- UPDATED: Match the C++ aiTimeFrames exactly (512) ---
    # Shape: (Batch=1, Channels=2, FreqBins=512, TimeFrames=512)
    dummy_input = torch.randn(1, 2, 512, 512)

    onnx_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plugin", "stem_extractor.onnx"))
    print(f"Exporting to {onnx_filename}...")
    
    # --- UPDATED: Removed dynamic_shapes entirely for a static, optimized graph ---
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

    print(f"\nConsolidating model into a single file...")
    
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
        print("🗑️ Cleaned up orphaned external data file.")
        
    print(f"✅ Export successful! The completely self-contained model is at: '{onnx_filename}'")

if __name__ == "__main__":
    export_to_onnx()

