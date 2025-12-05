import torch
import argparse
import os

# Since the checkpoint was saved from a different architecture,
# we need to recreate that architecture to load the weights properly
from model.vballnet_v3 import VballNetV3 as ModelClass  # This matches the checkpoint structure


def export_model_to_onnx(model_path):
    device = torch.device("cpu")  # Using CPU for export to ensure compatibility

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Check if this is a full checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✅ Loaded weights from 'model_state_dict' checkpoint")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("✅ Loaded weights from 'state_dict' checkpoint")
    else:
        state_dict = checkpoint
        print("✅ Direct state_dict loading")

    # Initialize model
    height, width, in_dim, out_dim = 288, 512, 9, 9
    model = ModelClass(height=height, width=width, in_dim=in_dim, out_dim=out_dim)
    model.to(device)
    model.eval()

    # Handle state dict loading with filtering for 'module.' prefix
    filtered_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix if present (from DataParallel)
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        filtered_state_dict[new_key] = value

    # Load the filtered state dict
    try:
        model.load_state_dict(filtered_state_dict, strict=False)
        print("✅ Weights loaded successfully (non-strict mode)")
    except Exception as e:
        print(f"⚠️  Warning: Could not load all weights: {e}")

    # Create dummy input
    dummy_input = torch.randn(1, in_dim, height, width, device=device)
    onnx_path = model_path.replace(".pth", ".onnx")

    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )

    print(f"✅ Model exported to ONNX: {onnx_path}")

    # ONNX validation (optional)
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model is valid")
    except ImportError:
        print("ℹ️  Install onnx: pip install onnx")
    except Exception as e:
        print(f"❌ ONNX validation error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export VballNetV3 model to ONNX (matches VballNetV3c training checkpoint)")
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth model file')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    export_model_to_onnx(args.model_path)