import torch
import argparse
import os
from model.vballnet_v3b import VballNetV3b as VballNetV3


def export_model_to_onnx(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Check if this is a full checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✅ Loaded weights from 'model_state_dict' checkpoint")
    else:
        state_dict = checkpoint
        print("✅ Direct state_dict loading")

    # Initialize model
    height, width, in_dim, out_dim = 288, 512, 9, 9
    model = VballNetV3(height=height, width=width, in_dim=in_dim, out_dim=out_dim)
    model.to(device)
    model.eval()

    # Handle state dict loading with strict=False to skip mismatched keys
    try:
        model.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded successfully (non-strict mode)")
    except Exception as e:
        print(f"⚠️  Warning: Could not load all weights: {e}")
        # Try to load with more permissive approach
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(filtered_state_dict, strict=False)
        print("✅ Filtered weights loaded")

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
    parser = argparse.ArgumentParser(description="Export VballNetV3 model to ONNX")
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth model file')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    export_model_to_onnx(args.model_path)