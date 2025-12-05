#!/usr/bin/env python3
"""
Quick test script for VballNetV3c models
"""

import torch
from model.vballnet_v3c import VballNetV3b as VballNetV3cOriginal
from model.vballnet_v3c_minimal import VballNetV3cMinimal
from model.vballnet_v3c_improved import VballNetV3cImproved

def quick_test():
    """Run a quick test on all VballNetV3c variants"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample input (9 frames of 288x512)
    batch_size = 2
    sample_input = torch.randn(batch_size, 9, 288, 512, device=device)
    print(f"Sample input shape: {sample_input.shape}")
    
    # Test all model variants
    models = {
        'Original': VballNetV3cOriginal(height=288, width=512, in_dim=9, out_dim=9).to(device),
        'Minimal': VballNetV3cMinimal(height=288, width=512, in_dim=9, out_dim=9).to(device),
        'Improved': VballNetV3cImproved(height=288, width=512, in_dim=9, out_dim=9).to(device),
    }
    
    print("\nTesting all model variants...")
    for name, model in models.items():
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        
        with torch.no_grad():
            output = model(sample_input)
            print(f"{name:12} | Params: {params:8,} | Output: {tuple(output.shape)} | Range: [{output.min():.4f}, {output.max():.4f}]")
    
    print("\nQuick test completed successfully!")

if __name__ == "__main__":
    quick_test()