#!/usr/bin/env python3
"""
Script to compare trained VballNetV3c models
"""

import torch
import json
from pathlib import Path
import numpy as np

def load_training_history(history_file):
    """Load training history from a JSON file"""
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except:
        return None

def compare_model_results():
    """Compare results from different VballNetV3c model variants"""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("Outputs directory not found!")
        return
    
    # Find all VballNetV3c experiment directories
    v3c_experiments = []
    for exp_dir in outputs_dir.iterdir():
        if exp_dir.is_dir() and "VballNetV3c" in str(exp_dir):
            v3c_experiments.append(exp_dir)
    
    if not v3c_experiments:
        print("No VballNetV3c experiments found in outputs directory!")
        return
    
    print("Found VballNetV3c experiments:")
    for exp in v3c_experiments:
        print(f"  - {exp.name}")
    print()
    
    # Compare results
    results = []
    for exp_dir in v3c_experiments:
        config_file = exp_dir / "config.json"
        if not config_file.exists():
            continue
            
        # Load config to get model variant
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            variant = config.get("model_variant", "unknown")
        except:
            variant = "unknown"
        
        # Find best model checkpoint
        checkpoints_dir = exp_dir / "checkpoints"
        if not checkpoints_dir.exists():
            continue
            
        best_model = None
        for ckpt in checkpoints_dir.iterdir():
            if "best" in str(ckpt) and ckpt.suffix == ".pth":
                best_model = ckpt
                break
        
        if best_model is None:
            continue
        
        # Load checkpoint and extract metrics
        try:
            checkpoint = torch.load(best_model, map_location="cpu")
            train_loss = checkpoint.get("train_loss", float('inf'))
            val_loss = checkpoint.get("val_loss", float('inf'))
            epoch = checkpoint.get("epoch", -1)
            
            # Load training history if available
            history_file = exp_dir / "config.json"
            history = load_training_history(history_file)
            
            results.append({
                "experiment": exp_dir.name,
                "variant": variant,
                "best_epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "checkpoint": best_model
            })
        except Exception as e:
            print(f"Error loading checkpoint from {exp_dir.name}: {e}")
    
    # Sort results by validation loss
    results.sort(key=lambda x: x["val_loss"])
    
    # Display comparison
    print("VballNetV3c Model Comparison (sorted by validation loss):")
    print("=" * 80)
    print(f"{'Variant':<15} {'Val Loss':<12} {'Train Loss':<12} {'Best Epoch':<12} {'Experiment'}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['variant']:<15} {result['val_loss']:<12.6f} {result['train_loss']:<12.6f} {result['best_epoch']:<12} {result['experiment']}")
    
    print("\nBest performing model:")
    if results:
        best = results[0]
        print(f"  Variant: {best['variant']}")
        print(f"  Validation Loss: {best['val_loss']:.6f}")
        print(f"  Training Loss: {best['train_loss']:.6f}")
        print(f"  Checkpoint: {best['checkpoint']}")

if __name__ == "__main__":
    compare_model_results()