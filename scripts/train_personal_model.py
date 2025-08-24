#!/usr/bin/env python3
"""
Personal Food Taste Classification - Training Script

Main training script for fine-tuning personal food taste preferences.
Supports both 80/20 and 60/20/20 data splitting strategies.

Author: Noam Shani
Course: 046211 Deep Learning
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Import project modules
from src.model import create_model
from src.dataset import PersonalTasteDataset, create_personal_dataloaders, create_3way_dataloaders
from src.train import PersonalFineTuner3Way

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Personal Food Taste Classification Model')
    
    # Data arguments
    parser.add_argument('--data', type=str, default='data/personal_labels.xlsx',
                       help='Path to personal labels file')
    parser.add_argument('--split', choices=['2way', '3way'], default='3way',
                       help='Data split strategy: 2way (80/20) or 3way (60/20/20)')
    
    # Model arguments
    parser.add_argument('--config', type=str, default='configs/training_config.json',
                       help='Path to training configuration file')
    parser.add_argument('--pretrained', type=str,
                       default='experiments/base_model/best_model.pt',
                       help='Path to pre-trained model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=15,
                       help='Total number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--lr-head', type=float, default=1e-3,
                       help='Learning rate for classification head')
    parser.add_argument('--lr-backbone', type=float, default=1e-5,
                       help='Learning rate for backbone')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='experiments/',
                       help='Output directory for experiments')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (auto-generated if None)')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    return parser.parse_args()

def load_config(config_path):
    """Load training configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠️ Config file not found: {config_path}")
        print("Using default configuration...")
        return create_default_config()

def create_default_config():
    """Create default training configuration."""
    return {
        'model': {
            'num_classes': 3,
            'backbone': 'tf_efficientnet_b3',
            'pretrained': True,
            'dropout': 0.3,
            'hidden_dim': 256
        },
        'training': {
            'phase1_epochs': 8,
            'phase2_epochs': 7,
            'freeze_backbone_initially': True,
            'use_class_weights': True,
            'patience': 5
        },
        'optimization': {
            'weight_decay': 0.01,
            'label_smoothing': 0.05
        }
    }

def prepare_data_splits(data_path, split_strategy, output_dir):
    """Prepare train/val/test splits from personal labels."""
    print(f"📊 Preparing data splits ({split_strategy})...")
    
    # Load personal labels
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} personal samples")
    
    # Map labels to class IDs
    label_mapping = {'disgusting': 0, 'neutral': 1, 'tasty': 2}
    df['class_id'] = df['label'].map(label_mapping)
    
    # Remove unmapped labels
    df = df.dropna(subset=['class_id']).reset_index(drop=True)
    df['class_id'] = df['class_id'].astype(int)
    
    print(f"Class distribution: {df['class_id'].value_counts().sort_index().to_dict()}")
    
    # Create splits
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if split_strategy == '2way':
        # 80/20 split
        train_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df['class_id'], random_state=42
        )
        
        train_path = output_dir / 'personal_train_2way.csv'
        test_path = output_dir / 'personal_test_2way.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"📁 2-way split created:")
        print(f"   Train: {len(train_df)} samples → {train_path}")
        print(f"   Test: {len(test_df)} samples → {test_path}")
        
        return str(train_path), None, str(test_path)
        
    else:  # 3way
        # 60/20/20 split
        train_df, temp_df = train_test_split(
            df, test_size=0.4, stratify=df['class_id'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['class_id'], random_state=42
        )
        
        train_path = output_dir / 'personal_train_3way.csv'
        val_path = output_dir / 'personal_val_3way.csv'
        test_path = output_dir / 'personal_test_3way.csv'
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"📁 3-way split created:")
        print(f"   Train: {len(train_df)} samples → {train_path}")
        print(f"   Val: {len(val_df)} samples → {val_path}")
        print(f"   Test: {len(test_df)} samples → {test_path}")
        
        return str(train_path), str(val_path), str(test_path)

def create_experiment_config(args, config, train_path, val_path, test_path):
    """Create experiment-specific configuration."""
    
    # Generate experiment name if not provided
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"personal_{args.split}_{timestamp}"
    
    # Override config with command line arguments
    experiment_config = config.copy()
    
    # Update paths
    experiment_config['train_csv'] = train_path
    experiment_config['val_csv'] = val_path
    experiment_config['test_csv'] = test_path
    experiment_config['pretrained_model_path'] = args.pretrained
    experiment_config['data_dir'] = 'data/food101_cache/'
    
    # Update training parameters
    experiment_config['total_epochs'] = args.epochs
    experiment_config['batch_size'] = args.batch_size
    experiment_config['head_lr'] = args.lr_head
    experiment_config['backbone_lr'] = args.lr_backbone
    experiment_config['num_workers'] = args.num_workers
    
    # Update experiment info
    experiment_config['exp_dir'] = f"{args.output_dir}/{args.name}"
    experiment_config['split_strategy'] = args.split
    
    return experiment_config

def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    print("🍽️ Personal Food Taste Classification - Training")
    print("=" * 60)
    print(f"Split Strategy: {args.split}")
    print(f"Data: {args.data}")
    print(f"Pre-trained Model: {args.pretrained}")
    print(f"Output Directory: {args.output_dir}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Prepare data splits
    train_path, val_path, test_path = prepare_data_splits(
        args.data, args.split, 'data/'
    )
    
    # Create experiment configuration
    experiment_config = create_experiment_config(
        args, config, train_path, val_path, test_path
    )
    
    # Verify required files exist
    required_files = [train_path, test_path, args.pretrained]
    if val_path:
        required_files.append(val_path)
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            return
    
    print(f"✅ All required files verified")
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Using device: {device}")
    
    try:
        if args.split == '3way':
            # Train with validation set
            print(f"🚀 Starting 3-way training with validation...")
            trainer = PersonalFineTuner3Way(experiment_config)
            
            # Load pre-trained model
            trainer.load_pretrained_model()
            
            # Setup fine-tuning
            trainer.setup_fine_tuning()
            
            # Create data loaders
            trainer.create_dataloaders()
            
            # Train
            final_test_acc, final_test_f1 = trainer.train()
            
            print(f"\n🎉 Training completed!")
            print(f"📊 Final test accuracy: {final_test_acc:.2f}%")
            print(f"📈 Final test F1-score: {final_test_f1:.4f}")
            print(f"📂 Results saved in: {trainer.exp_dir}")
            
        else:
            # 2-way training (simplified version)
            print(f"🚀 Starting 2-way training...")
            print(f"⚠️ Note: 2-way training uses test set for validation (not recommended)")
            
            # For 2-way, we'd need to implement a simplified trainer
            # This is left as an exercise or can use the existing approach
            print(f"❌ 2-way training not fully implemented in this script")
            print(f"💡 Recommendation: Use --split 3way for proper validation")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
