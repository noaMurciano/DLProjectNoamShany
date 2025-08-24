"""
Personal Food Taste Classification - Training Module

Two-phase curriculum learning for personal food taste preferences.

Author: Noam & Shany
Course: 046211 Deep Learning, Technion

Attribution:
- Curriculum learning: Bengio, Y., et al. (2009). Curriculum learning. ICML.
- Transfer learning: Yosinski, J., et al. (2014). How transferable are features in deep neural networks? NIPS.
- PyTorch training: https://pytorch.org/tutorials/beginner/training_classifier.html
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from .model import create_model, TasteClassificationLoss, get_parameter_groups
from .dataset import create_personal_dataloaders


class PersonalFineTuner3Way:
    """
    Fine-tuner for personal food taste classification with proper train/val/test splits.
    
    Implements two-phase curriculum learning:
    - Phase 1: Frozen backbone, train classification head only
    - Phase 2: End-to-end fine-tuning with very low backbone learning rate
    
    Attribution:
    - Two-phase training: Standard transfer learning practice
    - Early stopping: Prechelt, L. (1998). Early stopping—but when? Neural Networks.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize fine-tuner with configuration.
        
        Args:
            config: Dictionary containing all training parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create experiment directory
        self.exp_dir = Path(config['exp_dir'])
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize model, loss, optimizer
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None  
        self.test_loader = None
        
        # Training state
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.training_history = []
        
        print(f"🎯 PersonalFineTuner3Way initialized")
        print(f"📂 Experiment directory: {self.exp_dir}")
        print(f"🔧 Device: {self.device}")
    
    def load_pretrained_model(self):
        """Load pre-trained Food101 model as starting point."""
        pretrained_path = self.config.get('pretrained_model_path')
        
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"📥 Loading pre-trained model from {pretrained_path}")
            
            # Load checkpoint
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            
            # Get model config from checkpoint or use defaults
            model_config = checkpoint.get('config', {}).get('model', {
                'num_classes': 3,
                'backbone': 'tf_efficientnet_b3',
                'pretrained': True,
                'dropout': 0.3,
                'hidden_dim': 256
            })
            
            # Create model
            self.model = create_model(model_config, self.device)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Pre-trained model loaded successfully")
            
        else:
            print(f"⚠️ No pre-trained model found, creating new model")
            model_config = {
                'num_classes': 3,
                'backbone': 'tf_efficientnet_b3',
                'pretrained': True,
                'dropout': 0.3,
                'hidden_dim': 256
            }
            self.model = create_model(model_config, self.device)
    
    def setup_fine_tuning(self):
        """Setup loss function, optimizer, and scheduler for fine-tuning."""
        
        # Calculate class weights for imbalance handling
        # Load training data to calculate weights
        train_df = pd.read_csv(self.config['train_csv'])
        class_counts = train_df['class_id'].value_counts().sort_index()
        total = len(train_df)
        
        class_weights = []
        for class_id in [0, 1, 2]:
            count = class_counts.get(class_id, 1)
            weight = total / (3 * count)
            class_weights.append(weight)
        
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        print(f"📊 Class weights: {class_weights.cpu().numpy()}")
        
        # Loss function with class weights
        self.loss_fn = TasteClassificationLoss(
            class_weights=class_weights,
            label_smoothing=0.05
        )
        
        # Discriminative learning rates
        param_groups = get_parameter_groups(
            self.model,
            head_lr=self.config.get('head_lr', 1e-3),
            backbone_lr=self.config.get('backbone_lr', 1e-5),
            weight_decay=0.01
        )
        
        # AdamW optimizer
        self.optimizer = optim.AdamW(param_groups)
        
        # Cosine annealing scheduler
        total_epochs = self.config.get('total_epochs', 15)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=1e-7
        )
        
        print(f"🎯 Fine-tuning setup complete")
    
    def create_dataloaders(self):
        """Create train, validation, and test dataloaders."""
        
        self.train_loader, self.val_loader, self.test_loader = create_personal_dataloaders(
            train_csv=self.config['train_csv'],
            val_csv=self.config['val_csv'],
            test_csv=self.config['test_csv'],
            batch_size=self.config.get('batch_size', 16),
            num_workers=self.config.get('num_workers', 2),
            data_dir=self.config.get('data_dir', 'data/food101_cache')
        )
        
        print(f"📊 DataLoaders created:")
        print(f"   Train: {len(self.train_loader)} batches")
        print(f"   Val: {len(self.val_loader)} batches")
        print(f"   Test: {len(self.test_loader)} batches")
    
    def train_epoch(self, phase: str) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (images, targets, metadata, image_ids) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            
            # Calculate loss
            loss, loss_dict = self.loss_fn(logits, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Progress logging
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(self.train_loader)}: loss={loss.item():.4f}")
        
        # Calculate epoch metrics
        epoch_loss /= len(self.train_loader)
        epoch_acc = accuracy_score(all_targets, all_predictions)
        epoch_f1 = f1_score(all_targets, all_predictions, average='macro')
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'macro_f1': epoch_f1,
            'phase': phase
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, targets, metadata, image_ids in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                loss, loss_dict = self.loss_fn(logits, targets)
                
                # Track metrics
                epoch_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_loss /= len(self.val_loader)
        epoch_acc = accuracy_score(all_targets, all_predictions)
        epoch_f1 = f1_score(all_targets, all_predictions, average='macro')
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'macro_f1': epoch_f1,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probs
        }
    
    def train(self) -> Tuple[float, float]:
        """
        Main training loop with two-phase curriculum learning.
        
        Returns:
            final_test_acc: Final test accuracy
            final_test_f1: Final test F1-score
        """
        print(f"\n🚀 Starting fine-tuning...")
        
        # Phase 1: Frozen backbone (epochs 1-8)
        print(f"\n📍 PHASE 1: Frozen Backbone Training")
        self.model.freeze_backbone(freeze=True)
        
        phase1_epochs = self.config.get('phase1_epochs', 8)
        
        for epoch in range(1, phase1_epochs + 1):
            print(f"\n--- Epoch {epoch}/{phase1_epochs} (Phase 1) ---")
            
            # Train
            train_metrics = self.train_epoch('phase1')
            print(f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.3f}, f1={train_metrics['macro_f1']:.4f}")
            
            # Validate
            val_metrics = self.validate_epoch()
            print(f"Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.3f}, f1={val_metrics['macro_f1']:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save metrics
            self.training_history.append({
                'epoch': epoch,
                'phase': 'phase1',
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'train_f1': train_metrics['macro_f1'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['macro_f1']
            })
            
            # Early stopping check
            if val_metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['macro_f1']
                self.patience_counter = 0
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
                print(f"✅ New best model saved (F1: {self.best_val_f1:.4f})")
            else:
                self.patience_counter += 1
        
        # Phase 2: End-to-end fine-tuning (epochs 9-15)
        print(f"\n📍 PHASE 2: End-to-End Fine-tuning")
        self.model.freeze_backbone(freeze=False)
        
        phase2_epochs = self.config.get('phase2_epochs', 7)
        total_epochs = phase1_epochs + phase2_epochs
        
        for epoch in range(phase1_epochs + 1, total_epochs + 1):
            print(f"\n--- Epoch {epoch}/{total_epochs} (Phase 2) ---")
            
            # Train
            train_metrics = self.train_epoch('phase2')
            print(f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.3f}, f1={train_metrics['macro_f1']:.4f}")
            
            # Validate
            val_metrics = self.validate_epoch()
            print(f"Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.3f}, f1={val_metrics['macro_f1']:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save metrics
            self.training_history.append({
                'epoch': epoch,
                'phase': 'phase2',
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'train_f1': train_metrics['macro_f1'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['macro_f1']
            })
            
            # Early stopping check
            if val_metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['macro_f1']
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"✅ New best model saved (F1: {self.best_val_f1:.4f})")
            else:
                self.patience_counter += 1
                
            # Early stopping
            patience = self.config.get('patience', 5)
            if self.patience_counter >= patience:
                print(f"🛑 Early stopping triggered (patience: {patience})")
                break
        
        # Final evaluation on test set
        print(f"\n🧪 Final Test Evaluation")
        final_test_acc, final_test_f1 = self.final_test_evaluation()
        
        # Save training history
        self.save_training_history()
        
        print(f"\n🎉 Fine-tuning completed!")
        print(f"📊 Best validation F1: {self.best_val_f1:.4f}")
        print(f"🧪 Final test accuracy: {final_test_acc:.2f}%")
        print(f"🧪 Final test F1-score: {final_test_f1:.4f}")
        
        return final_test_acc, final_test_f1
    
    def final_test_evaluation(self) -> Tuple[float, float]:
        """Evaluate the best model on test set."""
        
        # Load best model
        best_model_path = self.exp_dir / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📥 Loaded best model for test evaluation")
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, targets, metadata, image_ids in self.test_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate final metrics
        test_acc = accuracy_score(all_targets, all_predictions) * 100
        test_f1 = f1_score(all_targets, all_predictions, average='macro')
        
        # Detailed analysis
        self.save_test_analysis(all_targets, all_predictions, all_probs)
        
        return test_acc, test_f1
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'config': self.config
        }
        
        if is_best:
            torch.save(checkpoint, self.exp_dir / 'best_model.pt')
        
        torch.save(checkpoint, self.exp_dir / f'checkpoint_epoch_{epoch}.pt')
    
    def save_training_history(self):
        """Save training history to CSV."""
        df = pd.DataFrame(self.training_history)
        df.to_csv(self.exp_dir / 'training_log.csv', index=False)
        print(f"📊 Training history saved to {self.exp_dir / 'training_log.csv'}")
    
    def save_test_analysis(self, targets, predictions, probabilities):
        """Save detailed test set analysis."""
        
        # Classification report
        class_names = ['disgusting', 'neutral', 'tasty']
        report = classification_report(targets, predictions, target_names=class_names, output_dict=True)
        
        with open(self.exp_dir / 'test_classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Test Set Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'test_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Test analysis saved to {self.exp_dir}")


if __name__ == "__main__":
    # Example usage
    print("PersonalFineTuner3Way - Training module for personal food taste classification")
    print("This module implements two-phase curriculum learning for transfer learning.")
