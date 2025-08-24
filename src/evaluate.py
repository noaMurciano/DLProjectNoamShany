"""
Personal Food Taste Classification - Evaluation Module

Comprehensive model evaluation and analysis tools.

Author: Noam Shani
Course: 046211 Deep Learning, Technion

Attribution:
- Evaluation metrics: scikit-learn library
- Confusion matrix: Standard classification evaluation
- ROC curves: Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, 
    roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .model import create_model
from .dataset import PersonalTasteDataset, get_personal_transforms


def load_trained_model(model_path: str, device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Target device for model
        
    Returns:
        Loaded model in evaluation mode
        
    Attribution:
    - PyTorch model loading: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    model_config = checkpoint.get('config', {}).get('model', {
        'num_classes': 3,
        'backbone': 'tf_efficientnet_b3',
        'pretrained': True,
        'dropout': 0.3,
        'hidden_dim': 256
    })
    
    # Create and load model
    model = create_model(model_config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    return model


def evaluate_model(
    model: torch.nn.Module,
    test_csv: str,
    data_dir: str = 'data/food101_cache',
    batch_size: int = 16,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Comprehensive model evaluation on test set.
    
    Args:
        model: Trained model for evaluation
        test_csv: Path to test set CSV
        data_dir: Data directory path
        batch_size: Batch size for evaluation
        device: Device for computation
        
    Returns:
        Dictionary containing all evaluation metrics
        
    Attribution:
    - Evaluation framework: Standard machine learning evaluation practices
    - Metrics: scikit-learn implementations
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Create test dataset and loader
    test_dataset = PersonalTasteDataset(
        csv_path=test_csv,
        split='test',
        data_dir=data_dir,
        transform=get_personal_transforms('test')
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Collect predictions
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_image_ids = []
    
    print(f"🧪 Evaluating on {len(test_dataset)} test samples...")
    
    with torch.no_grad():
        for images, targets, metadata, image_ids in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_image_ids.extend(image_ids)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate comprehensive metrics
    results = calculate_comprehensive_metrics(
        all_targets, all_predictions, all_probabilities
    )
    
    # Add prediction details
    results['prediction_details'] = {
        'predictions': all_predictions.tolist(),
        'targets': all_targets.tolist(),
        'probabilities': all_probabilities.tolist(),
        'image_ids': all_image_ids
    }
    
    print(f"✅ Evaluation completed")
    return results


def calculate_comprehensive_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray
) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        targets: True labels
        predictions: Predicted labels  
        probabilities: Prediction probabilities
        
    Returns:
        Dictionary with all calculated metrics
        
    Attribution:
    - Metrics implementation: scikit-learn library
    - Macro/micro averaging: Standard multi-class evaluation practices
    """
    class_names = ['disgusting', 'neutral', 'tasty']
    
    # Basic metrics
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average='macro')
    micro_f1 = f1_score(targets, predictions, average='micro')
    weighted_f1 = f1_score(targets, predictions, average='weighted')
    
    # Per-class F1 scores
    per_class_f1 = f1_score(targets, predictions, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Classification report
    class_report = classification_report(
        targets, predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # ROC AUC (for multi-class)
    try:
        # Binarize labels for multi-class ROC
        targets_binarized = label_binarize(targets, classes=[0, 1, 2])
        if targets_binarized.shape[1] == 1:  # Handle case with only 2 classes
            roc_auc = roc_auc_score(targets, probabilities[:, 1])
        else:
            roc_auc = roc_auc_score(targets_binarized, probabilities, multi_class='ovr', average='macro')
    except ValueError:
        roc_auc = None  # Not enough classes for ROC AUC
    
    # Confidence analysis
    max_probs = np.max(probabilities, axis=1)
    avg_confidence = np.mean(max_probs)
    confident_predictions = np.sum(max_probs > 0.7) / len(max_probs)
    
    # Per-class analysis
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_mask = targets == i
        if np.sum(class_mask) > 0:
            class_predictions = predictions[class_mask]
            class_accuracy = np.sum(class_predictions == i) / len(class_predictions)
            per_class_metrics[class_name] = {
                'count': int(np.sum(class_mask)),
                'accuracy': float(class_accuracy),
                'f1_score': float(per_class_f1[i])
            }
    
    results = {
        'overall_metrics': {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'micro_f1': float(micro_f1),
            'weighted_f1': float(weighted_f1),
            'roc_auc': float(roc_auc) if roc_auc is not None else None
        },
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'confidence_analysis': {
            'average_confidence': float(avg_confidence),
            'high_confidence_ratio': float(confident_predictions)
        }
    }
    
    return results


def create_evaluation_plots(
    results: Dict,
    output_dir: str,
    title_prefix: str = ""
) -> None:
    """
    Create comprehensive evaluation plots.
    
    Args:
        results: Results dictionary from evaluate_model
        output_dir: Directory to save plots
        title_prefix: Prefix for plot titles
        
    Attribution:
    - Visualization: matplotlib and seaborn libraries
    - Confusion matrix heatmap: Standard ML visualization practice
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ['disgusting', 'neutral', 'tasty']
    
    # Confusion Matrix
    cm = np.array(results['confusion_matrix'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{title_prefix}Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add accuracy annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i == j:  # Diagonal elements (correct predictions)
                class_total = np.sum(cm[i, :])
                accuracy = cm[i, j] / class_total if class_total > 0 else 0
                plt.text(j+0.5, i-0.3, f'({accuracy:.1%})', 
                        ha='center', va='center', fontsize=10, color='darkblue')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-class Performance Bar Chart
    per_class = results['per_class_metrics']
    classes = list(per_class.keys())
    f1_scores = [per_class[cls]['f1_score'] for cls in classes]
    accuracies = [per_class[cls]['accuracy'] for cls in classes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # F1 Scores
    bars1 = ax1.bar(classes, f1_scores, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax1.set_title(f'{title_prefix}Per-Class F1 Scores')
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0, 1)
    for bar, score in zip(bars1, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Accuracies
    bars2 = ax2.bar(classes, accuracies, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax2.set_title(f'{title_prefix}Per-Class Accuracies')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    for bar, acc in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confidence Distribution
    if 'prediction_details' in results:
        probabilities = np.array(results['prediction_details']['probabilities'])
        max_probs = np.max(probabilities, axis=1)
        
        plt.figure(figsize=(8, 6))
        plt.hist(max_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(max_probs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_probs):.3f}')
        plt.xlabel('Maximum Probability (Confidence)')
        plt.ylabel('Number of Predictions')
        plt.title(f'{title_prefix}Prediction Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Evaluation plots saved to {output_dir}")


def compare_models(
    model_paths: Dict[str, str],
    test_csv: str,
    output_dir: str,
    data_dir: str = 'data/food101_cache'
) -> Dict:
    """
    Compare multiple trained models on the same test set.
    
    Args:
        model_paths: Dictionary mapping model names to checkpoint paths
        test_csv: Path to test set CSV
        output_dir: Directory to save comparison results
        data_dir: Data directory path
        
    Returns:
        Dictionary containing comparison results
        
    Attribution:
    - Model comparison: Standard ML model selection practice
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_results = {}
    
    print(f"🔍 Comparing {len(model_paths)} models...")
    
    for model_name, model_path in model_paths.items():
        print(f"\n📊 Evaluating {model_name}...")
        
        # Load and evaluate model
        model = load_trained_model(model_path, device)
        results = evaluate_model(model, test_csv, data_dir, device=device)
        
        # Store results
        comparison_results[model_name] = results
        
        # Create individual plots
        model_output_dir = output_dir / model_name
        create_evaluation_plots(results, model_output_dir, f"{model_name} - ")
    
    # Create comparison plots
    create_comparison_plots(comparison_results, output_dir)
    
    # Save comparison summary
    with open(output_dir / 'model_comparison.json', 'w') as f:
        # Remove prediction details for JSON serialization
        clean_results = {}
        for name, results in comparison_results.items():
            clean_results[name] = {k: v for k, v in results.items() if k != 'prediction_details'}
        json.dump(clean_results, f, indent=2)
    
    print(f"✅ Model comparison completed. Results saved to {output_dir}")
    return comparison_results


def create_comparison_plots(comparison_results: Dict, output_dir: str) -> None:
    """Create plots comparing multiple models."""
    output_dir = Path(output_dir)
    
    model_names = list(comparison_results.keys())
    
    # Extract metrics for comparison
    accuracies = [comparison_results[name]['overall_metrics']['accuracy'] for name in model_names]
    macro_f1s = [comparison_results[name]['overall_metrics']['macro_f1'] for name in model_names]
    
    # Overall performance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, color=['#ff6b6b', '#4ecdc4', '#45b7d1'][:len(model_names)])
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # F1 score comparison
    bars2 = ax2.bar(model_names, macro_f1s, color=['#ff6b6b', '#4ecdc4', '#45b7d1'][:len(model_names)])
    ax2.set_title('Model Macro F1-Score Comparison')
    ax2.set_ylabel('Macro F1-Score')
    ax2.set_ylim(0, 1)
    for bar, f1 in zip(bars2, macro_f1s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plots saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Module")
    print("Usage:")
    print("1. evaluate_model(model, test_csv) - Evaluate single model")
    print("2. compare_models(model_paths, test_csv) - Compare multiple models")
    print("3. create_evaluation_plots(results, output_dir) - Create visualizations")
