#!/usr/bin/env python3
"""
Model Evaluation Script

Comprehensive evaluation of trained personal food taste classification models.

Author: Noam Shani
Course: 046211 Deep Learning, Technion

Usage:
    python scripts/evaluate_model.py --model experiments/personal_60_20_20/best_model.pt --test-data data/personal_test_3way.csv
    python scripts/evaluate_model.py --compare-all --output-dir results/comparison/
"""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.evaluate import (
    load_trained_model, evaluate_model, compare_models, 
    create_evaluation_plots
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Personal Food Taste Classification Models')
    
    # Single model evaluation
    parser.add_argument('--model', type=str,
                       help='Path to model checkpoint for single model evaluation')
    parser.add_argument('--test-data', type=str, default='data/personal_test_3way.csv',
                       help='Path to test set CSV file')
    
    # Model comparison
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all available models')
    parser.add_argument('--models', nargs='+',
                       help='List of model paths for comparison')
    parser.add_argument('--model-names', nargs='+',
                       help='Names for models (if not provided, use filenames)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Output directory for results')
    parser.add_argument('--data-dir', type=str, default='data/food101_cache',
                       help='HuggingFace dataset cache directory')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    
    # Display options
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating evaluation plots')
    
    return parser.parse_args()

def find_available_models(experiments_dir: str = 'experiments') -> dict:
    """Find all available trained models."""
    experiments_path = Path(experiments_dir)
    
    if not experiments_path.exists():
        print(f"⚠️ Experiments directory not found: {experiments_dir}")
        return {}
    
    models = {}
    
    for exp_dir in experiments_path.iterdir():
        if exp_dir.is_dir():
            model_file = exp_dir / 'best_model.pt'
            if model_file.exists():
                models[exp_dir.name] = str(model_file)
    
    return models

def evaluate_single_model(args):
    """Evaluate a single model."""
    print(f"🧪 Single Model Evaluation")
    print(f"Model: {args.model}")
    print(f"Test data: {args.test_data}")
    print("=" * 50)
    
    # Check if files exist
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return
    
    if not Path(args.test_data).exists():
        print(f"❌ Test data file not found: {args.test_data}")
        return
    
    try:
        # Load model
        model = load_trained_model(args.model)
        
        # Evaluate model
        results = evaluate_model(
            model, 
            args.test_data, 
            args.data_dir, 
            args.batch_size
        )
        
        # Print results
        print_evaluation_results(results, args.model)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            # Remove prediction details for JSON serialization
            clean_results = {k: v for k, v in results.items() if k != 'prediction_details'}
            json.dump(clean_results, f, indent=2)
        
        # Create plots
        if not args.no_plots:
            create_evaluation_plots(results, output_dir, "Single Model - ")
        
        print(f"✅ Evaluation completed. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

def compare_multiple_models(args):
    """Compare multiple models."""
    print(f"🔍 Multi-Model Comparison")
    print("=" * 50)
    
    if args.compare_all:
        # Find all available models
        model_paths = find_available_models()
        
        if not model_paths:
            print("❌ No trained models found in experiments/ directory")
            return
        
        print(f"Found {len(model_paths)} trained models:")
        for name, path in model_paths.items():
            print(f"  - {name}: {path}")
        
    else:
        # Use specified models
        if not args.models:
            print("❌ No models specified for comparison")
            return
        
        model_paths = {}
        model_names = args.model_names or []
        
        for i, model_path in enumerate(args.models):
            if not Path(model_path).exists():
                print(f"⚠️ Model file not found: {model_path}")
                continue
            
            if i < len(model_names):
                name = model_names[i]
            else:
                name = Path(model_path).parent.name
            
            model_paths[name] = model_path
    
    if not model_paths:
        print("❌ No valid models to compare")
        return
    
    # Check test data
    if not Path(args.test_data).exists():
        print(f"❌ Test data file not found: {args.test_data}")
        return
    
    try:
        # Compare models
        comparison_results = compare_models(
            model_paths,
            args.test_data,
            args.output_dir,
            args.data_dir
        )
        
        # Print comparison summary
        print_comparison_summary(comparison_results)
        
        print(f"✅ Model comparison completed. Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Model comparison failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

def print_evaluation_results(results: dict, model_name: str):
    """Print formatted evaluation results."""
    print(f"\n📊 Evaluation Results for {Path(model_name).parent.name}")
    print("-" * 60)
    
    # Overall metrics
    overall = results['overall_metrics']
    print(f"Overall Performance:")
    print(f"  Accuracy:     {overall['accuracy']:.1%}")
    print(f"  Macro F1:     {overall['macro_f1']:.4f}")
    print(f"  Weighted F1:  {overall['weighted_f1']:.4f}")
    if overall['roc_auc'] is not None:
        print(f"  ROC AUC:      {overall['roc_auc']:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Performance:")
    per_class = results['per_class_metrics']
    for class_name, metrics in per_class.items():
        print(f"  {class_name.capitalize():>10}: "
              f"F1={metrics['f1_score']:.3f}, "
              f"Acc={metrics['accuracy']:.1%}, "
              f"Count={metrics['count']}")
    
    # Confidence analysis
    confidence = results['confidence_analysis']
    print(f"\nConfidence Analysis:")
    print(f"  Average confidence: {confidence['average_confidence']:.1%}")
    print(f"  High confidence (>70%): {confidence['high_confidence_ratio']:.1%}")

def print_comparison_summary(comparison_results: dict):
    """Print comparison summary."""
    print(f"\n📊 Model Comparison Summary")
    print("-" * 60)
    
    # Create comparison table
    print(f"{'Model':<20} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12}")
    print("-" * 52)
    
    for model_name, results in comparison_results.items():
        overall = results['overall_metrics']
        print(f"{model_name:<20} "
              f"{overall['accuracy']:.1%}      "
              f"{overall['macro_f1']:.4f}    "
              f"{overall['weighted_f1']:.4f}")
    
    # Find best model
    best_accuracy = max(comparison_results.items(), 
                       key=lambda x: x[1]['overall_metrics']['accuracy'])
    best_f1 = max(comparison_results.items(), 
                  key=lambda x: x[1]['overall_metrics']['macro_f1'])
    
    print(f"\n🏆 Best Models:")
    print(f"  Highest Accuracy: {best_accuracy[0]} ({best_accuracy[1]['overall_metrics']['accuracy']:.1%})")
    print(f"  Highest Macro F1: {best_f1[0]} ({best_f1[1]['overall_metrics']['macro_f1']:.4f})")

def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    print("🍽️ Personal Food Taste Classification - Model Evaluation")
    print("=" * 60)
    
    if args.model:
        # Single model evaluation
        evaluate_single_model(args)
    elif args.compare_all or args.models:
        # Multi-model comparison
        compare_multiple_models(args)
    else:
        # Show help if no specific action
        print("Please specify either:")
        print("  --model <path>     for single model evaluation")
        print("  --compare-all      to compare all trained models")
        print("  --models <paths>   to compare specific models")
        print("\nExample usage:")
        print("  python scripts/evaluate_model.py --model experiments/personal_60_20_20/best_model.pt")
        print("  python scripts/evaluate_model.py --compare-all")

if __name__ == "__main__":
    main()
