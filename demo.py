#!/usr/bin/env python3
"""
Personal Food Taste Classification Demo

Quick demonstration script for the personal food taste classification system.
This script provides an interactive way to test the trained models.

Authors: Noam & Shany

Attribution:
- Model architecture: Custom EfficientNet-B3 based classifier
- EfficientNet: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling. ICML.
- timm library: https://github.com/rwightman/pytorch-image-models
- PyTorch: https://pytorch.org/
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import argparse
import os
from pathlib import Path

# Import our modules
from src.model import create_model

class PersonalTasteDemo:
    """Interactive demo for personal food taste classification."""
    
    def __init__(self, model_path=None):
        """Initialize the demo with a trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = {0: 'disgusting', 1: 'neutral', 2: 'tasty'}
        
        # Default to best model if not specified
        if model_path is None:
            # Check for trained models
            default_paths = [
                "experiments/personal_60_20_20/best_model.pt",
                "experiments/personal_80_20/best_model.pt",
                "experiments/base_model/best_model.pt"
            ]
            model_path = None
            for path in default_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                print("❌ No trained models found!")
                print("🚀 Please train a model first using:")
                print("   python scripts/train_personal_model.py --data your_labels.xlsx")
                raise FileNotFoundError("No trained models available")
        
        self.model = self.load_model(model_path)
        self.transform = self.get_transforms()
        
        print(f"🍽️ Personal Food Taste Classifier Demo")
        print(f"Device: {self.device}")
        print(f"Model: {model_path}")
        print("="*50)
    
    def get_transforms(self):
        """Get image preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        """Load the trained model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model config
            model_config = checkpoint.get('config', {}).get('model', {
                'num_classes': 3,
                'backbone': 'tf_efficientnet_b3',
                'pretrained': True,
                'dropout': 0.3,
                'hidden_dim': 256
            })
            
            # Create and load model
            model = create_model(model_config, self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"✅ Model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🚀 Please train a model first using:")
            print("   python scripts/train_personal_model.py --data your_labels.xlsx")
            print("📁 Models will be saved in experiments/ directory")
            raise
    
    def predict_image(self, image_path):
        """Predict taste preference for a food image."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
            
            # Format results
            results = {
                'predicted_class': predicted_class,
                'predicted_label': self.class_names[predicted_class],
                'confidence': probabilities[0][predicted_class].item(),
                'all_probabilities': {
                    self.class_names[i]: probabilities[0][i].item() 
                    for i in range(3)
                }
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return None
    
    def print_results(self, results, image_path):
        """Print prediction results in a nice format."""
        if results is None:
            return
        
        print(f"\n📸 Image: {image_path}")
        print(f"🎯 Prediction: {results['predicted_label'].upper()}")
        print(f"🎪 Confidence: {results['confidence']:.1%}")
        print(f"\n📊 All Probabilities:")
        
        for label, prob in results['all_probabilities'].items():
            bar_length = int(prob * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {label:>10}: {bar} {prob:.1%}")
        
        # Interpretation
        print(f"\n💭 Interpretation:")
        if results['confidence'] > 0.7:
            print(f"   High confidence - the model is quite sure this food is {results['predicted_label']}")
        elif results['confidence'] > 0.5:
            print(f"   Moderate confidence - the model thinks this food is {results['predicted_label']}")
        else:
            print(f"   Low confidence - the model is uncertain about this prediction")
    
    def interactive_mode(self):
        """Run interactive demo mode."""
        print(f"\n🎮 Interactive Mode")
        print(f"Enter image paths to classify (type 'quit' to exit)")
        print(f"Example: path/to/your/food/image.jpg")
        
        while True:
            image_path = input(f"\n📁 Image path: ").strip()
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                print(f"👋 Thanks for using the demo!")
                break
            
            if not Path(image_path).exists():
                print(f"❌ File not found: {image_path}")
                continue
            
            results = self.predict_image(image_path)
            self.print_results(results, image_path)

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Personal Food Taste Classification Demo')
    parser.add_argument('--image', type=str, help='Path to food image to classify')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        # Initialize demo
        demo = PersonalTasteDemo(model_path=args.model)
        
        if args.image:
            # Single image prediction
            results = demo.predict_image(args.image)
            demo.print_results(results, args.image)
            
        elif args.interactive:
            # Interactive mode
            demo.interactive_mode()
            
        else:
            # Default: show demo info and run interactive
            print(f"\n🎯 First time? Train a model:")
            print(f"  python scripts/train_personal_model.py --data your_labels.xlsx")
            print(f"  Models will be saved in experiments/ directory")
            
            print(f"\n🚀 Usage Examples:")
            print(f"  python demo.py --image path/to/food/image.jpg")
            print(f"  python demo.py --interactive")
            print(f"  python demo.py --image pizza.jpg --model experiments/personal_60_20_20/best_model.pt")
            
            # Ask user if they want to run interactive mode
            response = input(f"\nRun interactive mode? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                demo.interactive_mode()
                
    except KeyboardInterrupt:
        print(f"\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    main()
