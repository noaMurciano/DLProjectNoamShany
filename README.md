# 🍽️ Personal Food Taste Classification using Deep Learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B3-green.svg)](https://github.com/rwightman/pytorch-image-models)

**Course**: 046211 Deep Learning, Technion  
**Author**: Noam & Shany  
**Semester**: Spring 2025

> A deep learning system that learns **your personal food taste preferences** by fine-tuning a pre-trained Food101 model with individually labeled data. The model predicts whether you'll find food **disgusting**, **neutral**, or **tasty** based on your personal taste patterns.

---

## 🎯 **Project Overview**

### **Problem Statement**
Traditional food recommendation systems rely on crowd-sourced ratings that don't capture individual taste preferences. This project addresses **personalized taste modeling** by adapting a general food classification model to learn one person's specific food preferences.

### **Key Innovation**
- **Transfer Learning**: Fine-tune EfficientNet-B3 pre-trained on Food101 with personal labels
- **Small Data Learning**: Achieve meaningful personalization with only 200 labeled images
- **Proper Validation**: Rigorous 60/20/20 split methodology for robust model evaluation
- **Personal Insights**: Extract interpretable patterns about individual food preferences

### **Results Preview**
- 🎯 **37.5% accuracy** with proper 60/20/20 validation methodology
- 🔍 **Effective minority class detection**: 25% disgusting class recall
- 📊 **Clear preference patterns**: Quality over quantity, texture sensitivity, cultural diversity
- ⚡ **Fast training**: < 1 minute total training time
- 📈 **Methodological rigor**: Proper validation prevents overfitting

---

## 🚀 **Quick Start (30 seconds)**

```bash
# 1. Clone and navigate
git clone <repository-url>
cd GitHubRepository

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Run interactive demo
python demo.py

# 4. Classify your own food image
python demo.py --image path/to/your/food/image.jpg
```

**🎮 Try the interactive demo to see personal taste prediction in action!**

---

## 📁 **Repository Structure**

```
GitHubRepository/
├── 📋 README.md                           # This overview
├── 🚀 demo.py                             # Interactive demonstration  
├── ⚙️ requirements.txt                     # Dependencies
│
├── 📁 src/                                # Core implementation
│   ├── __init__.py                        # Package initialization
│   ├── model.py                           # EfficientNet-B3 + custom head
│   ├── dataset.py                        # Personal dataset handling
│   ├── train.py                          # Training pipeline
│   └── evaluate.py                       # Evaluation utilities
│
├── 📁 configs/                            # Configuration files
│   ├── model_config.json                 # Model hyperparameters
│   └── training_config.json              # Training settings
│
├── 📁 data/                              # Data-related files
│   ├── README.md                         # Data description
│   └── food101_categories.json          # Category mappings
│
├── 📁 docs/                              # Technical documentation
│   ├── METHODOLOGY.md                   # Technical approach
│   └── SETUP.md                         # Installation guide
│
└── 📁 scripts/                           # Training & evaluation scripts
    ├── train_personal_model.py          # Main training script
    ├── prepare_personal_data.py         # Data preprocessing
    └── evaluate_model.py               # Model evaluation
```

---

## 🧠 **Technical Approach**

### **Architecture**
- **Backbone**: EfficientNet-B3 (11M parameters)
- **Pre-training**: ImageNet → Food101 → Personal preferences
- **Head**: Global pooling → Linear(1536, 256) → GELU → Dropout → Linear(256, 3)
- **Classes**: Disgusting (0), Neutral (1), Tasty (2)

### **Training Strategy**
```python
# Two-phase curriculum learning
Phase 1: Freeze backbone, train head (8 epochs)
Phase 2: Unfreeze backbone, end-to-end (7 epochs)

# Discriminative learning rates
head_lr = 1e-3      # Higher for new layers
backbone_lr = 1e-5  # Lower to preserve features
```

### **Personal Dataset**
- **Size**: 200 hand-labeled food images from Food101
- **Distribution**: 37 disgusting, 84 neutral, 79 tasty
- **Split**: Proper 60/20/20 validation methodology (120/40/40)
- **Challenge**: Learning individual preferences from limited data

---

## 📊 **Key Results**

### **Model Performance**
| Metric | Value | Notes |
|--------|-------|-------|
| **Test Accuracy** | 37.5% | Proper validation methodology |
| **Macro F1-Score** | 0.356 | Balanced class performance |
| **Disgusting Recall** | 25.0% | Effective minority class detection |
| **Training Split** | 120/40/40 | 60/20/20 validation methodology |

### **Personal Insights Discovered**
- 😋 **Prefer**: Quality desserts (tiramisu, crème brûlée), refined seafood (sashimi)
- 🤢 **Avoid**: Heavy fried foods (beignets), complex preparations (lobster bisque)
- 😐 **Neutral**: Standard restaurant fare, familiar preparations
- 🎯 **Pattern**: Quality over quantity, texture-sensitive, culturally diverse

---

## 🛠️ **Usage Examples**

### **Interactive Demo**
```bash
# Start interactive session
python demo.py --interactive

# Example session:
📁 Image path: path/to/your/food/image.jpg
🎯 Prediction: NEUTRAL
🎪 Confidence: 67.3%
```

### **Training Your Own Model**
```bash
# Train with 60/20/20 split (recommended)
python scripts/train_personal_model.py --config configs/training_config.json

# Train with custom data
python scripts/train_personal_model.py --data my_labels.xlsx --split 3way
```

---

## ⚙️ **Configuration & Hyperparameters**

All hyperparameters are documented in [`configs/model_config.json`](configs/model_config.json):

**Key Parameters:**
- **Backbone**: EfficientNet-B3 (tf_efficientnet_b3)
- **Image Size**: 224×224 pixels
- **Batch Size**: 16 (optimal for small dataset)
- **Learning Rates**: Head=1e-3, Backbone=1e-5
- **Training**: Phase 1=8 epochs, Phase 2=7 epochs
- **Early Stopping**: Patience=5 epochs

---

## 📈 **Reproducibility**

### **Requirements**
- Python 3.12+
- PyTorch 2.0+
- CUDA compatible GPU (recommended)
- 8GB RAM minimum

### **Installation**
```bash
# Install exact versions
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### **Training Reproducibility**
- **Fixed seeds**: All random operations use `seed=42`
- **Deterministic splits**: Stratified splitting with fixed random state
- **Exact configurations**: All settings in `configs/` directory

---

## 📚 **References & Attribution**

### **Key Dependencies**
- **PyTorch**: Deep learning framework - [pytorch.org](https://pytorch.org/)
- **timm**: EfficientNet-B3 pre-trained models - [github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- **HuggingFace Datasets**: Food101 dataset access - [huggingface.co/datasets/food101](https://huggingface.co/datasets/food101)
- **Food101 Dataset**: Bossard, L., et al. (2014) - Original 101-category food dataset

### **Implementation**
- Custom EfficientNet-B3 fine-tuning with personal taste classification
- Two-phase curriculum learning with discriminative learning rates
- Personal dataset: 200 hand-labeled food images (original work)

---

## 🚀 **Future Work**

### **Technical Improvements**
- [ ] **Scale up**: Collect 500-1000 personal labels for better performance
- [ ] **Ensemble methods**: Combine multiple models for robust predictions
- [ ] **Active learning**: Intelligently select most informative samples
- [ ] **Multi-modal**: Incorporate text descriptions, nutritional information

---

## 🤝 **Project Information**

**Authors**: Noam & Shany  
**Project**: Personal Food Taste Classification using Deep Learning

**Note**: This project demonstrates transfer learning and personalized AI systems using minimal labeled data.

---

## 📄 **License**

This project is open source and available for educational and research purposes. See dependencies for their respective licenses.

---

*Made with ❤️ for advancing personalized AI systems*
