# 🍽️ Personal Food Taste Classification using Deep Learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B3-green.svg)](https://github.com/rwightman/pytorch-image-models)

**Course**: 046211 Deep Learning, Technion  
**Author**: Noam Shani  
**Semester**: Spring 2025

> A deep learning system that learns **your personal food taste preferences** by fine-tuning a pre-trained Food101 model with individually labeled data. The model predicts whether you'll find food **disgusting**, **neutral**, or **tasty** based on your personal taste patterns.

---

## 🎯 **Project Overview**

### **Problem Statement**
Traditional food recommendation systems rely on crowd-sourced ratings that don't capture individual taste preferences. This project addresses **personalized taste modeling** by adapting a general food classification model to learn one person's specific food preferences.

### **Key Innovation**
- **Transfer Learning**: Fine-tune EfficientNet-B3 pre-trained on Food101 with personal labels
- **Small Data Learning**: Achieve meaningful personalization with only 199 labeled images
- **Methodology Comparison**: Rigorous evaluation of 80/20 vs 60/20/20 data splits
- **Personal Insights**: Extract interpretable patterns about individual food preferences

### **Results Preview**
- 🎯 **52.5% accuracy** (80/20 split) vs **37.5% accuracy** (60/20/20 split)
- 🔍 **Improved disgusting detection**: 0% → 25% with proper validation methodology
- 📊 **Clear preference patterns**: Quality over quantity, texture sensitivity, cultural diversity
- ⚡ **Fast training**: < 1 minute total training time

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
│   ├── training_config.json              # Training settings
│   └── data_config.json                  # Data processing config
│
├── 📁 experiments/                        # Trained models & results
│   ├── base_model/                       # Original Food101 model
│   ├── personal_80_20/                   # 80/20 split results
│   └── personal_60_20_20/                # 60/20/20 split results (recommended)
│
├── 📁 data/                              # Data-related files
│   ├── README.md                         # Data description
│   ├── food101_categories.json          # Category mappings
│   └── sample_results.csv               # Example outputs
│
├── 📁 docs/                              # Technical documentation  
│   ├── METHODOLOGY.md                   # Detailed technical approach
│   ├── RESULTS.md                       # Comprehensive results analysis
│   └── SETUP.md                         # Installation & setup guide
│
├── 📁 scripts/                           # Training & evaluation scripts
│   ├── train_personal_model.py          # Main training script
│   ├── prepare_personal_data.py         # Data preprocessing
│   └── evaluate_model.py               # Model evaluation
│
└── 📁 assets/                            # Media files & visualizations
    ├── architecture_diagram.png
    ├── results_comparison.png
    └── confusion_matrices/
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
- **Size**: 199 hand-labeled food images from Food101
- **Distribution**: 37 disgusting, 84 neutral, 78 tasty
- **Splits**: 80/20 vs 60/20/20 comparison study
- **Challenge**: Learning individual preferences from limited data

---

## 📊 **Key Results**

### **Performance Comparison**
| Approach | Train | Val | Test | Test Acc | Disgusting F1 | Notes |
|----------|-------|-----|------|----------|---------------|-------|
| **80/20 Split** | 159 | - | 40 | **52.5%** | 0.0% | Higher accuracy, overfitting |
| **60/20/20 Split** | 119 | 40 | 40 | 37.5% | **25.0%** | Better methodology |

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
📁 Image path: experiments/base_model/sample_images/pizza.jpg
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

## 📚 **Technical References & Attribution**

### **Core Research Papers**
1. **EfficientNet Architecture**: Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks.* International Conference on Machine Learning (ICML).
2. **Food101 Dataset**: Bossard, L., Guillaumin, M., & Van Gool, L. (2014). *Food-101–mining discriminative components with random forests.* European Conference on Computer Vision (ECCV).
3. **Transfer Learning**: Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are features in deep neural networks?* Advances in Neural Information Processing Systems (NIPS).
4. **Discriminative Learning Rates**: Howard, J., & Ruder, S. (2018). *Universal language model fine-tuning for text classification.* Association for Computational Linguistics (ACL).
5. **Label Smoothing**: Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). *Rethinking the inception architecture for computer vision.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
6. **Class Imbalance**: He, H., & Garcia, E. A. (2009). *Learning from imbalanced data.* IEEE Transactions on Knowledge and Data Engineering (TKDE).

### **Software Libraries & Code Attribution**

#### **Core Deep Learning Framework**
- **PyTorch**: Paszke, A., et al. (2019). *PyTorch: An imperative style, high-performance deep learning library.* 
  - Website: [https://pytorch.org/](https://pytorch.org/)
  - Used for: Model architecture, training pipeline, data loading
  - License: BSD 3-Clause License

#### **Pre-trained Models**
- **timm (PyTorch Image Models)**: Ross Wightman
  - Repository: [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
  - Used for: EfficientNet-B3 pre-trained weights and architecture
  - Specific model: `tf_efficientnet_b3` (TensorFlow compatible weights)
  - License: Apache 2.0 License

#### **Dataset Access**
- **HuggingFace Datasets**: Wolf, T., et al. (2020). *Transformers: State-of-the-art natural language processing.*
  - Repository: [https://github.com/huggingface/datasets](https://github.com/huggingface/datasets)
  - Dataset: [https://huggingface.co/datasets/food101](https://huggingface.co/datasets/food101)
  - Used for: Food101 dataset loading and caching
  - License: Apache 2.0 License

#### **Computer Vision & Image Processing**
- **torchvision**: PyTorch Team
  - Website: [https://pytorch.org/vision/](https://pytorch.org/vision/)
  - Used for: Image transformations, data augmentation, normalization
  - License: BSD 3-Clause License

#### **Data Science & Analysis**
- **pandas**: McKinney, W. (2010). *Data structures for statistical computing in Python.*
  - Used for: Data manipulation, CSV handling, analysis
  - License: BSD 3-Clause License
- **NumPy**: Harris, C. R., et al. (2020). *Array programming with NumPy.*
  - Used for: Numerical computations, array operations
  - License: BSD 3-Clause License
- **scikit-learn**: Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python.*
  - Used for: Train/test splits, stratified sampling, metrics
  - License: BSD 3-Clause License

#### **Visualization**
- **Matplotlib**: Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment.*
  - Used for: Plotting, confusion matrices, training curves
  - License: PSF License
- **Seaborn**: Waskom, M. L. (2021). *seaborn: statistical data visualization.*
  - Used for: Statistical plots, data visualization
  - License: BSD 3-Clause License

### **Academic Integrity Statement**
This project is developed as original work for course 046211 Deep Learning at Technion. All external code, models, and datasets are properly attributed above. The personal taste labeling, model fine-tuning approach, and validation methodology comparison represent original research contributions.

### **Implementation Attribution**
- **Model Architecture**: Custom implementation combining EfficientNet-B3 backbone with personal classification head
- **Loss Function**: Custom `TasteClassificationLoss` combining cross-entropy with optional consistency loss
- **Dataset Integration**: Custom `PersonalTasteDataset` class integrating personal labels with HuggingFace Food101
- **Training Pipeline**: Two-phase curriculum learning with discriminative learning rates
- **Evaluation Framework**: Comprehensive metrics including macro F1-score for class imbalance

### **Data Attribution**
- **Food101 Original Dataset**: Bossard, L., et al. (2014) - 101,000 images across 101 food categories
- **Personal Labels**: 199 manually labeled images with individual taste preferences (original work)
- **Category Mapping**: Extracted from HuggingFace Food101 dataset for index-to-name conversion

---

## 🚀 **Future Work**

### **Technical Improvements**
- [ ] **Scale up**: Collect 500-1000 personal labels for better performance
- [ ] **Ensemble methods**: Combine multiple models for robust predictions
- [ ] **Active learning**: Intelligently select most informative samples
- [ ] **Multi-modal**: Incorporate text descriptions, nutritional information

---

## 🤝 **Academic Information**

**Course**: 046211 Deep Learning, Technion  
**Author**: Noam Shani  
**Semester**: Spring 2025  

**Academic Integrity**: This implementation follows all course guidelines for original work and proper attribution. All external code and models are properly credited.

---

## 📄 **License**

This project is developed for academic purposes as part of course 046211 Deep Learning at Technion. For academic use and reference only.

---

*Made with ❤️ for 046211 Deep Learning Course*
