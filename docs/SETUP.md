# Setup Guide

This guide provides step-by-step instructions for setting up and running the Personal Food Taste Classification project.

## 🚀 **Quick Setup (5 minutes)**

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd GitHubRepository

# Create conda environment (recommended)
conda create -n food-taste python=3.12
conda activate food-taste

# OR create virtual environment
python -m venv food-taste-env
source food-taste-env/bin/activate  # Linux/Mac
# food-taste-env\Scripts\activate   # Windows
```

### 2. Install Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
```

### 3. Quick Test
```bash
# Test with demo
python demo.py

# Should output:
# 🍽️ Personal Food Taste Classifier Demo
# Device: cuda/cpu
# Note: You'll need to train a model first using scripts/
```

---

## 📋 **Detailed Setup**

### **System Requirements**

**Minimum:**
- Python 3.12+
- 8GB RAM
- 2GB free disk space
- CPU (Intel/AMD x64)

**Recommended:**
- Python 3.12+
- 16GB+ RAM
- CUDA-compatible GPU
- 10GB+ free disk space

### **Dependencies Overview**

**Core Libraries:**
```bash
torch>=2.0.0              # Deep learning framework
torchvision>=0.15.0        # Computer vision utilities
timm>=0.9.0                # Pre-trained models
```

**Data Processing:**
```bash
pandas>=1.5.0              # Data manipulation
numpy>=1.24.0              # Numerical computing
scikit-learn>=1.3.0        # Machine learning utilities
datasets>=2.14.0           # HuggingFace datasets
```

**Visualization:**
```bash
matplotlib>=3.7.0          # Plotting
seaborn>=0.12.0            # Statistical visualization
```

---

## 🔧 **Environment Configuration**

### **Conda Environment (Recommended)**
```bash
# Create environment with Python 3.12
conda create -n food-taste python=3.12

# Activate environment
conda activate food-taste

# Install PyTorch with CUDA (if available)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining packages
pip install timm datasets pandas scikit-learn matplotlib seaborn openpyxl PyMuPDF tqdm
```

### **Virtual Environment Alternative**
```bash
# Create virtual environment
python -m venv food-taste-env

# Activate
source food-taste-env/bin/activate  # Linux/Mac
# food-taste-env\Scripts\activate   # Windows

# Install all requirements
pip install -r requirements.txt
```

### **Docker Setup (Optional)**
```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "demo.py"]
```

```bash
# Build and run
docker build -t food-taste-classifier .
docker run -it food-taste-classifier
```

---

## 📊 **Data Setup**

### **Model Training Required**
This repository contains the framework but requires training your own models:

```bash
# Train your first model using the provided scripts
python scripts/train_personal_model.py --data your_labels.xlsx --split 3way

# Models will be saved in a new experiments/ directory
# See training section below for detailed instructions
```

### **Food101 Dataset (Automatic)**
The HuggingFace Food101 dataset will be automatically downloaded on first use:

```bash
# Test dataset loading
python -c "
from datasets import load_dataset
dataset = load_dataset('food101')
print(f'Food101 loaded: {len(dataset[\"train\"])} train samples')
"
```

### **Personal Dataset Preparation**
To use your own labeled data:

```bash
# Prepare your personal labels in Excel format
# Columns: image_path, label (disgusting/neutral/tasty)

# Run data preparation
python scripts/prepare_personal_data.py --data your_labels.xlsx
```

---

## 🎯 **Usage Examples**

### **Interactive Demo**
```bash
# Start interactive session
python demo.py --interactive

# Test with specific image
python demo.py --image path/to/food/image.jpg

# Use trained model (after training)
python demo.py --model experiments/personal_60_20_20/best_model.pt --interactive
```

### **Training Your Own Model**
```bash
# Train with 60/20/20 split (recommended)
python scripts/train_personal_model.py \
    --data data/personal_labels.xlsx \
    --split 3way \
    --epochs 15 \
    --output-dir experiments/

# Train with custom parameters
python scripts/train_personal_model.py \
    --config configs/training_config.json \
    --lr-head 0.001 \
    --lr-backbone 0.00001 \
    --batch-size 16
```

### **Model Evaluation**
```bash
# Evaluate trained model performance
python scripts/evaluate_model.py \
    --model experiments/personal_60_20_20/best_model.pt \
    --test-data personal_test_3way.csv

# Note: Test data and models are created during training
```

---

## 🐛 **Troubleshooting**

### **Common Issues**

**1. Import Errors**
```bash
# Error: No module named 'timm'
pip install timm

# Error: No module named 'datasets'
pip install datasets

# Error: No module named 'openpyxl'
pip install openpyxl
```

**2. CUDA Issues**
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CPU-only PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**3. Memory Issues**
```bash
# Reduce batch size in configs/training_config.json
"batch_size": 8  # Instead of 16

# Or use gradient accumulation
"gradient_accumulation_steps": 2
```

**4. Dataset Loading Errors**
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/datasets/food101

# Re-download dataset
python -c "from datasets import load_dataset; load_dataset('food101', cache_dir='data/food101_cache')"
```

### **Performance Optimization**

**GPU Optimization:**
```bash
# Check GPU memory
nvidia-smi

# Enable mixed precision in config
"mixed_precision": true

# Use smaller image size if needed
"image_size": 224  # Default, try 192 for less memory
```

**CPU Optimization:**
```bash
# Reduce number of workers
"num_workers": 2  # Default, reduce to 0-1 for limited CPU

# Disable pin_memory for CPU-only
"pin_memory": false
```

---

## 📁 **Directory Structure After Setup**

```
GitHubRepository/
├── 📋 README.md                           # ✅ Setup complete
├── 🚀 demo.py                             # ✅ Ready to run
├── ⚙️ requirements.txt                     # ✅ Installed
│
├── 📁 src/                                # ✅ Source code
│   ├── __init__.py
│   ├── model.py                           # ✅ Model definition
│   ├── dataset.py                        # ✅ Data handling
│   ├── train.py                          # ⚠️ Need to copy
│   └── evaluate.py                       # ⚠️ Need to copy
│
├── 📁 configs/                            # ✅ Configuration
│   ├── model_config.json                 # ✅ Model settings
│   └── training_config.json              # ✅ Training settings
│
├── 📁 data/                              # ✅ Data files
│   ├── README.md                         # ✅ Data documentation
│   ├── food101_categories.json          # ✅ Category mapping
│   └── food101_cache/                   # 🔄 Auto-downloaded
│
├── 📁 docs/                              # ✅ Documentation
│   ├── METHODOLOGY.md                   # ✅ Technical details
│   └── SETUP.md                         # ✅ This file
│
└── 📁 scripts/                           # ✅ Training & evaluation
    ├── train_personal_model.py          # ✅ Model training
    ├── prepare_personal_data.py         # ✅ Data preparation
    └── evaluate_model.py               # ✅ Model evaluation
```

---

## ✅ **Verification Checklist**

Run these commands to verify your setup:

```bash
# 1. Python environment
python --version
# Should output: Python 3.12.x

# 2. Core dependencies
python -c "import torch, torchvision, timm, datasets, pandas, sklearn"
echo "✅ Core imports successful"

# 3. GPU detection (if applicable)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 4. Model creation test
python -c "
from src.model import create_model
config = {'num_classes': 3, 'backbone': 'tf_efficientnet_b3'}
model = create_model(config)
print('✅ Model creation successful')
"

# 5. Demo functionality
python demo.py --help
# Should show help message without errors
```

---

## 🚀 **Next Steps**

After setup completion:

1. **Test Demo**: `python demo.py --interactive`
2. **Explore Code**: Check `src/` directory for implementation details
3. **Review Results**: Look at `docs/METHODOLOGY.md` for technical approach
4. **Train Model**: Use `scripts/train_personal_model.py` for custom training
5. **Customize**: Modify `configs/` files for different experiments

---

## 📞 **Support**

**Issues with Setup:**
- Check troubleshooting section above
- Verify Python version (3.12+ required)
- Ensure adequate disk space (10GB+)
- Try CPU-only installation if GPU issues persist

**For Technical Questions:**
- Review `docs/METHODOLOGY.md` for technical details
- Check `configs/` for parameter explanations
- See main `README.md` for usage examples

---

*Complete setup guide for Personal Food Taste Classification*
