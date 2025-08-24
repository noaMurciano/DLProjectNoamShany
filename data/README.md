# Data Directory

This directory contains data-related files and configurations for the personal food taste classification project.

## 📁 **Contents**

### **Configuration Files**
- `food101_categories.json` - Mapping of Food101 category IDs to names (101 categories)
- `sample_results.csv` - Example model predictions and confidence scores

### **Data Sources**

#### **Base Dataset: Food101**
- **Source**: [Food101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- **Size**: 101,000 images across 101 food categories
- **Access**: Via HuggingFace Datasets (`datasets.load_dataset("food101")`)
- **Usage**: Pre-training base model, source images for personal labeling

#### **Personal Dataset**
- **Size**: 200 personally labeled images from Food101
- **Labels**: disgusting (37), neutral (84), tasty (78)
- **Format**: Excel file with image paths and personal taste labels
- **Splits**: 60/20/20 (train/val/test) vs 80/20 comparison

## 🔄 **Data Pipeline**

```
Food101 Dataset (101K images)
        ↓
Personal Image Selection (200 images)
        ↓
Manual Labeling (disgusting/neutral/tasty)
        ↓
HuggingFace Dataset Integration
        ↓
Stratified Train/Val/Test Splits
        ↓
PyTorch DataLoader with Transforms
```

## 📊 **Dataset Statistics**

### **Personal Dataset Distribution**
```
Class Distribution:
├── Disgusting: 37 images (18.6%)
├── Neutral:    84 images (42.2%)
└── Tasty:      78 images (39.2%)

Split Distribution (60/20/20):
├── Train:      119 images
├── Validation:  40 images
└── Test:        40 images
```

### **Food Categories Coverage**
- **Total Categories**: 95 out of 101 Food101 categories
- **Most Frequent**: hot_dog (6), chicken_curry (5), mussels (5)
- **Coverage**: Diverse across cuisines and preparation methods

## 🔧 **Data Preprocessing**

### **Image Transformations**

**Training:**
```python
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, 
                          saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

**Validation/Test:**
```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

## 📁 **File Descriptions**

### **food101_categories.json**
```json
{
  "0": "apple_pie",
  "1": "baby_back_ribs",
  "2": "baklava",
  ...
  "100": "waffles"
}
```

### **sample_results.csv**
Example model predictions with confidence scores:
```csv
image_id,true_label,predicted_label,confidence,prob_disgusting,prob_neutral,prob_tasty
pizza_001,neutral,neutral,0.73,0.12,0.73,0.15
sashimi_002,tasty,tasty,0.85,0.05,0.10,0.85
```

## 🚀 **Usage**

### **Loading Personal Dataset**
```python
from src.dataset import PersonalTasteDataset

# Create dataset
dataset = PersonalTasteDataset(
    csv_path="data/personal_train.csv",
    transform=train_transforms
)

# Access sample
image, label, metadata, info = dataset[0]
```

### **Using Food101 Categories**
```python
import json

with open("data/food101_categories.json", "r") as f:
    categories = json.load(f)

# Map category ID to name
category_name = categories[str(category_id)]
```

## ⚠️ **Important Notes**

### **Data Privacy**
- Personal labels reflect individual taste preferences
- Not representative of general population preferences
- Intended for academic research purposes only

### **Reproducibility**
- All random operations use fixed seed (42)
- Stratified splits maintain class distribution
- Consistent preprocessing across experiments

### **Limitations**
- Small dataset size (199 samples) limits generalization
- Class imbalance (neutral class dominant)
- Personal subjectivity in labeling
- Limited context information (no preparation quality, freshness, etc.)

## 📋 **Data Requirements**

To reproduce this project with your own data:

1. **Personal Labeling**: Label 200+ food images with taste preferences
2. **Format**: Excel/CSV with columns: image_path, label
3. **Labels**: Use "disgusting", "neutral", "tasty" format
4. **Balance**: Try to have reasonable distribution across classes
5. **Quality**: Ensure consistent labeling criteria

## 📚 **References**

- **Food101 Dataset**: Bossard, L., et al. (2014). Food-101 – Mining Discriminative Components with Random Forests. ECCV.
- **HuggingFace Datasets**: Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. EMNLP.

---

*For questions about data usage or format, see the main README.md or technical documentation.*
