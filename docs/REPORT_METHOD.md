# Personal Food Taste Classification using Deep Learning
## Method Section

**Course**: 046211 Deep Learning, Technion  
**Author**: Noam Shani  
**Semester**: Spring 2025

---

## 2. Method

This section presents our technical approach for personal food taste classification, including the model architecture, training strategy, loss function design, and experimental methodology. Our approach builds upon established transfer learning principles while introducing novel adaptations for the personal preference learning scenario.

### 2.1 Problem Formulation

We formulate personal food taste prediction as a supervised learning problem. Given a food image $x \in \mathbb{R}^{3 \times H \times W}$, our goal is to predict the personal taste preference $y \in \{0, 1, 2\}$, where:

- $y = 0$: **Disgusting** - food the individual would avoid
- $y = 1$: **Neutral** - food with no strong preference  
- $y = 2$: **Tasty** - food the individual would enjoy

The challenge lies in learning a mapping $f: \mathbb{R}^{3 \times H \times W} \rightarrow \{0, 1, 2\}$ from a small dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ where $N = 199$ represents the limited personal labeled data.

### 2.2 Model Architecture

#### 2.2.1 Base Architecture Selection

Our model builds upon EfficientNet-B3 [1], selected for its proven effectiveness on food image classification tasks and optimal balance between model complexity and performance. EfficientNet architectures use compound scaling to systematically scale network depth, width, and resolution, making them particularly suitable for transfer learning scenarios.

The base EfficientNet-B3 model provides:
- **Input resolution**: $224 \times 224$ pixels
- **Parameters**: Approximately 11M parameters
- **Pre-training**: ImageNet + Food101 sequential training
- **Feature extraction**: 1536-dimensional global average pooled features

#### 2.2.2 Custom Classification Head

We replace the original Food101 classification head with a specialized architecture for personal taste prediction:

```
Global Average Pooling (1536 → 1536)
    ↓
Linear Layer (1536 → 256)
    ↓
GELU Activation
    ↓
Dropout (p = 0.3)
    ↓
Linear Layer (256 → 3)
```

The classification head design incorporates several important considerations:

1. **Hidden Dimension**: 256 units provide sufficient capacity for personal preference modeling without overfitting
2. **Activation Function**: GELU (Gaussian Error Linear Unit) offers smoother gradients compared to ReLU
3. **Regularization**: Dropout with probability 0.3 prevents overfitting on small datasets
4. **Output Layer**: 3 units corresponding to our taste preference classes

#### 2.2.3 Mathematical Formulation

Let $\phi(\cdot; \theta_b)$ represent the EfficientNet-B3 backbone with parameters $\theta_b$, and $h(\cdot; \theta_h)$ represent the classification head with parameters $\theta_h$. Our model computes:

$$z = \phi(x; \theta_b)$$
$$\hat{y} = h(z; \theta_h) = W_2 \cdot \text{GELU}(W_1 z + b_1) + b_2$$

where $W_1 \in \mathbb{R}^{256 \times 1536}$, $W_2 \in \mathbb{R}^{3 \times 256}$, and $b_1, b_2$ are bias terms.

The final prediction is obtained via softmax normalization:
$$p(y = k | x) = \frac{\exp(\hat{y}_k)}{\sum_{j=0}^{2} \exp(\hat{y}_j)}$$

### 2.3 Transfer Learning Strategy

#### 2.3.1 Two-Phase Curriculum Learning

We implement a two-phase training strategy designed to balance knowledge preservation from the pre-trained model with adaptation to personal preferences:

**Phase 1: Frozen Backbone Training (Epochs 1-8)**
- Freeze all backbone parameters: $\theta_b$ remains fixed
- Train only classification head: optimize $\theta_h$
- Higher learning rate for new parameters
- Focus on learning personal preference patterns

**Phase 2: End-to-End Fine-tuning (Epochs 9-15)**
- Unfreeze backbone parameters: optimize both $\theta_b$ and $\theta_h$
- Very low learning rate for backbone preservation
- Discriminative learning rates prevent catastrophic forgetting
- Full model adaptation to personal preferences

#### 2.3.2 Discriminative Learning Rates

To prevent catastrophic forgetting while enabling adaptation, we employ discriminative learning rates:

- **Classification Head**: $\eta_h = 1 \times 10^{-3}$ (higher for new layers)
- **Backbone**: $\eta_b = 1 \times 10^{-5}$ (lower to preserve features)

This 100:1 ratio ensures that the pre-trained features evolve slowly while the classification head adapts quickly to personal preferences.

#### 2.3.3 Optimization Configuration

**Optimizer**: AdamW with weight decay $\lambda = 0.01$
**Scheduler**: Cosine annealing with $T_{max} = 15$, $\eta_{min} = 1 \times 10^{-7}$
**Batch Size**: 16 (optimal for small dataset scenarios)

### 2.4 Loss Function Design

#### 2.4.1 Primary Loss Function

We use a weighted cross-entropy loss to handle class imbalance in personal preference data:

$$\mathcal{L}_{CE} = -\sum_{i=1}^{N} w_{y_i} \log p(y_i | x_i)$$

where $w_c$ represents the class weight for class $c$, computed using inverse frequency weighting:

$$w_c = \frac{N}{3 \cdot N_c}$$

where $N_c$ is the number of samples in class $c$.

#### 2.4.2 Label Smoothing Regularization

To prevent overconfident predictions on limited data, we apply label smoothing with parameter $\alpha = 0.05$:

$$\tilde{y}_c = (1 - \alpha) \cdot y_c + \frac{\alpha}{K}$$

where $K = 3$ is the number of classes, and $y_c$ is the one-hot encoded true label.

#### 2.4.3 Complete Loss Formulation

The final loss combines cross-entropy with label smoothing:

$$\mathcal{L} = -\sum_{i=1}^{N} w_{y_i} \sum_{c=0}^{2} \tilde{y}_{i,c} \log p(y_i = c | x_i)$$

This formulation addresses both class imbalance and overfitting concerns inherent in small dataset scenarios.

### 2.5 Data Preprocessing and Augmentation

#### 2.5.1 Image Preprocessing Pipeline

All images undergo standardized preprocessing:

1. **Resize**: $256 \times 256$ pixels maintaining aspect ratio
2. **Center Crop**: $224 \times 224$ pixels for model input
3. **Normalization**: ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)
4. **Format**: RGB tensor with values in $[0, 1]$

#### 2.5.2 Data Augmentation Strategy

For training data, we apply conservative augmentation to prevent overfitting:

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

The augmentation parameters are deliberately conservative to avoid introducing artifacts that might confuse personal preference learning.

### 2.6 Handling Class Imbalance

#### 2.6.1 Weighted Random Sampling

During training, we employ weighted random sampling to balance class representation:

$$p(\text{sample}_i) = \frac{w_{y_i}}{\sum_{j=1}^{N} w_{y_j}}$$

This ensures each class receives equal attention during training despite uneven distribution in the dataset.

#### 2.6.2 Class Weight Analysis

For our personal dataset with distribution:
- Disgusting: 37 samples (18.6%)
- Neutral: 84 samples (42.2%)
- Tasty: 78 samples (39.2%)

The computed class weights are:
- $w_0 = 1.79$ (disgusting)
- $w_1 = 0.79$ (neutral)  
- $w_2 = 0.85$ (tasty)

### 2.7 Training Methodology

#### 2.7.1 Data Splitting Strategy Comparison

We evaluate two splitting strategies to understand their impact on small dataset performance:

**Strategy A: 80/20 Split**
- Training: 159 samples (79.9%)
- Testing: 40 samples (20.1%)
- No separate validation set

**Strategy B: 60/20/20 Split**
- Training: 119 samples (59.8%)
- Validation: 40 samples (20.1%)
- Testing: 40 samples (20.1%)

#### 2.7.2 Stratified Sampling

All splits maintain class distribution proportions using stratified sampling with random seed 42 for reproducibility. This ensures each split contains representative samples from all preference classes.

#### 2.7.3 Early Stopping Mechanism

For the 60/20/20 split, we implement early stopping based on validation macro F1-score:

- **Patience**: 5 epochs without improvement
- **Metric**: Macro F1-score (handles class imbalance better than accuracy)
- **Best Model**: Saved based on highest validation F1-score

### 2.8 Evaluation Methodology

#### 2.8.1 Performance Metrics

We evaluate model performance using multiple metrics to provide comprehensive assessment:

1. **Accuracy**: Overall correct prediction rate
2. **Macro F1-Score**: Unweighted average of per-class F1-scores
3. **Per-Class F1-Score**: Individual class performance analysis
4. **Confusion Matrix**: Detailed error pattern analysis
5. **Confidence Analysis**: Prediction certainty evaluation

#### 2.8.2 Statistical Significance

All experiments use fixed random seeds (seed=42) for reproducibility. We report mean performance across multiple runs where applicable and provide confidence intervals for key metrics.

#### 2.8.3 Baseline Comparisons

We compare against several baselines:
- **Random Classifier**: Uniform random predictions
- **Majority Class**: Always predict most frequent class
- **Pre-trained Food101**: Original model without personal adaptation

### 2.9 Implementation Details

#### 2.9.1 Software Framework

- **Deep Learning**: PyTorch 2.0+ for model implementation
- **Pre-trained Models**: timm library for EfficientNet-B3
- **Data Loading**: HuggingFace Datasets for Food101 access
- **Evaluation**: scikit-learn for metrics computation

#### 2.9.2 Hardware Configuration

- **GPU**: CUDA-compatible device (when available)
- **Memory**: Minimum 8GB RAM for data loading
- **Storage**: 10GB for dataset caching and model checkpoints

#### 2.9.3 Hyperparameter Selection

All hyperparameters are selected based on:
1. Literature best practices for transfer learning
2. Preliminary experiments on validation data
3. Computational constraints for practical deployment

Key hyperparameters and their justification:
- **Batch Size (16)**: Balance between gradient stability and memory constraints
- **Learning Rates**: 100:1 ratio based on transfer learning best practices
- **Dropout (0.3)**: Standard regularization for small datasets
- **Weight Decay (0.01)**: Moderate regularization to prevent overfitting

### 2.10 Reproducibility Considerations

To ensure reproducibility, we implement:

1. **Fixed Random Seeds**: All random operations use seed=42
2. **Deterministic Operations**: Consistent behavior across runs
3. **Version Control**: Exact library versions specified
4. **Configuration Management**: All hyperparameters stored in JSON files
5. **Environment Documentation**: Complete setup instructions provided

This methodology provides a robust framework for personal food taste classification while addressing the unique challenges of small dataset learning and preference personalization.

---

## References

[1] Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *International Conference on Machine Learning (ICML)*.

---

*This method section provides comprehensive technical details of our approach, enabling reproduction and extension of the work while clearly explaining the rationale behind each design choice.*
