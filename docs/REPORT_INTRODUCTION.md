# Personal Food Taste Classification using Deep Learning
## Introduction Section

**Course**: 046211 Deep Learning, Technion  
**Author**: Noam Shani  
**Semester**: Spring 2025

---

## 1. Introduction

### 1.1 Project Goal and Motivation

The proliferation of food recommendation systems in modern applications—from restaurant discovery platforms to meal planning services—has highlighted a fundamental limitation in current approaches: the reliance on aggregated, crowd-sourced preferences that fail to capture individual taste nuances. While collaborative filtering and content-based recommendation systems have achieved success in domains such as movie recommendations and e-commerce, food preference modeling presents unique challenges due to the highly subjective, personal, and culturally influenced nature of taste preferences.

This project addresses the research question: **"Can modern deep learning techniques effectively learn and predict individual food taste preferences from limited personal labeled data?"** Our approach leverages transfer learning principles to adapt a general food classification model to personal taste preferences, demonstrating how large-scale pre-trained models can be personalized for individual users with minimal additional data.

The motivation for this work stems from the growing need for personalized AI systems that can adapt to individual preferences rather than relying solely on population-level patterns. In the context of food recommendation, this personalization could enable more accurate meal suggestions, dietary planning assistance, and enhanced user experiences in food-related applications.

### 1.2 Problem Statement

Traditional food recommendation systems face several key limitations:

1. **Aggregation Bias**: Population-level preferences may not reflect individual taste patterns
2. **Cold Start Problem**: New users require extensive interaction history before receiving accurate recommendations
3. **Subjective Labeling**: Food preference is inherently subjective and context-dependent
4. **Limited Personalization**: Current systems struggle to capture nuanced individual preferences

We formulate the personal food taste prediction problem as a three-class classification task, where given a food image, the system predicts whether an individual will find the food:
- **Disgusting** (Class 0): Food the individual would avoid
- **Neutral** (Class 1): Food the individual has no strong preference about
- **Tasty** (Class 2): Food the individual would enjoy

The challenge lies in learning these highly personalized patterns from limited labeled data (199 samples in our case) while avoiding overfitting and maintaining generalization capability.

### 1.3 Previous Work and Background

#### 1.3.1 Food Image Classification

Food image classification has been an active area of research since the introduction of the Food101 dataset by Bossard et al. (2014) [1], which provided 101,000 images across 101 food categories. This dataset established a benchmark for food recognition tasks and enabled the development of specialized architectures for food image understanding.

Recent advances in food image classification have leveraged convolutional neural networks (CNNs) and transfer learning approaches. EfficientNet architectures, introduced by Tan and Le (2019) [2], have shown particularly strong performance on food image datasets due to their balanced scaling of network depth, width, and resolution.

#### 1.3.2 Transfer Learning in Computer Vision

Transfer learning has become a cornerstone technique in computer vision, particularly when working with limited data. Yosinski et al. (2014) [3] demonstrated that features learned on large datasets like ImageNet transfer effectively to specialized domains, with early layers capturing universal visual features and later layers becoming increasingly task-specific.

The application of transfer learning to food domains has shown promising results, with pre-trained models on ImageNet serving as effective feature extractors for food-related tasks. However, most existing work focuses on general food classification rather than personalized preference prediction.

#### 1.3.3 Personalization in Recommendation Systems

Personalized recommendation systems have traditionally relied on collaborative filtering and matrix factorization techniques. More recently, deep learning approaches have enabled more sophisticated modeling of user preferences, particularly in domains with rich content features.

However, food preference personalization remains understudied compared to other recommendation domains. Most existing food recommendation systems rely on explicit ratings, reviews, or demographic information rather than learning directly from individual preference patterns on food images.

#### 1.3.4 Few-Shot and Small Data Learning

Learning from limited data is a fundamental challenge in machine learning, particularly relevant to our personal preference scenario where collecting large amounts of individual labeled data is impractical. Few-shot learning techniques, meta-learning approaches, and careful regularization strategies have been developed to address this challenge.

Curriculum learning, introduced by Bengio et al. (2009) [4], provides a framework for gradually increasing task difficulty during training, which can be particularly beneficial when working with small datasets and class imbalances.

### 1.4 Research Contributions

This work makes several key contributions to the intersection of computer vision, transfer learning, and personalized recommendation systems:

#### 1.4.1 Novel Problem Formulation
We introduce the problem of personal food taste classification as a transfer learning task, adapting general food recognition models to individual preference patterns. This formulation bridges computer vision and personalization research.

#### 1.4.2 Methodology Comparison
We provide a rigorous comparison of two data splitting strategies for small dataset scenarios:
- **80/20 Split**: Traditional train/test split with 159 training and 40 test samples
- **60/20/20 Split**: Proper validation methodology with 119 train, 40 validation, and 40 test samples

Our analysis demonstrates the importance of proper validation even when it reduces training data, particularly for small dataset scenarios.

#### 1.4.3 Transfer Learning Strategy
We implement and evaluate a two-phase curriculum learning approach:
- **Phase 1**: Frozen backbone training focusing on classification head adaptation
- **Phase 2**: End-to-end fine-tuning with discriminative learning rates

This strategy balances knowledge preservation from the pre-trained model with adaptation to personal preferences.

#### 1.4.4 Personal Preference Analysis
Through model interpretation and error analysis, we extract meaningful insights about individual food preferences, demonstrating how deep learning models can capture and explain personal taste patterns.

### 1.5 Technical Approach Overview

Our approach builds upon the EfficientNet-B3 architecture pre-trained on Food101, adapting it for personal taste classification through:

1. **Custom Classification Head**: A specialized 3-class classifier with appropriate regularization
2. **Class Imbalance Handling**: Weighted sampling and loss functions to address uneven label distribution
3. **Discriminative Learning Rates**: Different learning rates for backbone and classification head
4. **Early Stopping**: Validation-based stopping to prevent overfitting on limited data

The system is designed to be practical and deployable, with training times under one minute and model sizes suitable for real-world applications.

### 1.6 Experimental Design

Our experimental framework includes:

- **Baseline Comparison**: Evaluation against random and majority class baselines
- **Ablation Studies**: Analysis of different components' contributions
- **Validation Methodology**: Comparison of different data splitting strategies
- **Qualitative Analysis**: Interpretation of learned preferences and failure cases

All experiments are conducted with fixed random seeds to ensure reproducibility, and comprehensive metrics are reported including accuracy, macro F1-score, per-class performance, and confidence analysis.

### 1.7 Report Organization

The remainder of this report is organized as follows:

- **Section 2 (Method)**: Detailed description of our technical approach, including model architecture, training strategy, and experimental setup
- **Section 3 (Experiments and Results)**: Comprehensive evaluation of our approach, including quantitative results, ablation studies, and qualitative analysis
- **Section 4 (Conclusion and Future Work)**: Summary of findings, limitations, and directions for future research

---

## References

[1] Bossard, L., Guillaumin, M., & Van Gool, L. (2014). Food-101–mining discriminative components with random forests. *European Conference on Computer Vision (ECCV)*.

[2] Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *International Conference on Machine Learning (ICML)*.

[3] Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *Advances in Neural Information Processing Systems (NIPS)*.

[4] Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *International Conference on Machine Learning (ICML)*.

---

*This introduction section provides the foundation for a comprehensive academic report on personal food taste classification using deep learning techniques. The work demonstrates the application of modern transfer learning methods to personalized recommendation scenarios.*
