# Methodology

LeafNet was organized as a progressive comparison of representation-learning strategies for plant leaf disease classification.

## Stage 0 — Leaf segmentation

Segmentation and preprocessing experiments were used to prepare or isolate useful leaf regions before downstream classification.

## Stage 1 — Handcrafted features

The first stage extracts manually designed image features and evaluates classical machine-learning models. This provides a lower-complexity baseline before introducing learned representations.

Main model families include:

- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors
- Random Forest
- LightGBM

## Stage 2 — Deep features

Pretrained convolutional networks are used as fixed or partially reused feature extractors. The resulting feature vectors are evaluated with classical classifiers and dimensionality-reduction strategies such as PCA.

Backbones represented in the repository include:

- DenseNet201
- ResNet50
- EfficientNet-B0
- VGG19

## Stage 3 — End-to-end learning

CNN and Vision Transformer models are trained directly for disease classification.

The tracked experiments include:

- DenseNet201
- ResNet50
- ResNet101
- ResNet50V2
- EfficientNet-B0
- Vision Transformer (ViT)
- Swin Transformer

The strongest tracked end-to-end result is DenseNet201 with 95.70% test accuracy and 95.55% weighted F1-score.

## Interpretation

The experimental progression is useful because it separates the benefit of:

1. handcrafted visual descriptors,
2. pretrained learned representations used with classical classifiers, and
3. fully end-to-end representation learning.

This repository preserves the notebooks from each stage rather than presenting only the final model.
