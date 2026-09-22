# Experiment Map

The notebooks are grouped by the role they played in the original LeafNet study.

## 00 — Leaf segmentation

- `segmentation.ipynb`: segmentation/preprocessing experiments.

## 01 — Handcrafted ML

- `feature_extraction.ipynb`: handcrafted feature extraction.
- `classical_models.ipynb`: classical model comparison.

## 02 — Deep features

Pretrained CNN representations combined with classical machine-learning classifiers.

- `feature_extraction_overview.ipynb`
- `densenet201/`
- `resnet50/`
- `efficientnet_b0/`
- `vgg19/`

Associated compact CSV summaries are stored under `results/deep_features/`.

## 03 — End-to-end models

### CNN
- DenseNet201
- ResNet50
- ResNet101
- ResNet50V2
- EfficientNet-B0

### Transformers
- ViT
- Swin Transformer

The notebooks are retained as experimental records. Some contain paths and environment assumptions from the original competition workflow and are not presented as a one-command reproducibility package.
