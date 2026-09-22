# Results

This directory contains compact result summaries extracted from the original LeafNet experiments.

## End-to-end model comparison

`model_comparison.csv` summarizes the tracked CNN and Transformer test results.

The strongest tracked configuration is DenseNet201:

- Test accuracy: **95.70%**
- Weighted F1-score: **95.55%**

## Deep-feature experiments

The `deep_features/` directory contains the original small CSV summaries for PCA/raw deep-feature experiments.

Large generated feature matrices are stored separately under `artifacts/feature_matrices/` so that result summaries are not mixed with intermediate data products.
