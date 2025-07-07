#  Leaf Disease Detection and Classification
A deep learning-powered solution for early, accurate plant disease diagnostics 
#  Overview
Timely and accurate identification of leaf diseases plays a crucial role in maintaining crop health, increasing yields, and ensuring food security. This repository presents a comprehensive AI-driven framework for automated detection and classification of foliar diseases, developed as part of a competition entry for ResConnect.

Our approach spans both traditional machine learning pipelines and state-of-the-art deep learning techniques, targeting high accuracy, computational efficiency, and real-world applicability in resource-constrained agricultural environments.

#  Key Features
 Classical ML models: Logistic Regression, SVM, KNN, Random Forest, LightGBM

 Traditional vision pipeline with handcrafted features: up to 87% accuracy

 Hybrid deep feature + classical classifier models: up to 91% accuracy

 End-to-end deep learning with CNNs and Transformers: up to 95% accuracy

 Architectures explored: DenseNet, ResNet, EfficientNet, ViT, Swin Transformer

 Scalable & deployable framework for real-world use

#  Methodology
The project is divided into three stages:

1. Baseline Traditional Models
Extract handcrafted features from leaf images.

Train classical ML models for baseline performance.

2. Hybrid Feature-Based Models
Use pre-trained CNNs to extract deep features.

Feed them into traditional classifiers (e.g., SVM, RF, LightGBM).

3. End-to-End Deep Learning
Train CNNs and vision transformers directly on raw images.

Optimize architectures for high accuracy and efficiency.

#  Applications
Early disease detection and precision agriculture

Smart crop surveillance

Scalable AI solutions for food security

On-device deployment for edge-based farm monitoring
