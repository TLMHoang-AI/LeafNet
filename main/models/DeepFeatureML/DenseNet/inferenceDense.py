import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import DenseNet201_Weights
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import pickle

# Load ảnh và transform
image_path = r"final_data/segmented_images(not_txt)/Apple___Apple_scab/1f6abf22-93fa-48f0-a509-cc3e210f75f0___FREC_Scab 3172_segmented.png"
img = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0)  # Shape: [1, 3, 224, 224]

# Load model với weight mới
weights = DenseNet201_Weights.DEFAULT
model = models.densenet201(weights=weights)
model.eval()

# Lấy output từ model.features + avg pooling
with torch.no_grad():
    features = model.features(img_tensor)
    pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
    gap_vector = pooled.view(pooled.size(0), -1).numpy()  # Shape: [1, 1920]

base_dir = r"C:\Users\Admin\Documents\Python Project\Res conn 2025\densenet\saved_modelsDeepFT"

with open(os.path.join(base_dir, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(base_dir, "pca.pkl"), "rb") as f:
    pca = pickle.load(f)
with open(os.path.join(base_dir, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

X_scaled = scaler.transform(gap_vector)
X_pca = pca.transform(X_scaled)

with open(os.path.join(base_dir, "Logistic_Regression_pca_DenseNet201.pkl"), "rb") as f:
    model = pickle.load(f)

# Dự đoán
y_pred = model.predict(X_pca)
label = le.inverse_transform(y_pred)

print(f"Dự đoán: {label[0]}")
