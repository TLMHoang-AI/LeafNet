from flask import Flask, request, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import os
import json
import pymysql

# ====== Cấu hình ======
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DB_CONFIG = {
    "charset": "utf8mb4",
    "connect_timeout": 10,
    "cursorclass": pymysql.cursors.DictCursor,
    "db": "defaultdb",
    "host": "leafnet-2025-minhhoangtran041105-00c6.c.aivencloud.com",
    "password": "UR PASS",
    "read_timeout": 10,
    "port": 15164,
    "user": "avnadmin",
    "write_timeout": 10,
}


# ====== Load class ======
data_dir = r"D:\Workplace\.SPRING2025\res\Final_segmentation"
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Grape Brown_Spot', 'Grape Downy_Mildew', 'Grape Mites_Disease', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Mango Gall Midge', 'Mango Powdery Mildew', 'Mango Sooty Mould', 'Peach___Bacterial_spot', 'Peach___healthy']
num_classes = 18

# ====== Load mô hình DenseNet201 đã huấn luyện ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet201(pretrained=False)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
model.load_state_dict(torch.load("BestDenseNet.pth", map_location=device))
model = model.to(device)
model.eval()

# ====== Transform ảnh ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== Dự đoán ảnh ======
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

def title_case(text):
    return " ".join([w.capitalize() for w in text.split()]) if isinstance(text, str) else text

# ====== Route chính ======
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            class_pred = predict_image(filepath)
            crop = disease = description = "Không rõ"

            connection = None
            try:
                connection = pymysql.connect(**DB_CONFIG)
                with connection.cursor() as cursor:
                    sql = "SELECT crop, disease, description FROM LEAFNET_des WHERE class = %s"
                    cursor.execute(sql, (class_pred,))
                    result = cursor.fetchone()

                    if result:
                        crop = title_case(result.get("crop"))
                        disease = title_case(result.get("disease"))
                        description = result.get("description", "").capitalize()

            except pymysql.Error as e:
                print(f"Lỗi truy vấn cơ sở dữ liệu: {e}")
            
            finally:
                if connection:
                    connection.close()

            image_url = f"/{filepath.replace(os.sep, '/')}"

            print(f"File path: {filepath}")
            print(f"Image URL: {image_url}")

            return render_template(
                "index.html",
                crop=crop,
                disease=disease,
                description=description,
                image_path=image_url
            )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)