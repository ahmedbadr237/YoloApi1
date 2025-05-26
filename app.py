from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO
import io
import yaml

# Load class names from your YAML file
with open("data (1).yaml", "r") as f:
    class_data = yaml.safe_load(f)

# If names are a list
class_names = class_data["names"]

app = Flask(__name__)
model = YOLO("best.pt")  # Load your YOLOv8 classification model

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert('RGB')

    results = model(image)
    if results[0].boxes.cls.numel() == 0:
        return jsonify({'error': 'No objects detected'}), 400

    class_id = int(results[0].boxes.cls[0].item())
    class_name = class_names[class_id]

    return class_name



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
