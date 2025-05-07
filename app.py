from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import io
import base64
import cv2
import numpy as np

app = Flask(__name__)

# ---- Correct CNN model (3 conv layers, trained on 3-channel RGB) ----
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 28 * 28, 512)  # ✅ must match training
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---- Load model weights (make sure .pth file is in the same folder) ----
model = EmotionCNN()
model.load_state_dict(torch.load("CNN_emotion_model .pth", map_location=torch.device("cpu")))
model.eval()

# ---- Class labels (must match your training dataset order) ----
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ---- Preprocessing (RGB, 224x224, to tensor) ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/')
def index():
    # return render_template('index.html')  # You must have templates/index.html
    return render_template('pic.html')  # You must have templates/index.html

# ---- Load OpenCV face detector ----
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        img_data = base64.b64decode(data)
        image = Image.open(io.BytesIO(img_data)).convert('RGB')  # Convert to RGB

        # Convert to NumPy and detect faces
        image_np = np.array(image.convert('L'))  # grayscale for face detection
        faces = face_cascade.detectMultiScale(image_np, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return jsonify({'emotion': 'no face detected'})

        # Use first detected face
        x, y, w, h = faces[0]
        face_img = image.crop((x, y, x + w, y + h))  # crop face in RGB

        # Preprocess for model
        face_tensor = transform(face_img).unsqueeze(0)

        with torch.no_grad():
            output = model(face_tensor)
            _, predicted = torch.max(output, 1)
            emotion = class_names[predicted.item()]
            print("Predicted Emotion:", emotion)

        return jsonify({'emotion': emotion})
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
