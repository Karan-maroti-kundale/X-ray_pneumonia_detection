from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import io
import os
import matplotlib
import requests

# =====================
# Suppress Warnings
# =====================
tf.get_logger().setLevel('ERROR')
matplotlib.use('Agg')  # non-GUI backend

# =====================
# Flask App
# =====================
app = Flask(__name__)

# =====================
# Model Setup
# =====================
MODEL_FOLDER = "models"
MODEL_FILENAME = "medical_ai_model.keras"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILENAME)

# Dropbox direct download link (dl=1)
MODEL_URL = "https://www.dropbox.com/scl/fi/tztitbkvl5qvftf8etevd/medical_ai_model.keras?rlkey=wimdr9c6g4y298k49icpqv7p9&dl=1"

# Create models folder if it doesn't exist
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Dropbox...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Model downloaded successfully.")

# Load the model
model = load_model(MODEL_PATH)
model.make_predict_function()

# =====================
# Image Preprocessing
# =====================
def prepare_image(file):
    img = image.load_img(io.BytesIO(file.read()), target_size=(254, 254))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =====================
# Home Page Route
# =====================
@app.route('/')
def index():
    return render_template("index.html")  # Serve your frontend HTML

# =====================
# Prediction API
# =====================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file.stream.seek(0)
    img_array = prepare_image(file)
    prediction = model.predict(img_array)[0][0]
    result = "Pneumonia" if prediction >= 0.5 else "Normal"

    return jsonify({
        'message': f'The uploaded X-ray indicates: {result}',
        'prediction': result,
        'score': float(prediction)
    })

# =====================
# Run Server
# =====================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))  # Use Render's PORT if available
    app.run(host='0.0.0.0', port=port, debug=False)

