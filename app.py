from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import io
import os
from huggingface_hub import hf_hub_download

# =====================
# Suppress Warnings
# =====================
tf.get_logger().setLevel('ERROR')

# =====================
# Flask App
# =====================
app = Flask(__name__)

# =====================
# Model Setup
# =====================
MODEL_FOLDER = "models"
MODEL_FILENAME = "best_model.keras"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILENAME)

# Hugging Face repo info
MODEL_REPO = "karankundale/model"
MODEL_FILE = "best_model.keras"

# Create models folder if it doesn't exist
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    MODEL_PATH = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        token=os.environ.get("HF_TOKEN"),  # Add HF_TOKEN env variable if private repo
        repo_type="model"
    )
    print("Model downloaded successfully.")

# Load the model (CPU only)
model = load_model(MODEL_PATH)
model.make_predict_function()

# =====================
# Image Preprocessing
# =====================
def prepare_image(file):
    img_size = (224, 224)  # Match your training size
    img = image.load_img(io.BytesIO(file.read()), target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =====================
# Home Page Route
# =====================
@app.route('/')
def index():
    return render_template("index.html")

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

    # Prediction
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
    port = int(os.environ.get("PORT", 4000))
    app.run(host='0.0.0.0', port=port, debug=False)
