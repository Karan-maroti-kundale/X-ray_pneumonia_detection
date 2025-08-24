from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import io
import os
import matplotlib

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
# Load Model
# =====================
MODEL_PATH = os.path.join("models", "medical_ai_model.keras")  # model inside models/ folder
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
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT if available
    app.run(host='0.0.0.0', port=port, debug=False)
