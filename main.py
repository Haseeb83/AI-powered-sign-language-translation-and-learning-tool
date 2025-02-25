from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORS for cross-origin requests

# Load the trained model
model = tf.keras.models.load_model("sign_language_model.keras")

# Function to generate 3D avatar animation data from model prediction
def generate_animation_from_model(prediction):
    frames = []
    for frame in prediction:
        keypoints = frame.tolist()  # Convert numpy array to list
        frames.append({'keypoints': keypoints})  # Store frame data
    return frames  # Return list of animation frames

# Sign Language to Text API
@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        if 'frame' not in data:
            return jsonify({'error': 'No frame provided'}), 400

        frame_data = np.array(data['frame'], dtype=np.uint8)
        
        if frame_data.size == 0:
            return jsonify({'error': 'Empty frame received'}), 400

        if frame_data.ndim != 3:
            return jsonify({'error': f'Invalid frame shape: {frame_data.shape}'}), 400

        frame = cv2.resize(frame_data, (28, 28))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frame = frame.reshape(1, 28, 28, 1) / 255.0

        prediction = model.predict(frame)
        predicted_class = np.argmax(prediction)
        predicted_label = chr(ord('A') + predicted_class)

        animation_data = generate_animation_from_model(prediction)

        return jsonify({
            'translation': predicted_label,
            'animation': animation_data
        })
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

# Text to Sign Language API
@app.route('/animate', methods=['POST'])
def animate():
    try:
        data = request.json
        text = data.get('text', '')

        text_input = np.array([ord(c) - ord('A') for c in text])  # Convert letters to numbers
        text_input = text_input.reshape(1, -1)

        prediction = model.predict(text_input)
        animation_data = generate_animation_from_model(prediction)

        return jsonify({'animation': animation_data, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

# Learning Feedback API
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        user_sign = np.array(data.get('user_sign', []), dtype=np.uint8)
        target_sign = data.get('target_sign', 'A')

        if user_sign.size == 0:
            return jsonify({'error': 'No sign provided'}), 400

        if user_sign.ndim != 3:
            return jsonify({'error': f'Invalid sign shape: {user_sign.shape}'}), 400

        user_sign = cv2.resize(user_sign, (28, 28))
        user_sign = cv2.cvtColor(user_sign, cv2.COLOR_BGR2GRAY)
        user_sign = user_sign.reshape(1, 28, 28, 1) / 255.0

        prediction = model.predict(user_sign)
        predicted_class = np.argmax(prediction)
        predicted_label = chr(ord('A') + predicted_class)

        feedback_msg = "Correct! Well done!" if predicted_label == target_sign else f"Incorrect. The correct sign is {target_sign}."

        return jsonify({'feedback': feedback_msg, 'predicted': predicted_label, 'target': target_sign})
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
