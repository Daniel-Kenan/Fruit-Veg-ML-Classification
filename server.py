from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Configure CORS to allow all origins, including all IPs
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the model
model = load_model('FV.h5')

# Labels dictionary
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

# Function to process image
def processed_img(img):
    img = img.resize((224, 224))  # Resize the image to the size expected by the model
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    y_class = prediction.argmax(axis=-1)
    result = labels[int(y_class)]
    return result.capitalize()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        img = Image.open(file.stream)
        result = processed_img(img)

        category = 'Vegetable' if result.lower() in labels.values() else 'Fruit'
        calories = "Unknown"  # Placeholder for actual calorie fetching

        return jsonify({
            "predicted": result,
            "category": category,
            "calories": calories
        })

if __name__ == '__main__':
    app.run(port=5000, debug=True,host='0.0.0.0')
