from flask import Flask, request, jsonify
import requests
from PIL import Image
import numpy as np
from keras.models import load_model
import os



app = Flask(__name__)

# Load model
model = load_model('projetIMN.h5')


# Function to make a prediction using the model
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    uploaded_file = request.files['image']
    img = Image.open(uploaded_file)
    img.save('temprory.jpg')
    preprocessed_data= []
    
    # Pass the image to preprocess_input microservice
    with open('temprory.jpg', 'rb') as f:
        preprocess_input_url = 'http://127.0.0.1:5001/preprocess_input'
        r = requests.post(preprocess_input_url, files={'image': f})
        preprocessed_data = np.array(r.json()['preprocessed_data'])
    os.remove('temprory.jpg')

    prediction = model.predict(preprocessed_data)
    label = np.argmax(prediction)

    classes = {0: 'EOSINOPHIL', 1: 'LYMPHOCYTE', 2: 'MONOCYTE', 3: 'NEUTROPHIL'}
    result = {'prediction': classes[label]}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)