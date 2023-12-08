# predict_microservice.py
from flask import Flask, request, jsonify
import requests
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model('projetIMN.h5')

# Function to make a prediction using the model
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    uploaded_file = request.files['image']
    print("fffffffffffffffffffffffffffffffffffffff",request.files)

    img = Image.open(uploaded_file)

    # # Pass the image to preprocess_input2 microservice
    # preprocess_input2_url = 'http://127.0.0.0:5001/preprocess_input2'
    # response = requests.post(preprocess_input2_url, json={'image': img.tolist()})
    # preprocessed_data = np.array(response.json()['preprocessed_data'])

    # # Make a prediction using the preprocessed data
    # Array_IMG = preprocessed_data.reshape((1,) + preprocessed_data.shape)
    # val_gen = ImageDataGenerator()
    # input_data = val_gen.flow(Array_IMG, batch_size=1)
    # prediction = model.predict(input_data)
    # label = np.argmax(prediction)

    # classes = {0: 'EOSINOPHIL', 1: 'LYMPHOCYTE', 2: 'MONOCYTE', 3: 'NEUTROPHIL'}
    # result = {'prediction': classes[label]}

    # return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True, host='127.0.0.1', port=5000)
