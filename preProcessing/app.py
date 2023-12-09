# preprocess_input2_microservice.py
from flask import Flask, request, jsonify
import numpy as np
import keras.utils as image
from PIL import Image



app = Flask(__name__)

# Function to preprocess the input
@app.route('/preprocess_input', methods=['POST'])
def preprocess_input():

    # Get the image from the request
    uploaded_image_bytes = request.files.get('image')
    # Process the image as needed (example: open the image using PIL)
    img = Image.open(uploaded_image_bytes)
    Load_IMG = img.resize((224,224))
    Array_IMG = image.img_to_array(Load_IMG)
    Array_IMG = Array_IMG.reshape((1,) + Array_IMG.shape)
    Array_IMG /= 127.5
    Array_IMG -= 1.


    result = {'preprocessed_data': Array_IMG.tolist()}
    return jsonify(result)

