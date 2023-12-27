from flask import Flask, request, jsonify
import requests
from PIL import Image
import numpy as np
from keras.models import load_model
import os
import boto3
from botocore.exceptions import NoCredentialsError
from datetime import datetime

from credentials import aws_access_key_id, aws_secret_access_key, bucket_name



def upload_to_s3(local_file, bucket_name, s3_key, aws_access_key_id, aws_secret_access_key):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    try:
        s3.upload_file(local_file, bucket_name, s3_key)
        print(f"File {local_file} uploaded to {bucket_name}/{s3_key}")
    except FileNotFoundError:
        print(f"The file {local_file} was not found")
    except NoCredentialsError:
        print("Credentials not available")




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

    local_file = 'temprory.jpg'
    s3_key = f'{datetime.now()}.jpg'

    # Upload the file to S3
    upload_to_s3(local_file, bucket_name, s3_key, aws_access_key_id, aws_secret_access_key)

    
    # Pass the image to preprocess_input microservice
    with open('temprory.jpg', 'rb') as f:
        preprocess_input_url = 'http://preprocess-service:5001/preprocess_input'
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