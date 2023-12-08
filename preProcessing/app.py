# preprocess_input2_microservice.py
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Function to preprocess the input
@app.route('/preprocess_input2', methods=['POST','GET'])
def preprocess_input2():
    image_data = request.json['image']
    img = np.array(image_data)

    # Perform preprocessing
    img /= 127.5
    img -= 1.

    result = {'preprocessed_data': img.tolist()}
    return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True, host='127.0.0.1', port=5001)
