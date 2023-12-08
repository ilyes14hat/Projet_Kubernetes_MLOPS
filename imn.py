import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import keras.utils as image
from keras.models import load_model


# load model
model = load_model('projetIMN.h5')

# Function to preprocess the uploaded image
# def preprocess_image(image):
#     img = image.resize((224, 224))  # Resize image to match model input size
#     img = np.array(img) / 255.0  # Normalize pixel values
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img
def preprocess_input2(x):
  x /= 127.5
  x -= 1.
  return x
# Function to make a prediction using the model
def predict(img):
    Load_IMG = img.resize((220,220))
    ######################################
    Array_IMG = image.img_to_array(Load_IMG)
    Array_IMG = Array_IMG.reshape((1,) + Array_IMG.shape)
    val_gen = ImageDataGenerator( preprocessing_function=preprocess_input2)
    input = val_gen.flow(Array_IMG,batch_size=1)
    prediction = model.predict(input)
    ######################################
    #Array_IMG = image.img_to_array(Load_IMG)
    #Array_IMG = Array_IMG.reshape((1,) + Array_IMG.shape)
    #Test_Generator = ImageDataGenerator(rescale=1./255)
    #input = Test_Generator.flow(Array_IMG,batch_size=1)
    prediction = model.predict(input)
    return prediction

# Create a Streamlit web application
def main():
    st.title("Blood Cells Classification")
    
    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Button to trigger classification
        if st.button('Classify'):
            # Perform prediction
            prediction = predict(image)
            label = np.argmax(prediction)
            # {'EOSINOPHIL': 0, 'LYMPHOCYTE': 1, 'MONOCYTE': 2, 'NEUTROPHIL': 3}

            if label==0:
                st.write("Prediction: EOSINOPHIL")
            elif label==1:
                st.write("Prediction: LYMPHOCYTE")
            elif label==2:
                st.write("Prediction: MONOCYTE")
            elif label==3:
                st.write("Prediction: NEUTROPHIL")



# Run the Streamlit app
if __name__ == '__main__':
    main()
