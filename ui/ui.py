import streamlit as st
from PIL import Image
import requests


# Streamlit UI function
def main():
    st.title("Blood Cells Classification")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
 
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Button to trigger classification
        if st.button('Classify'):
            # Convert image to bytes
            img_bytes = uploaded_file.getvalue()

            # Send the image to the predict microservice
            predict_microservice_url = 'http://predict-service:5000/predict'
            files = {'image': img_bytes}
            # print(img_bytes)
            response = requests.post(predict_microservice_url, files=files)
            # print(response)

            # Display the prediction
            if response.status_code == 200:
                result = response.json()
                st.write(f"Prediction: {result['prediction']}")
            else:
                st.write("Error in prediction")

# Run the Streamlit app
if __name__ == '__main__':
    main()
