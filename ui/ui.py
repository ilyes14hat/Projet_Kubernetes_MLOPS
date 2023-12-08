import streamlit as st
from PIL import Image
import requests
import time
# Streamlit UI function
def main():
    st.title("Blood Cells Classification")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # time.sleep(30)
    # print('ffffffffffffffffff',uploaded_file.read())

    if uploaded_file is not None:
        # print('not none = ',uploaded_file)
        image = Image.open(uploaded_file)
        
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Button to trigger classification
        if st.button('Classify'):
            # Convert image to bytes
            # print('not none = ',uploaded_file)
            img_bytes = uploaded_file.getvalue()
            #print("from ui =================", img_bytes[0:30])

            # Send the image to the predict microservice
            #predict_microservice_url = 'http://predict-microservice:5000/predict'
            predict_microservice_url = 'http://127.0.0.1:5000/predict'
            files = {'image': img_bytes}
            # print(files)
            response = requests.post(predict_microservice_url, files=files)

            # Display the prediction
            if response.status_code == 200:
                result = response.json()
                st.write(f"Prediction: {result['prediction']}")
            else:
                st.write("Error in prediction")

# Run the Streamlit app
if __name__ == '__main__':
    main()
