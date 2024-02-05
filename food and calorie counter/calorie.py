import streamlit as st
from PIL import Image, ImageOps  # Install pillow instead of PIL
import requests
from bs4 import BeautifulSoup
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

model = load_model("keras_model_food.h5",compile=False)
labels = open("label.txt", "r").readlines()
print(labels)

def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + 'chicken'
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)

def run():
    st.title("Classification and Calorie Calculator")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        img = Image.open(img_file).convert(
            "RGB")
        st.image(img, use_column_width=False)
        size = (224, 224)
        image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = labels[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)
        st.success("**Predicted : " + class_name[2:] + '**')
        cal = fetch_calories(class_name[2:])
        if cal:
            st.warning('**' + cal + '(100 grams)**')


run()