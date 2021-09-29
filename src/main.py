import numpy as np
import streamlit as st
from PIL import Image

from image import plot_image_with_bbs, crop_image
from object_detection import get_objects
from classifier import predict_cap


def app():

    st.title('Computer Vision Demo: Bottles')

    uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        _, center, _ = st.beta_columns([1, 1, 1])
        if center.button('Run detection'):
            bbs = get_objects(image, from_file=False)
            results = []
            for bb in bbs:
                crop = crop_image(image, bb)
                cap_conf = predict_cap(crop)
                results.append({'bb': bb, 'cap_conf': cap_conf})
            image = plot_image_with_bbs(image, results)

            st.image(np.array(image))


if __name__ == '__main__':
    app()
