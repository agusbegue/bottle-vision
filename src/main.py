import numpy as np
import streamlit as st
from PIL import Image

from image import plot_image_with_bbs, crop_image
from object_detection import get_objects
from classifier import predict_cap, ApplicationLayer

from keras.models import load_model
from settings import TF_CLF_MODEL_PATH as MODEL_PATH

FULL_PIPELINE = 'Object detection + Classification'
JUST_CLASSIFICATION = 'Just Classification'

from classifier import get_classifier_model
def app():
    # model = load_model('../models/classifier/classifier.h5', custom_objects={'ApplicationLayer': ApplicationLayer})
    model = get_classifier_model()
    st.title('Object detection demo: bottles')

    pipeline = st.radio('Select pipeline for image', [FULL_PIPELINE, JUST_CLASSIFICATION])
    if pipeline == FULL_PIPELINE:
        st.header('Running full pipeline on image')
    else:
        st.header('Running just classification on image')

    uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        _, center, _ = st.beta_columns([1, 1, 1])
        if center.button('Run detection'):
            if pipeline == FULL_PIPELINE:
                bbs = get_objects(image, from_file=False)
                for bb in bbs:
                    crop = crop_image(image, bb)
                    has_cap = predict_cap(crop, model)
                    print(bb, has_cap)
                image = plot_image_with_bbs(image, bbs)
            else:
                has_cap = predict_cap(image, model)
                print(has_cap)

            st.image(np.array(image))


if __name__ == '__main__':
    app()
