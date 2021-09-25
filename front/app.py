import json
import numpy as np
import requests
from PIL import Image, ImageFont, ImageDraw
import streamlit as st


from settings import TF_OD_BOTTLE_INDEX as BOTTLE_INDEX
from settings import TF_OD_IMAGE_SIZE as INPUT_SIZE
from settings import TF_OD_CONF_THRESHOLD as THRESH
from settings import TF_OD_URL



def get_predictions(image, from_file=False):
    if not from_file:
        image_content = np.array(image).tolist()
        input_data = {"instances": [image_content]}
        r = requests.post(url=TF_OD_URL, data=json.dumps(input_data))
                          #headers={"content-type": "application/json"})
        data = json.loads(r.content.decode('utf-8'))['predictions'][0]
    else:
        with open('tests/prediction.json', 'r') as f:
            data = json.loads(f.read())['predictions'][0]

    indexes = [i for i in range(int(data['num_detections']))
               if data['detection_classes'][i] == BOTTLE_INDEX
               and data['detection_scores'][i] > THRESH]
    bbs = [data['detection_boxes'][i] for i in indexes]
    bbs = [{'box_top': bb[0], 'box_left': bb[1], 'box_bottom': bb[2], 'box_right': bb[3],
            'label': 'Bottle'} for bb in bbs]
    return bbs


def plot_image_with_bbs(image, bbs=None, color='red'):
    width, height = image.size
    if width > height:
        image = image.resize((INPUT_SIZE, round(height/width*INPUT_SIZE)))
    else:
        image = image.resize((round(width/height*INPUT_SIZE), INPUT_SIZE))
    width, height = image.size

    try:
        font = ImageFont.truetype('resources/arial.ttf', size=18)
    except IOError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)

    for bb in bbs if bbs else []:
        top, bottom = bb['box_top']*height, bb['box_bottom']*height
        left, right = bb['box_left']*width, bb['box_right']*width
        text_width, text_height = font.getsize(bb['label'])
        display_str_height = (1 + 2 * 0.05) * text_height

        if True or top > display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + display_str_height

        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, top), (right, bottom)], width=2, outline=color)
        draw.rectangle([(right-text_width-2*margin, text_bottom-text_height-2*margin), (right+margin, text_bottom)], fill=color)
        draw.text((right-text_width, text_bottom-text_height-margin), bb['label'], fill='black', font=font)

    return image


def app():
    st.title('Object detection demo: bottles')

    uploaded_file = st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        _, center, _ = st.beta_columns([1, 1, 1])
        if center.button('Run detection'):
            bbs = get_predictions(image, from_file=False)

            image = plot_image_with_bbs(image, bbs)
            st.image(np.array(image))


