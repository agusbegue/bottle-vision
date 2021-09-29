import numpy as np
from PIL import Image, ImageFont, ImageDraw

from settings import TF_OD_IMAGE_SIZE as INPUT_SIZE


def crop_image(image, bb):
    w, h = image.size
    return image.crop((w*bb['box_left'], h*bb['box_top'], w*bb['box_right'], h*bb['box_bottom']))


def plot_image_with_bbs(image, results, color='red'):
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

    for r in results:
        top, bottom = r['bb']['box_top']*height, r['bb']['box_bottom']*height
        left, right = r['bb']['box_left']*width, r['bb']['box_right']*width
        text = r['bb']['label'] + ' - Cap: ' + str(r['cap_conf']) + '%'
        text_width, text_height = font.getsize(text)
        display_str_height = (1 + 2 * 0.05) * text_height

        if True or top > display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + display_str_height

        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, top), (right, bottom)], width=2, outline=color)
        draw.rectangle([(right-text_width-2*margin, text_bottom-text_height-2*margin), (right+margin, text_bottom)], fill=color)
        draw.text((right-text_width, text_bottom-text_height-margin), text, fill='black', font=font)

    return image





