import numpy as np
import json
import requests
import cv2

from settings import TF_CLF_URL
from settings import TF_CLF_INPUT_SIZE as INPUT_SIZE


def predict_cap(crop):
    crop = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    crop = cv2.resize(crop, (INPUT_SIZE[1], INPUT_SIZE[0]), interpolation=cv2.INTER_AREA)
    crop = rescale(crop)

    crop = np.array(crop).tolist()
    input_data = {"instances": [crop]}
    prediction = json.loads(requests.post(url=TF_CLF_URL, data=json.dumps(input_data)).content.decode('utf-8'))
    return int(100*prediction['predictions'][0][0])


def rescale(data, to_255=False):
    if to_255:
        return np.multiply(np.add(data, 1), 255/2).astype(np.int32)
    return np.add(np.multiply(data, 2/255), -1)

