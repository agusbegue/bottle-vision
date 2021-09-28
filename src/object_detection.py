import json
import numpy as np
import requests

from settings import TF_OD_URL
from settings import TF_OD_BOTTLE_INDEX as BOTTLE_INDEX
from settings import TF_OD_CONF_THRESHOLD as THRESH


def get_objects(image, from_file=False):
    if not from_file:
        image_content = np.array(image).tolist()
        input_data = {"instances": [image_content]}
        r = requests.post(url=TF_OD_URL, data=json.dumps(input_data))
                          #headers={"content-type": "application/json"})
        data = json.loads(r.content.decode('utf-8'))['predictions'][0]
    else:
        with open('data/prediction.json', 'r') as f:
            data = json.loads(f.read())['predictions'][0]

    indexes = [i for i in range(int(data['num_detections']))
               if data['detection_classes'][i] == BOTTLE_INDEX
               and data['detection_scores'][i] > THRESH]
    bbs = [data['detection_boxes'][i] for i in indexes]
    bbs = [{'box_top': bb[0], 'box_left': bb[1], 'box_bottom': bb[2], 'box_right': bb[3],
            'label': 'Bottle'} for bb in bbs]
    return bbs