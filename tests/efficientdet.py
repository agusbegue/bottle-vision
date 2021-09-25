import os
import sys
import json
import requests
import numpy as np
from PIL import Image
sys.path.append(os.getcwd())

#from object_detection.utils import label_map_util

from settings import TF_OD_IMAGE_SIZE as IMAGE_SIZE, TF_OD_BOTTLE_INDEX as BOTTLE_INDEX

#category_index = label_map_util.create_category_index_from_labelmap(TF_OD_LABELS_PATH,
#                                                                    use_display_name=True)

IMAGE_PATH = 'tests/coca512.png'


#image_content = cv2.imread(IMAGE_PATH, 1).astype('uint8').tolist()

image_content = np.array(Image.open(IMAGE_PATH)).tolist()
data = {"instances": [image_content]}
r = requests.post(url="http://localhost:8501/v1/models/efficientdet:predict", data=json.dumps(data),
                  headers={"content-type": "application/json"})

# print(r.status_code, r.content)
with open('tests/prediction.json', 'w') as f:
    f.write(r.content.decode('utf-8'))

