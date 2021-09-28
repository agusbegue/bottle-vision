
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


TF_OD_URL = "http://localhost:8501/v1/models/efficientdet:predict"
TF_OD_IMAGE_SIZE = 512
TF_OD_BOTTLE_INDEX = 44
TF_OD_CONF_THRESHOLD = 0.23


TF_CLF_MODEL_PATH = os.path.join(BASE_DIR, 'models/classifier/classifier.h5')
TF_CLF_INPUT_SIZE = (600, 200, 3)
