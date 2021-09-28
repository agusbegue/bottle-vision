import numpy as np
import cv2
from keras.models import load_model
from keras.applications.xception import preprocess_input
from keras.layers import Layer

from settings import TF_CLF_MODEL_PATH as MODEL_PATH
from settings import TF_CLF_INPUT_SIZE as INPUT_SIZE


#model = load_model(MODEL_PATH)


def predict_cap(crop, model):
    crop = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    crop = cv2.resize(crop, (INPUT_SIZE[1], INPUT_SIZE[0]), interpolation=cv2.INTER_AREA)
    crop = preprocess_input(crop)
    prediction = model.predict(crop)
    return prediction


class ApplicationLayer(Layer):

    def __init__(self, application, **kwargs):
        self.application = application
        self.app_params = {param: kwargs.pop(param) for param in kwargs.copy().keys()}
        self.functional = self.application(**self.app_params)
        super().__init__(**kwargs)
        for layer in self.functional.layers:
            layer.trainable = False

    def call(self, inputs):
        return self.functional(inputs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'application': self.application,
            'app_params': self.app_params
        })
        return config


from keras.layers import InputLayer,BatchNormalization, Dropout, Dense
from keras.applications.xception import Xception
from keras import Sequential
from keras.regularizers import l2
def get_classifier_model():

    xception_config = {
        'include_top': False,
        'weights': 'imagenet',
        'pooling': 'max',
    }

    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SIZE))
    model.add(ApplicationLayer(Xception, **xception_config))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dense(1, activation='sigmoid')) # 2 nodes when learning tag
    model.load_weights('../models/classifier/classifier.h5')
    return model


