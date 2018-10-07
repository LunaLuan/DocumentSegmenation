import numpy as np
import tensorflow as tf

from keras.layers import *
from keras.models import *

import cv2

import keras
from processing import  resize_handle
from util import helpers

LABEL_INFO = 'lib/labels.csv'


def pre_processing(image):
    image = resize_handle.resize_image_by_neg_pad(image, desired_size=1024)
    image = image.reshape(-1, image.shape[0], image.shape[1], image.shape[2])
    return image

def post_processing(predict, image):
    class_names, label_values = helpers.get_label_info(LABEL_INFO)

    predict = helpers.reverse_one_hot(predict)
    predict = helpers.colour_code_segmentation(predict, label_values)
    predict = predict.astype('uint8')

    predict = resize_handle.resize_image_from_neg_pad(predict[0], image.shape[:2])

    return predict



def run(image):
    model = load_model('./lib/document_layout_big_image_1024.h5')
    p_image = pre_processing(image)

    predict = model.predict(p_image)
    p_predict = post_processing(predict, image)

    return p_predict