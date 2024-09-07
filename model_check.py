import tensorflow as tf
import numpy as np
import cv2
import keras
from tensorflow.keras.models import Model
model = tf.keras.models.load_model('models/efficient_net_B2.keras')
def prediction_to_name(data):
    if(data>0.4):  #threshold =0.4
        return 1
    else:
        return 0
CLASS_NAMES=["CANCER","NON CANCER"]
test_img=cv2.imread("non_cancer.png")
test_img= cv2.resize(test_img,(224,224))
test_img= tf.expand_dims(test_img,axis=0)
print(CLASS_NAMES[prediction_to_name(model.predict(test_img)[0][0])])
print(model.predict(test_img)[0][0])