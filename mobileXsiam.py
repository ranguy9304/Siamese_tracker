
from gc import callbacks
import itertools
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import datetime
"""
## Hyperparameters
"""

epochs = 3
batch_size = 64
margin = 2# Margin for constrastive loss.

"""
## Load the MNIST dataset
"""


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import data_maker

import model

siamese=model.model_maker()

data_holder=data_maker.data_holder("/home/summer/samik/auvsi_perception/colordataset/cropped_new",None,siamese.nn_input_shape)


siamese.train('/home/summer/samik/auvsi_perception/siam_keras',"/home/summer/samik/auvsi_perception/siam_keras/sia_color.h5",data_holder=data_holder)



results = siamese.evaluate([data_holder.x_train_1_2[0], data_holder.x_train_1_2[1]], data_holder.labels_test)
print("test loss, test acc:", results)
