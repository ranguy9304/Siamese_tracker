import itertools
from numpy import imag
import tensorflow as tf
import numpy as np
from gc import callbacks
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
import model 
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
import os
import data_maker


siamese_class=model.model_maker()
path_weights='/home/summer/samik/auvsi_perception/siam_keras/sia_color.h5'
siamese_class.load_model_weights(path_weights)
path1="/home/summer/samik/auvsi_perception/yolov5/runs/detect/exp/crops/PENTAGON/"
path2="'/home/summer/samik/auvsi_perception/colordataset/cropped_new/8_42_3_7_302.jpg'"
data_holder=data_maker.data_holder(None,"/home/summer/samik/auvsi_perception/colordataset/cropped_new",siamese_class.nn_input_shape)
x_test_1 = data_holder.pairs_test[:, 0]  # x_test_1.shape = (20000, 28, 28)
x_test_2 = data_holder.pairs_test[:, 1]

def evaluate_model(x_test_1,x_test_2,labels_test):

    results = siamese_class.siamese.evaluate([x_test_1, x_test_2], labels_test)
    print("test loss, test acc:", results)

def confusion_matrix(x_test_1,x_test_2,labels_test,predictions):
    predictions = siamese_class.siamese.predict([x_test_1, x_test_2])
    predicted_labels = predictions > 0.5
    from sklearn.metrics import confusion_matrix
    # Create the confusion matrix
    cm = confusion_matrix(labels_test, predicted_labels)

    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '1'], rotation=45)
    plt.yticks(tick_marks, ['0', '1'])
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main():
    siamese_class.predict(path1=path1,path2=path2,type=1)
    # evaluate_model(x_test_1,x_test_2,data_holder.labels_test)
    # predictions = siamese_class.siamese.predict([x_test_1, x_test_2])

    # confusion_matrix(x_test_1,x_test_2,data_holder.labels_test,predictions)

main()