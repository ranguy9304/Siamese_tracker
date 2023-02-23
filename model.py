import itertools
from numpy import imag
import tensorflow as tf
import numpy as np
import os

from gc import callbacks
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
import datetime
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
class model_maker:
    nn_input_shape=[32,32]
    siamese=None
    epoch=5
    batch=10

    def __init__(self):

        # Load the MobileNet model
        mobilenet = MobileNet(
            include_top=False,
            weights="imagenet",
            input_shape=(self.nn_input_shape[0], self.nn_input_shape[1], 3),
            pooling='avg'
        )

        # Define the input layers
        input_1 = layers.Input((self.nn_input_shape[0], self.nn_input_shape[1], 3))
        input_2 = layers.Input((self.nn_input_shape[0], self.nn_input_shape[1], 3))

        # Define the embedding network
        embedding_network = keras.models.Sequential([
            layers.Lambda(preprocess_input, input_shape=(self.nn_input_shape[0], self.nn_input_shape[1], 3)),
            mobilenet,
            layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
        ])

        # As mentioned above, Siamese Network share weights between
        # tower networks (sister networks). To allow this, we will use
        # same embedding network for both tower networks.
        tower_1 = embedding_network(input_1)
        tower_2 = embedding_network(input_2)

        # # Define the distance metric layer
        # merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
        # normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)

        # # Define the output layer
        # output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
        nn_concat=layers.Dense(64,activation='relu')(layers.concatenate([tower_1,tower_2]))
        output_layer=layers.Dense(1,activation='sigmoid')(nn_concat)
        # Define the Siamese model
        siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
        siamese.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer="RMSprop", metrics=["accuracy"])
        self.siamese=siamese
    def load_model_weights(self,path):
        self.siamese.load_weights(path)
    def predict(self,path1,path2,type):
        if(type==0):
            print(path1)
            print(path2)
            image1=cv2.imread(path1)
            image2=cv2.imread(path2)
            image1=cv2.resize(image1,(self.nn_input_shape[0],self.nn_input_shape[1]))
            image2=cv2.resize(image2,(32,32))
            # cv2.imshow("img1",image1)
            # cv2.waitKey(0)

            # cv2.imshow("img2",image2)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # image1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
            # image2=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
            print(np.shape(image1))
            image1=tf.expand_dims(image1,0)
            image2=tf.expand_dims(image2,0)

            # Print the output
            diff=self.siamese.predict([image2,image1])
            if(diff > 0.7):
                print("different : "+str(diff))
            else:
                print("same : "+str(diff))
            return diff
        else :
            print("enter checker")
            best_pre=1
            best_match_path="None"
            best_2nd=""
            best_3rd=""


            for filename in os.listdir(path1):
    # check if the file is an image
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    # open the image file
                    image_path = os.path.join(path1, filename)
                    
                        
                    result = self.predict(image_path,path2,0)
                    best_3rd=image_path
                    if result<best_pre:
                        best_pre=result
                        best_3rd=best_2nd
                        best_2nd=best_match_path
                        best_match_path=image_path
                        

            # do something with the result (e.g., print it)
            image1=cv2.imread(best_match_path)
            image2=cv2.imread(path2)
            print("-------  best match diff : "+str(best_pre)+" -------")
            best_match=cv2.imread(best_match_path)
            cv2.imshow("img1",best_match)
            cv2.waitKey(0)
            best_match2nd=cv2.imread(best_2nd)
            cv2.imshow("img1",best_match2nd)
            cv2.waitKey(0)
            print(best_3rd)
            best_match3nd=cv2.imread(best_3rd)
            cv2.imshow("img1",best_match3nd)
            cv2.waitKey(0)

            cv2.imshow("img2",image2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
           
    def train(self,weigths_save_path,model_save_path,data_holder):

        log_dir = "runs/train" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


        """
        ## Train the model
        """
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weigths_save_path, save_weights_only=True)
        history = self.siamese.fit(
            [data_holder.x_train_1_2[0], data_holder.x_train_1_2[1]],
            data_holder.labels_train,
            # validation_data=([x_val_1, x_val_2], labels_val),
            batch_size=self.batch,
            epochs=self.epoch,
            callbacks=[cp_callback,tensorboard_callback]
        )
        self.siamese.save(model_save_path)



