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
from tensorflow.keras.applications import MobileNetV3Small

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Compute the confusion matrix on the validation set
        val_preds = self.model.predict(self.train_data[:2])
        val_cm = tf.math.confusion_matrix(self.train_data[2], val_preds)
        
        # Extract true positives, true negatives, false positives, and false negatives
        tp = val_cm[1, 1].numpy()
        tn = val_cm[0, 0].numpy()
        fp = val_cm[0, 1].numpy()
        fn = val_cm[1, 0].numpy()
        
        # Print the confusion matrix and evaluation metrics
        print("Confusion matrix:")
        print(val_cm.numpy())
        print("True positives:", tp)
        print("True negatives:", tn)
        print("False positives:", fp)
        print("False negatives:", fn)

class model_maker:
    nn_input_shape=[32,32]
    siamese=None
    epoch=1
    batch=128

    def __init__(self):

        # Load the MobileNet model
        mobilenet = MobileNetV3Small(
            include_top=False,
            weights="imagenet",
            input_shape=(self.nn_input_shape[0], self.nn_input_shape[1], 3),
            # input_tensor = (None, self.nn_input_shape[0], self.nn_input_shape[1], 3),
            pooling='max',
            classifier_activation=None,
            include_preprocessing = True
        )

        # Define the input layers
        input_1 = layers.Input((self.nn_input_shape[0], self.nn_input_shape[1], 3))
        input_2 = layers.Input((self.nn_input_shape[0], self.nn_input_shape[1], 3))

        # Define the embedding network
        mobilenet1 = mobilenet(input_1)
        mobilenet2 = mobilenet(input_2)
        flattened1 = layers.Flatten()(mobilenet1)
        flattened2 = layers.Flatten()(mobilenet2)
        nn1 = layers.Dense(128, activation="relu")(flattened1)
        nn2 = layers.Dense(128, activation="relu")(flattened2)
        # Define the output layer
        nn_concat=layers.Dense(64,activation='relu')(layers.concatenate([nn1,nn2]))

        output_layer=layers.Dense(4,activation='sigmoid')(nn_concat)
        # Define the Siamese model
        siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
        siamese.summary()
        adam = tf.keras.optimizers.Adam(learning_rate=1)
        siamese.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=adam, metrics=["accuracy"])
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
           
    def train(self,weigths_save_path,model_save_path,x,y):

        log_dir = "runs/train" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        confusion_matrix_callback = ConfusionMatrixCallback()

        """
        ## Train the model
        """
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weigths_save_path, save_weights_only=True)
        history = self.siamese.fit(
            [x[:,0],x[:,1]],
            y,
            # validation_data=([x_val_1, x_val_2], labels_val),
            batch_size=self.batch,
            epochs=self.epoch,
            # callbacks=[confusion_matrix_callback]
        )
        # self.siamese.save(model_save_path)



