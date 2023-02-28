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
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives
import tensorflow_addons as tfa
import gc
tf.config.experimental.set_memory_growth(
    tf.config.experimental.list_physical_devices('GPU')[0], True)
class ClearSessionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()

class model_maker:
    nn_input_shape=[32,32]
    siamese=None
    mobilenet=None
    epoch=1
    batch=128

    def __init__(self):
        pass

    def load_siamese(self):
        
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
        adam = tf.keras.optimizers.Adam()
        tp,fp,tn,fn=FalsePositives(),FalseNegatives(),TruePositives(),TrueNegatives()
        siamese.compile(loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits= False,
                                                                    alpha= 0.25,
                                                                    gamma= 2.0,
                                                                    name = 'sigmoid_focal_crossentropy'), 
                                                                    optimizer=adam, 
                                                                    metrics=["accuracy",tp,fp,tn,fn])
        self.siamese=siamese
    
    def load_mobilenet(self):
        
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
        input = layers.Input((self.nn_input_shape[0], self.nn_input_shape[1], 3))

        # Define the embedding network
        mobilenet = mobilenet(input)
        flattened = layers.Flatten()(mobilenet)

        #Shape nn
        shape_nn_drop = layers.Dropout(0.3)(flattened)
        shape_nn = layers.Dense(256, activation='relu')(shape_nn_drop)
        shape_nn_drop_2 = layers.Dropout(0.3)(shape_nn)
        shape_nn_2 = layers.Dense(64, activation='relu')(shape_nn_drop_2)
        shape_nn_drop_3 = layers.Dropout(0.2)(shape_nn_2)
        shape_nn_3 = layers.Dense(14, activation='softmax')(shape_nn_drop_3)

        #letter nn
        letter_nn_drop = layers.Dropout(0.3)(flattened)
        letter_nn = layers.Dense(256, activation='relu')(letter_nn_drop)
        letter_nn_drop_2 = layers.Dropout(0.3)(letter_nn)
        letter_nn_2 = layers.Dense(64, activation='relu')(letter_nn_drop_2)
        letter_nn_drop_3 = layers.Dropout(0.2)(letter_nn_2)
        letter_nn_3 = layers.Dense(36, activation='softmax')(letter_nn_drop_3)

        #shape_colour nn
        shape_colour_nn_drop = layers.Dropout(0.3)(flattened)
        shape_colour_nn = layers.Dense(256, activation='relu')(shape_colour_nn_drop)
        shape_colour_nn_drop_2 = layers.Dropout(0.3)(shape_colour_nn)
        shape_colour_nn_2 = layers.Dense(64, activation='relu')(shape_colour_nn_drop_2)
        shape_colour_nn_drop_3 = layers.Dropout(0.2)(shape_colour_nn_2)
        shape_colour_nn_3 = layers.Dense(10, activation='softmax')(shape_colour_nn_drop_3)

        #letter_colour nn
        letter_colour_nn_drop = layers.Dropout(0.3)(flattened)
        letter_colour_nn = layers.Dense(256, activation='relu')(letter_colour_nn_drop)
        letter_colour_nn_drop_2 = layers.Dropout(0.3)(letter_colour_nn)
        letter_colour_nn_2 = layers.Dense(64, activation='relu')(letter_colour_nn_drop_2)
        letter_colour_nn_drop_3 = layers.Dropout(0.2)(letter_colour_nn_2)
        letter_colour_nn_3 = layers.Dense(10, activation='softmax')(letter_colour_nn_drop_3)

        # Define the output layer
        # output_layer=
        # Define the model
        self.mobilenet = keras.Model(inputs=input, outputs=[shape_nn_3,letter_nn_3,shape_colour_nn_3,letter_colour_nn_3])
        self.mobilenet.summary()
        adam = tf.keras.optimizers.Adam()
        tp,fp,tn,fn=FalsePositives(),FalseNegatives(),TruePositives(),TrueNegatives()
        self.mobilenet.compile(loss = tf.keras.losses.CategoricalCrossentropy(from_logits= False),
                                                                    optimizer=adam, 
                                                                    metrics=["accuracy"])
    def load_model_weights(self,path):
        self.siamese.load_weights(path)
    def predict_mobilenet(self,path,weights):
        self.mobilenet.load_weights(weights)
        image = cv2.imread(path)
        image = cv2.resize(image,(self.nn_input_shape[0],self.nn_input_shape[1]))
        pred = self.mobilenet.predict(tf.expand_dims(image,0))
        print(pred)
        #add threshold and print prediction
        whitelist_shape = ["CIRCLE","SEMI CIRCLE","QUARTER CIRCLE","TRIANGLE",
                            "SQUARE", "DIAMOND","TRAPEZOID","RECTANGLE",
                            "PENTAGON","HEXAGON","HEPTAGON",
                            "OCTAGON","STAR","CROSS"]
        whitelist_letter = ["0","1","2","3","4","5","6","7","8",
                            "9","A","B","C","D","E","F","G","H","I",
                            "J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
        whitelist_color = ["white","gray","black","red","blue","green","yellow","orange","brown","purple"]
        shape = whitelist_shape[np.where(pred[0][0]==max(pred[0][0]))[0][0]]
        letter = whitelist_letter[np.where(pred[1][0]==max(pred[1][0]))[0][0]]
        shape_colour = whitelist_color[np.where(pred[2][0]==max(pred[2][0]))[0][0]]
        letter_colour = whitelist_color[np.where(pred[3][0]==max(pred[3][0]))[0][0]]
        print("shape: ",shape," confidence: ",max(pred[0][0])*100,"%")
        print("letter: ",letter," confidence: ",max(pred[1][0])*100,"%")
        print("shape_colour: ",shape_colour," confidence: ",max(pred[2][0])*100,"%")
        print("letter_colour: ",letter_colour," confidence: ",max(pred[3][0])*100,"%")

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
    def train_mobilenet(self,weigths_save_path,model_save_path,x,y):
        log_dir = "runs/train" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        """
        ## Train the model
        """
        self.mobilenet.reset_states()
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weigths_save_path, save_weights_only=True)
        self.mobilenet.fit(
            x,
            y,
            # validation_data=([x_val_1, x_val_2], labels_val),
            batch_size=self.batch,
            epochs=self.epoch,
            # callbacks=[confusion_matrix_callback]
        )
        # self.siamese.save(model_save_path)
        # self.siamese.save_weights(weigths_save_path)
    def train_siamese(self,weigths_save_path,model_save_path,x,y):

        log_dir = "runs/train" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        """
        ## Train the model
        """
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weigths_save_path, save_weights_only=True)
        self.siamese.fit(
            [x[:,0],x[:,1]],
            y,
            # validation_data=([x_val_1, x_val_2], labels_val),
            batch_size=self.batch,
            epochs=self.epoch,
            # callbacks=ClearMemory()
            callbacks=[ClearSessionCallback()]
        )
        # self.siamese.save(model_save_path)