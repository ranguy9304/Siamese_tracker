import os
import cv2
import numpy as np
import model
import random
from tqdm import tqdm
from multiprocessing import Pool
import tensorflow as tf

# function to compare a,b,c,d of two file names
def onehot(label):
    shape = np.zeros(14,dtype=np.float32)
    letter = np.zeros(36,dtype=np.float32)
    shape_colour = np.zeros(10,dtype=np.float32)
    letter_colour = np.zeros(10,dtype=np.float32)
    # print(label)
    # print(shape,letter,shape_colour,letter_colour)
    shape[int(label[0])] = 1.0
    letter[int(label[1])-14] = 1.0
    shape_colour[int(label[2])] = 1.0
    letter_colour[int(label[3])] = 1.0
    # print(shape,letter,shape_colour,letter_colour)
    vector = [shape,letter,shape_colour,letter_colour]
    return vector

def load_images(crop_name):
    crop_image = cv2.imread(crop_name)
    crop_image = cv2.resize(crop_image, (32,32))
    label = crop_name.split("/")[-1].split("_")[:4]
    return crop_image ,label

# main function
def main():
    epochs = 100
    batch_size = 50000
    main_folder_path = "C:/Users/divya/Desktop/Blender/ODCL_Synthetic_Data_Generator/cropped/"
    mobilenet = model.model_maker()
    mobilenet.load_mobilenet()
    crop_images_labels = []
    files_names = os.listdir(main_folder_path)
    crop_images = []
    labels = []
    files_names = np.random.permutation(files_names)[:200000]
    crop_paths = np.array([os.path.join(main_folder_path,crop_name) for crop_name in files_names if crop_name.endswith(".jpg")])
    with Pool() as p:
        crop_images_labels.append(p.map(load_images, crop_paths))
    print("images loaded")
    crop_images_labels = np.array(crop_images_labels[0], dtype=object)
    for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        
        print("beginning shuffling images")
        crop_images_labels = np.random.permutation(crop_images_labels)
        len_crop_train = len(crop_images_labels)   
        # print(labels_train)
        print("shuffled images")
        temp_batch_low = 0
        temp_batch_high = batch_size
        for j in range(0,(len_crop_train//batch_size)+1):
            crops_train = []
            shape_train = []
            letter_train = []
            shape_colour_train = []
            letter_colour_train = []
            if temp_batch_high > len_crop_train:
                temp_batch_high = len_crop_train
            
            for i in range(temp_batch_low,temp_batch_high):
                crops_train.append(crop_images_labels[i][0])
                shape_train.append(onehot(crop_images_labels[i][1])[0])
                letter_train.append(onehot(crop_images_labels[i][1])[1])
                shape_colour_train.append(onehot(crop_images_labels[i][1])[2])
                letter_colour_train.append(onehot(crop_images_labels[i][1])[3])
            crops_train = tf.convert_to_tensor(crops_train,dtype=tf.float32)
            # print(crops_train)
            print("epoch: ",epoch,"/",epochs," ----- batch: ",j+1,"/",(len_crop_train//batch_size)+1)
            mobilenet.train_mobilenet('C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/', "C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/", crops_train, [tf.convert_to_tensor(shape_train),tf.convert_to_tensor(letter_train),tf.convert_to_tensor(shape_colour_train),tf.convert_to_tensor(letter_colour_train)])
            temp_batch_low = temp_batch_high
            temp_batch_high = temp_batch_high + batch_size
        mobilenet.mobilenet.save('C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/mobilenet.h5')
if __name__ == '__main__':
    main()