import os
import cv2
import numpy as np
import model
import random
from tqdm import tqdm
from multiprocessing import Pool

# function to compare a,b,c,d of two file names
def compare_files(values1, values2):
    vector = []
    for i in range(0,4):
        if values1[i] == values2[i]:
            vector.append(1)
        else:
            vector.append(0)
    return vector

def load_images(crop_name):
    crop_image = cv2.imread(crop_name)
    crop_image = cv2.resize(crop_image, (32,32))
    label = crop_name.split("/")[-1].split("_")[:4]
    return crop_image ,label

# main function
def main():
    epochs = 100
    batch_size = 20000
    main_folder_path = "C:/Users/divya/Desktop/Blender/ODCL_Synthetic_Data_Generator/cropped/"
    siamese = model.model_maker()
    crop_images_labels = []
    files_names = os.listdir(main_folder_path)
    crop_images = []
    labels = []
    files_names = np.random.permutation(files_names)
    crop_paths = np.array([os.path.join(main_folder_path,crop_name) for crop_name in files_names if crop_name.endswith(".jpg")])
    with Pool() as p:
        crop_images_labels.append(p.map(load_images, crop_paths))
    print("images loaded")
    crop_images_labels = np.array(crop_images_labels[0], dtype=object)
    for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        crops_train = []
        labels_train = []
        print("beginning shuffling images")
        crop_images_labels = np.random.permutation(crop_images_labels)                 
        crop_images = np.array([crop_images_labels[i][0] for i in range(0,len(crop_images_labels))])
        labels = np.array([crop_images_labels[i][1] for i in range(0,len(crop_images_labels))])
        # Verify the shape of the new array
        if crop_images.shape[0] % 2 == 0:
            # Reshape the array into a nx2 array
            crops_train = crop_images.reshape((int(crop_images.shape[0]/2),2,32,32,3))
            labels_train = labels.reshape((int(crop_images.shape[0]/2), 2,4))
            #compare labels and save vector
            labels_train = np.array([compare_files(labels_train[i][0], labels_train[i][1]) for i in range(0,len(labels_train))])
        else:
            # Reshape the array into a n-1x2 array
            crops_train = crop_images[:-1].reshape((int((crop_images.shape[0]-1)/2),2,32,32,3))
            labels_train = labels[:-1].reshape((int((crop_images.shape[0]-1)/2), 2,4))
            #compare labels and save vector
            labels_train = np.array([compare_files(labels_train[i][0], labels_train[i][1]) for i in range(0,len(labels_train))])
        print("shuffled images")
        len_crop_train = len(crops_train)
        temp_batch_low = 0
        temp_batch_high = batch_size
        
        for j in range(0,len_crop_train//batch_size):
            print("epoch: ",epoch,"/",(len_crop_train//batch_size)+1," ----- batch:",j+1)
            siamese.train('C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/', "C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/", crops_train[temp_batch_low:temp_batch_high], labels_train[temp_batch_low:temp_batch_high])
            temp_batch_low = temp_batch_high
            temp_batch_high = temp_batch_high + batch_size
        siamese.train('C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/', "C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/", crops_train[temp_batch_low:], labels_train[temp_batch_low:])
        siamese.siamese.save('C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/siamese.h5')
if __name__ == '__main__':
    main()