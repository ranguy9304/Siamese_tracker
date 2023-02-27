import os
import cv2
import numpy as np
import model
import random
from tqdm import tqdm
from multiprocessing import Pool

# function to compare a,b,c,d of two file names
def compare_files(file1, file2):
    vector = []
    values1 = file1.split("_")[:4]
    values2 = file2.split("_")[:4]
    for i in range(0,4):
        if values1[i] == values2[i]:
            vector.append(1.0)
        else:
            vector.append(0.0)
    return vector

def load_images(crop_name):
    crop_image1 = cv2.imread(crop_name[0])
    crop_image1 = cv2.resize(crop_image1,(32,32))
    crop_image2 = cv2.imread(crop_name[1])
    crop_image2 = cv2.resize(crop_image2,(32,32))
    crop_label1 = crop_name[0].split("/")[-1]
    crop_label2 = crop_name[1].split("/")[-1]
    return crop_image1,crop_image2, compare_files(crop_label1, crop_label2)

# main function
def main():
    epochs = 3
    batch_size = 30000
    main_folder_path = "C:/Users/divya/Desktop/Blender/ODCL_Synthetic_Data_Generator/cropped/"
    n = 50000 # number of pairs to create at once
    siamese = model.model_maker()
    x_pairs = []
    y_labels = []
    
    count = 0
    files_names = os.listdir(main_folder_path)
    len_files=len(files_names)-1
    pbar = tqdm(total=len_files/2, desc="global_progress", unit="pair")
    while(epochs >0):
        crop_images = []
        files_names = np.random.permutation(files_names)
        crop_paths = np.array([os.path.join(main_folder_path,crop_name) for crop_name in files_names if crop_name.endswith(".jpg")])
        if len(crop_paths) % 2 == 0:
        # Reshape the array into a 3200x2 array
            crop_paths = crop_paths.reshape((int((len_files+1)/2), 2))
        else:
            # Reshape the array into a 3199x2 array
            crop_paths = crop_paths[:-1].reshape((int(len_files/2), 2))

        # Verify the shape of the new array
        with Pool() as p:
            crop_images.append(p.map(load_images, crop_paths))
        for j in range(0,len(crop_images[0])):
            #load img
            x_pairs.append([crop_images[0][j][0],crop_images[0][j][1]])
            y_labels.append(crop_images[0][j][2])
            if (j%batch_size == 0 and j!=0) or j == len(crop_images[0])-1:
                siamese.train('C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/', "C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/", np.array(x_pairs), np.array(y_labels))
                x_pairs = []
                y_labels = []
        pbar.update(batch_size)
        
        x_pairs = []
        y_labels = []
        siamese.siamese.save('C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/siamese.h5')
    pbar.close()
if __name__ == '__main__':
    main()