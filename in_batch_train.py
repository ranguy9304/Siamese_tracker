
# from gc import callbacks
# import itertools
# import random
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import matplotlib.pyplot as plt

# import datetime
# """
# ## Hyperparameters
# """

# epochs = 3
# batch_size = 64
# margin = 2# Margin for constrastive loss.

# """
# ## Load the MNIST dataset
# """


# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split

# import model

# siamese=model.model_maker()




# siamese.train('/home/summer/samik/auvsi_perception/siam_keras',"/home/summer/samik/auvsi_perception/siam_keras/sia_color.h5",data_holder=data_holder)



# results = siamese.evaluate([x_train_1_2[0], x_train_1_2[1]], labels_test)
# print("test loss, test acc:", results)


import os
import cv2
import numpy as np
from siamese import Siamese
import random
from tqdm import tqdm
# function to compare a,b,c,d of two file names
def compare_files(file1, file2):
    values1 = file1.split("_")[:4]
    values2 = file2.split("_")[:4]
    vector = [int(values1[i] == values2[i]) for i in range(4)]
    return vector

# main function
def main():
    epochs = 3
    batch_size = 64
    main_folder_path = "/path/to/folder/"
    n = 50000 # number of pairs to create at once
    siamese = Siamese()
    x_pairs = []
    y_labels = []
    count = 0
    files_names=os.listdir(main_folder_path)
    len_files=len(files_names)-1
    pbar = tqdm(total=len_files/2, desc="global_progress", unit="pair")
    while(len_files/2>1 and  epochs >0):
        if len_files/2<batch_size :
            batch_size=len_files
            epochs = epochs - 1
            temp = True
        for j in range(0,batch_size):  
            file_index1 = random.randint(0,len_files)
            files_names.pop(file_index1)
            len_files=len_files-1
            file_index2 = random.randint(0,len_files)
            files_names.pop(file_index1)
            len_files=len_files-1
            #load img
            image1 = cv2.imread(main_folder_path+files_names[file_index1])
            image2 = cv2.imread(main_folder_path+files_names[file_index2])
            x_pairs.append((image1, image2))
            y_labels.append(compare_files(files_names[file_index1], files_names[file_index2]))
        pbar.update(batch_size)
        if(temp)==True:
            files_names=os.listdir(main_folder_path)
            len_files=len(files_names)-1
        siamese.train('/home/summer/samik/auvsi_perception/siam_keras', "/home/summer/samik/auvsi_perception/siam_keras/sia_color.h5", np.array(x_pairs), np.array(y_labels))
        x_pairs = []
        y_labels = []
    pbar.close()
if __name__ == '__main__':
    main()