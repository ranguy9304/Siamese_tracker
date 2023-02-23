import os
from tkinter import Y
import cv2
import numpy as np
from multiprocessing import Pool
from numpy import empty
from scipy import rand
import tqdm
import random

# define input and output directories
image_dir = "/home/summer/samik/auvsi_perception/colordataset/Images"
label_dir = "/home/summer/samik/auvsi_perception/colordataset/Bounding Boxes"
output_dir = "/home/summer/samik/auvsi_perception/colordataset/cropped_new"


def process_image(image_path):
    crops_per_image=0
    count=0
    # read image
    img = cv2.imread(image_path)

    # get corresponding label file path
    label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")

    # read label file
    labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            label = line.strip().split()
            label[0] = int(label[0])
            label[1:] = [float(x) for x in label[1:]]
            labels.append(label)
    # filter boxes with labels >=14
    filtered_labels = [label for label in labels if label[0] >= 14]
    # loop over boxes with labels <14 and check for overlaps with filtered boxes
    for label in labels:
        if label[0] < 14:
            if(label[1]<7/img.shape[1] or label[1]>1-7/img.shape[1] or label[2]<7/img.shape[0] or label[2]>1-7/img.shape[0] ):
                count+=1
                continue
            for filtered_label in filtered_labels:
                label_box = [label[1], label[2], label[3], label[4]]
                filtered_label_box = [filtered_label[1], filtered_label[2], filtered_label[3], filtered_label[4]]
                if is_box_inside(label_box, filtered_label_box):
                    # crop and save box
                    output_label = "{}_{}_{}_{}".format(label[0], filtered_label[0], int(label[5]), int(filtered_label[5]))
                    # output_path = os.path.join(output_dir, output_label)
                    # os.makedirs(output_path, exist_ok=True)
                    y1,y2,x1,x2=int((label[2]-(label[4]/2))*img.shape[0]),int((label[2]+(label[4]/2))*img.shape[0]), int((label[1]-(label[3]/2))*img.shape[1]),int((label[1]+(label[3]/2))*img.shape[1])
                    cropped_img = img[y1-1:y2+1, x1-1:x2+1]
                    try:
                        crops_per_image+=1
                        cv2.imwrite(os.path.join(output_dir,output_label)+"_"+str(random.randint(1, 1000))+".jpg", cropped_img)
                    except:
                        count+=1
                        continue
    print("og crop : --> "+str(len(labels)))
    print("number of crops : --> "+str(crops_per_image))
    print("number of failed crops : -->" + str(count))
    print("\n ------------------------------------ \n ")



def is_box_inside(box1, box2):
    # if(box1[0]-(box1[2]/2)<box2[0]<box1[0]+(box1[2]/2) and box1[1]-(box1[3]/2)<box2[1]<box1[1]+(box1[3]/2)):
    #     return True
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_overlap = max(0, min(x1+w1/2, x2+w2/2) - max(x1-w1/2, x2-w2/2))
    y_overlap = max(0, min(y1+h1/2, y2+h2/2) - max(y1-h1/2, y2-h2/2))
    overlap_area = x_overlap * y_overlap
    box1_area = w1 * h1
    return overlap_area > 0 and overlap_area / box1_area > 0.001

if __name__ == '__main__':
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get list of image file paths
    image_paths = [os.path.join(image_dir, f.replace(".txt",".jpg")) for f in os.listdir(label_dir) if f.endswith(".txt") ]
    # process images using multiprocessing
    with Pool() as p:
        print("entered pool")
        for _ in tqdm.tqdm(p.imap(process_image, image_paths), total=len(image_paths)):
            pass