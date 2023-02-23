from itertools import count
import os
from traceback import print_tb
from turtle import width
import cv2


img_width=3840
img_height =2160
def crop_images(images_folder, labels_folder, output_folder):
    # Get a list of all the files in the images folder
    # images = ['0000.jpg']
    images=[f for f in os.listdir(images_folder) if f.endswith('.jpg')]    
    # print(images)
    # Create the output folder if it doesn't already exist

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through each image
    for image_filename in images:
        current_letter=""
    
        # Get the corresponding label file
        label_filename = image_filename.replace('.jpg', '.txt')
        label_path = os.path.join(labels_folder, label_filename)
        print(os.path.join(images_folder, image_filename))
        # print()
        
        # Read the image and the label data
        image = cv2.imread(os.path.join(images_folder, image_filename))
        print(image)

        if not os.path.exists(label_path) :
                continue
        with open(label_path, 'r') as f:
            label_data = f.readlines()
        
        # Loop through each object in the label data
        count=0     
        for label in label_data:
            current_letter=""
            # Split the label data into columns
            columns = label.strip().split()
            
            # Extract the object id and bounding box coordinates
            object_id = int(columns[0])
            color_id=int(columns[5])
            if object_id>13:
                continue
            center_x, center_y, box_width,box_height  = [float(x) for x in columns[1:5]]
            center_x *= img_width
            center_y *= img_height
            box_width *= img_width
            box_height *= img_height
            x_min=center_x-box_width/2
            x_max=center_x+box_width/2
            y_min=center_y-box_height/2
            y_max=center_y+box_height/2


            if(y_min<7 or y_max>img_width-7 or x_min<7 or x_max<img_height-7 ):
                continue
            # print(x_min)
            # print(x_max)
            
            # Crop the object from the image
            # print(image[int(y_min):int(y_max), int(x_min):int(x_max)])
            object_image = image[int(y_min)-1:int(y_max)+1, int(x_min)-1:int(x_max)+1]
            # print(object_image)
            for label1 in label_data:
            # Split the label data into columns
                columns1 = label1.strip().split()
                
          
                object_id1 = int(columns1[0])
                color_id1=int(columns1[5])
                print(object_id1)
                if object_id1<=13:
                    continue
                center_x_letter, center_y_letter, box_width_rn,box_height_rn  = [float(x) for x in columns1[1:5]]
                center_x_letter *= img_width
                center_y_letter *= img_height
                print(f'{x_min} , {x_max} : {center_x_letter}')
                if(x_min<center_x_letter<x_max and y_min<center_y_letter<y_max):
                    print("assign")
                    current_letter=object_id1
                    current_color=color_id1
                    break
        
            
            # Save the cropped object image to the output folder
            object_filename = f'{object_id}_{current_letter}_{color_id}_{current_color}'+str(count)+'.jpg'
            count+=1
            if not os.path.exists(os.path.join(output_folder,str(object_id)+"_"+str(current_letter)+"_"+str(color_id)+"_"+str(current_color))) :
                print("created path")
                os.makedirs(os.path.join(output_folder,str(object_id)+"_"+str(current_letter)+"_"+str(color_id)+"_"+str(current_color)))
         
            object_path = os.path.join(output_folder,str(object_id)+"_"+str(current_letter)+"_"+str(color_id)+"_"+str(current_color), object_filename)
            print(object_path)
            cv2.imwrite(object_path, object_image)

# Example usage
images_folder = '/home/summer/samik/auvsi_perception/colordataset/Images/'
labels_folder = '/home/summer/samik/auvsi_perception/colordataset/Bounding Boxes/'
output_folder = '/home/summer/samik/auvsi_perception/colordataset/cropped/'

crop_images(images_folder, labels_folder, output_folder)