import os
from re import X
import cv2
import numpy as np
from numpy import imag
import random
class data_holder:
    x_test=[]
    y_test=[]
    x_train=[]
    y_train=[]
    pairs_train=[]
    labels_train=[]
    x_train_1_2 =[[None],[None]]
    x_test_1_2 =[[None],[None]]
    pairs_test=[]
    labels_test=[]
    class_count=0

    def __init__(self,train_folder_path,test_folder_path,nn_input_size) :
        if train_folder_path != None :
            self.x_train , self.y_train=self.make_data(train_folder_path,"train",nn_input_size)
            self.pairs_train,self.labels_train=data_holder.make_pairs(self.x_train,self.y_train)

            self.x_train_1_2[0] = self.pairs_train[:, 0]  # x_train_1.shape is (60000, 28, 28)
            self.x_train_1_2[1] = self.pairs_train[:, 1]            

        if test_folder_path != None:
            self.x_test,self.y_test=self.make_data(test_folder_path,"test",nn_input_size)
            self.pairs_test,self.labels_test=data_holder.make_pairs(self.x_test,self.y_test)
            self.x_test_1_2[0] = self.pairs_test[:, 0]  # x_train_1.shape is (60000, 28, 28)
            self.x_test_1_2[1] = self.pairs_test[:, 1]
 
    def make_data(self,main_folder_path,type_of_data,nn_input_size):
        x = []
        y = []
        class_count = -1
        class_dict = {}
        count =0
        count2 =0
        for filename in os.listdir(main_folder_path):
            if filename.endswith(".jpg"):
              
                values = filename.split("_")[:4]
                digits = "_".join(values)
                if digits not in class_dict:
                    class_count += 1
                    class_dict[digits] = class_count
                image_path = os.path.join(main_folder_path, filename)
                # print(image_path)
                image = cv2.imread(image_path)
                try:
                    count2+=1
                    image = cv2.resize(image, (nn_input_size[0], nn_input_size[1]))
                except:
                    count = count+1
                    print("error----"+str(count)+"-----worked-----"+str(count2))
                    if ( count2 > 100000):
                        break
                    continue
                x.append(image)
                y.append(class_dict[digits])

        print("y min " + str(min(y)))
        x = np.array(x)
        y = np.array(y)
        x = x.astype("float32")
        self.class_count = class_count

        return x, y
        # x=[]
        # y=[]
        # class_count=-1
        # flag=0
        # for folder_name,subfolders,files in os.walk(main_folder_path):
            
        #     if folder_name[-1] == "_":
        #         continue
        #     # print(folder_name)
            
        #     for filename in files:
        #         flag=0
        #         if filename.endswith(".jpg"):
        #             flag=1
        #             image_path=os.path.join(folder_name,filename)
        #             image =cv2.imread(image_path)
        #             image=cv2.resize(image,(nn_input_size[0],nn_input_size[1]))
        #             x.append(image)
        #             y.append(class_count)
        #     class_count+=1
        #     # if flag==1:
        # print("y min " + str(min(y)))
        # x=np.array(x)
        # y=np.array(y)
        # x=x.astype("float32") 
        # # if type_of_data == "train":
        # self.class_count=class_count    
        # return x,y       
    @staticmethod
    def make_pairs(x, y):
    

        num_classes = max(y) +1
        print(num_classes)
        digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
        print(digit_indices[0])

        pairs = []
        labels = []

        for idx1 in range(len(x)):
            # add a matching example
            x1 = x[idx1]
            label1 = y[idx1]
            idx2 = random.choice(digit_indices[label1])
            x2 = x[idx2]

            pairs += [[x1, x2]]
            labels += [0]

            # add a non-matching example
            label2 = random.randint(0, num_classes-1 )
            while label2 == label1 or label2>num_classes:
                label2 = random.randint(0, num_classes - 1)
            # print(label2)
            # if(label2 > num_classes):
            #     print("out--")
            #     break

            # print(label2)
            # print(label2)
            try :
                idx2 = random.choice(digit_indices[label2])
            except:
                continue
            x2 = x[idx2]

            pairs += [[x1, x2]]
            labels += [1]

        return np.array(pairs), np.array(labels).astype("float32")


            






