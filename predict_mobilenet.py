import model
import numpy as np
import cv2

mobilenet = model.model_maker()
mobilenet.load_mobilenet()
mobilenet.predict_mobilenet("C:/Users/divya/Desktop/Blender/ODCL_Synthetic_Data_Generator/cropped/0_33_9_4_4525.jpg",
                            "C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/mobilenet.h5")
