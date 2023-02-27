import model
import numpy as np
import cv2

mobilenet = model.model_maker()
mobilenet.load_mobilenet()
mobilenet.predict_mobilenet("C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/pentagon_S.jpg",
                            "C:/Users/divya/Desktop/ML/ODCL/saimese_tracker/mobilenet.h5")
