import cv2
import numpy as np
import os

IMG_SIZE = 64

def load_data(data_dir):
    data = []
    labels = []
    
    for label in os.listdir(data_dir):
        path = os.path.join(data_dir, label)
        
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            data.append(image)
            labels.append(label)
    
    return np.array(data), np.array(labels)