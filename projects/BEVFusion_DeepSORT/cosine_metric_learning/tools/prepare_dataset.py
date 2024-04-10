from datasets import veri
import numpy as np
import cv2
import os

DATASET_DIR = './downloads/datasets/veri_resized'

def resize_veri_dataset(image_shape=(128,64,3)):
    filenames, ids, camera_indices = veri.read_train_split_to_str(
            DATASET_DIR)
    
    # Image directory 
    image_dir = os.path.join(DATASET_DIR, 'image_train')
    
    #Alterando para diretório das imagens
    os.chdir(image_dir) 
    
    for i, filename in enumerate(filenames):
        filename = os.path.basename(filename)
        if i % 50 == 0:
            print("Reading %s, %d / %d" % (DATASET_DIR, i, len(filenames)))
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        image = cv2.resize(image,image_shape[:2])
        cv2.imwrite(filename, image)
    

resize_veri_dataset()