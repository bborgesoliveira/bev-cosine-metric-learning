# vim: expandtab:ts=4:sw=4
import os
import numpy as np
import cv2
import scipy.io as sio
import json

# The maximum vehicle ID in the dataset.
MAX_LABEL = 3457

IMAGE_SHAPE = 128, 64, 3

#CONCLUÍDO
def read_train_split_to_str(dataset_dir):
    """Read training data to list of filenames.

    Parameters
    ----------
    dataset_dir : str
        Path to the nuscenes_reid dataset directory.

    Returns
    -------
    (List[str], List[int], List[int])
        Returns a tuple with the following values:

        * List of image filenames (full path to image files).
        * List of unique IDs for the individuals in the images.
        * List of camera indices.

    """
    filenames, ids, camera_indices = [], [], []
    
    #Obtendo arquivo json do dataset
    filepath = os.path.join(dataset_dir, "object_data.json")
    label_file = {}
    with open(filepath) as json_file:
        label_file:dict = json.load(json_file)
    
    print(len(label_file.keys()))
    exit()
    
    
    #Obtendo todos os elmentos com tag item para extração dos dados
    items = bs_data.find_all('item') 
    for item in items:
        filenames.append(os.path.join(dataset_dir, 'image_train', item.get('imagename')))
        ids.append(int(item.get('vehicleid')))
        camera_indices.append(int(str(item.get('cameraid')).replace('c', '')))

    return filenames, ids, camera_indices

#CONCLUÍDO
def read_train_split_to_image(dataset_dir, image_shape=(128,64)):
    """Read training images to memory. This consumes a lot of memory.

    Parameters
    ----------
    dataset_dir : str
        Path to the VeRi dataset directory.

    Returns
    -------
    (ndarray, ndarray, ndarray)
        Returns a tuple with the following values:

        * Tensor of images in BGR color space of shape 128x64x3.
        * One dimensional array of unique IDs for the individuals in the images.
        * One dimensional array of camera indices.

    """    
    reshape_fn = (
        (lambda x: x) if image_shape == IMAGE_SHAPE[:2]
        else (lambda x: cv2.resize(x, image_shape[::-1])))
    
    filenames, ids, camera_indices = read_train_split_to_str(dataset_dir)

    #images = np.zeros((len(filenames), ) + image_shape + (3, ), np.uint8)
    images = np.zeros((len(filenames), 128, 64, 3), np.uint8)
    for i, filename in enumerate(filenames):
        if i % 1000 == 0:
            print("Reading %s, %d / %d" % (dataset_dir, i, len(filenames)))
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        #images[i] =  reshape_fn(image)
        images[i] = cv2.resize(image,image_shape[::-1])
    
    ids = np.asarray(ids, np.int64)
    camera_indices = np.asarray(camera_indices, np.int64)
    return images, ids, camera_indices


if __name__ == '__main__':
    dataset_dir = './downloads/datasets/nuscenes_reid'
    read_train_split_to_str(dataset_dir)