from bs4 import BeautifulSoup
import os
import numpy as np
import cv2

dataset_dir = '/home/lasi/Downloads/datasets/veri'

def open_xml():  

    xml_dir = os.path.join(dataset_dir, "train_label.xml")

    # Reading the data inside the xml file to a variable under the name  data
    with open(xml_dir, 'r') as f:
        data = f.read() 

    # Passing the stored data inside the beautifulsoup parser 
    bs_data = BeautifulSoup(data, 'html.parser') 

    # Finding all instances of tag   
    filenames, ids, camera_indices = [], [], []
    items = bs_data.find_all('item') 
    for item in items:
        filenames.append(os.path.join(dataset_dir, 'image_train', item.get('imagename')))
        ids.append(int(item.get('vehicleid')))
        camera_indices.append(int(str(item.get('cameraid')).replace('c', '')))

    print(filenames[:10]) 
    print()
    print(ids[:10])
    print()
    print(camera_indices[:10])

    # i = 0
    # for tag in items:
    #     print(tag.get('imagename'))
    #     i+=1
        
    #     if i == 10: break

    # Using find() to extract attributes of the first instance of the tag 
    # b_name = bs_data.find('child', {'name':'Acer'}) 
    # print(b_name) 

    # # Extracting the data stored in a specific attribute of the `child` tag 
    # value = b_name.get('qty') 
    # print(value)

def read_train_split_to_str(dataset_dir):
    """Read training data to list of filenames.

    Parameters
    ----------
    dataset_dir : str
        Path to the VeRi dataset directory.

    Returns
    -------
    (List[str], List[int], List[int])
        Returns a tuple with the following values:

        * List of image filenames (full path to image files).
        * List of unique IDs for the individuals in the images.
        * List of camera indices.

    """
    filenames, ids, camera_indices = [], [], []
    
    #Obtendo arquivo XML do treinamento
    xml_dir = os.path.join(dataset_dir, "train_label.xml")
    
    # Lendo arquivo xml
    with open(xml_dir, 'r') as f:
        data = f.read() 

    # Convertendo arquivo para ser analisado
    bs_data = BeautifulSoup(data, 'html.parser') 
    
    #Obtendo todos os elmentos com tag item para extração dos dados
    items = bs_data.find_all('item') 
    for item in items:
        filenames.append(os.path.join(dataset_dir, 'image_train', item.get('imagename')))
        ids.append(int(item.get('vehicleid')))
        camera_indices.append(int(str(item.get('cameraid')).replace('c', '')))

    return filenames, ids, camera_indices


def read_train_split_to_image(dataset_dir, image_shape=(128, 64)):
        
    filenames, ids, camera_indices = read_train_split_to_str(dataset_dir)

    images = np.zeros((len(filenames), ) + image_shape + (3, ), np.uint8)
    i = 0
    for i, filename in enumerate(filenames):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        images[i] = cv2.resize(image,image_shape[::-1])
        path = os.path.join(os.getcwd(), 'images', filename.split('/')[-1])
        cv2.imwrite(path, images[i])
         
        print(path)
        
        if i == 30: break



#open_xml()
#read_train_split_to_image(dataset_dir)

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# with tf.device('/gpu:1'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)
# with tf.compat.v1.Session() as sess:
#     print (sess.run(c))

def visualize_cosine_metric_learning_output():
    import pandas as pd
    import tensorflow as tf
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # read from summary writer
    event_acc = EventAccumulator("./eval_output/veri/cosine-softmax")
    event_acc.Reload()
    print(event_acc.Tags())
    print('\nPrecision_1:')
    print(pd.DataFrame(event_acc.Scalars('Precision_1')))
    print('\nPrecision_5:')
    print(pd.DataFrame(event_acc.Scalars('Precision_5')))
    print('\nPrecision_10:')
    print(pd.DataFrame(event_acc.Scalars('Precision_10')))
    print('\nPrecision_20:')
    print(pd.DataFrame(event_acc.Scalars('Precision_20')))
    # pd.DataFrame([(w, s, tf.make_ndarray(t)) for w, s, t in event_acc.Scalars('Precision_1')],
    #          columns=['wall_time', 'step', 'tensor'])
    
visualize_cosine_metric_learning_output()
