from datasets import veri
import cv2
import os

DATASET_DIR = './downloads/datasets/veri_resized'

def main():
    filenames, ids, camera_indices = veri.read_train_split_to_str(
        DATASET_DIR)
    
        # Image directory 
    image_dir = os.path.join(DATASET_DIR, 'image_train')
    
    #Alterando para diretório das imagens
    os.chdir(image_dir) 
    
    for i, filename in enumerate(filenames):
        filename = os.path.basename(filename)

        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        print(image)
        exit()


if __name__ == '__main__':
    main()