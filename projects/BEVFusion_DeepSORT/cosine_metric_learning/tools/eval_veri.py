import tensorflow.compat.v1 as tf
import os

tf.disable_v2_behavior()
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

#Reduzir warnings não essenciais
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = "2"

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

tf.disable_eager_execution()

def eval_veri():
    dataset_dir = './downloads/datasets/veri'
    query_folder = 'image_query'
    test_folder = 'image_test'
    
    #Obtendo imagem do folder image_query e buscando equivalentes no folder image_test
    folder = os.path.join(dataset_dir, query_folder)
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f))]
    print(len(files))
    for i, f in enumerate(files):
        print(f)
        if i == 10:
            break
        
        
if __name__ == '__main__':
    eval_veri()