import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
import cv2
from cv2.typing import MatLike
import numpy as np
from PIL import Image

import torch
from tools.lss_transform.src.tools import normalize_img, img_transform, denormalize_img, gen_dx_bx
from tools.lss_transform.src.models import compile_model, LiftSplatShoot
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
plt.ioff()

# Path to your nuScenes dataset
DATASET_PATH = './downloads/datasets/nuscenes'

# Initialize NuScenes object
#nusc = NuScenes(version='v1.0-trainval', dataroot=DATASET_PATH, verbose=True)

class LSSTransformModified(LiftSplatShoot):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LSSTransformModified, self).__init__(grid_conf, data_aug_conf, outC)
        
    def compile(self):
        self.model = compile_model(self.grid_conf, self.data_aug_conf, outC=64)
        #Path to pre-trained model
        #modelf = '/lasi/cosine_metric_learning/tools/lss_transform/models/1channel/model525000.pt'
        #modelf = '/lasi/cosine_metric_learning/tools/lss_transform/models/3channel/model730000.pt'
        modelf = '/lasi/cosine_metric_learning/tools/lss_transform/models/64channel/model670000.pt'
        self.model.load_state_dict(torch.load(modelf))
        self.model.eval()
        
    def transform(self, imgs, rots, trans, intrins, post_rots, post_trans):
        """ Apply transform to image.        
        """
        out = self.model(imgs, rots, trans, intrins, post_rots, post_trans)
    
        return out

class NuscenesReId():
    def __init__(self, data_aug_conf, grid_conf):
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=DATASET_PATH, verbose=True)       
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']

        resize = max(fH/H, fW/W)
        resize_dims = (int(W*resize), int(H*resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        return resize, resize_dims, crop, flip, rotate
    
    def is_fully_visible(self, box: Box, image_size: tuple) -> bool:
        corners = view_points(box.corners(), np.eye(4), normalize=True)[:2, :]
        min_corner = corners.min(axis=1)
        max_corner = corners.max(axis=1)
        if np.any(min_corner < 0) or np.any(max_corner > np.array(image_size)):
            return False
        return True
    
    def box_to_bev(self, sample, ann): 
        """ Transform bbox coordinates of ann from image view to bev, using sample data to get
            the coordinates of the ego (center of view).
        """   
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound'])
        dx, bx, nx = dx.numpy(), bx.numpy(), nx.numpy()
        egopose = self.nusc.get('ego_pose',
            self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        box.translate(trans)
        box.rotate(rot)

        pts = box.bottom_corners()[:2].T
        pts = np.round(
            (pts - bx[:2] + dx[:2]/2.) / dx[:2]
            ).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]                            

        return pts  
    
    def get_image_data(self, cam_data):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []    
        
        imgname = os.path.join(DATASET_PATH, cam_data['filename'])
        img = Image.open(imgname)
        
        sens = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
        tran = torch.Tensor(sens['translation'])
        intrin = torch.Tensor(sens['camera_intrinsic'])
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        
        # augmentation (resize, crop, horizontal flip, rotate)
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
        img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                resize=resize,
                                                resize_dims=resize_dims,
                                                crop=crop,
                                                flip=flip,
                                                rotate=rotate,
                                                )
        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2
        
        imgs.append(normalize_img(img))
        intrins.append(intrin)
        rots.append(rot)
        trans.append(tran)
        post_rots.append(post_rot)
        post_trans.append(post_tran)
        
        #Adiciona dimensão que representa o número de câmeras, visto que foi utilizada apenas
        #uma câmera
        return(torch.stack(imgs).unsqueeze(0), torch.stack(rots).unsqueeze(0), 
                    torch.stack(trans).unsqueeze(0), torch.stack(intrins).unsqueeze(0), 
                    torch.stack(post_rots).unsqueeze(0), torch.stack(post_trans).unsqueeze(0))
        
    def save_to_file(self, feat, filename, bbox=None, rgb=False):
        """ Cut features tensor (or image) in bbox and save to .npy (numpy) or .jpg file.
        """
        def draw_rect(ax, selected_corners):
            prev = selected_corners[-1]
            for corner in selected_corners:
                ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color='b', linewidth=2)
                prev = corner
                
        val = 0.01
        final_dim=(128, 352)
        fH, fW = final_dim
        fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
        gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
        gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
        

        
        if rgb:   
            A = np.min(bbox[0,:])
            B = np.max(bbox[0,:])
            C = np.min(bbox[1,:])
            D = np.max(bbox[1,:]) 
            x_min, y_min = int(A), int(C)
            x_max, y_max = int(B), int(D)
            
            plt.clf()
            ax = plt.subplot(gs[0, :])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            
            #cropped_img = feat.crop((A, C, B, D))         
            plt.imshow(feat)  
            draw_rect(ax, bbox) 
            plt.savefig(filename)
            plt.close()
            return      
        else:
            #Getting bbox coordinates
            print(bbox)
            A = np.min(bbox[:,0])
            B = np.max(bbox[:,0])
            C = np.min(bbox[:,1])
            D = np.max(bbox[:,1]) 
            x_min, y_min = int(A), int(C)
            x_max, y_max = int(B), int(D)
            print(f'\nx_min = {x_min}\tx_max = {x_max}\ny_min = {y_min}\ty_max = {y_max}')
            
            #Cutting features tensor or image in bbox coordinates
            cropped_feat = feat[:, x_min:x_max, y_min:y_max]
            print(cropped_feat.shape)
            #Saving tensor to .npy
            with open(filename, 'wb') as f:
                np.save(f, cropped_feat.detach().numpy())
            # with open(filename.replace('.npy', '_full.npy'), 'wb') as f:
            #     np.save(f, feat.detach().numpy())
                
            plt.clf()
            ax = plt.subplot(gs[0, :])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            # conv_layer = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
            # x = conv_layer(feat)  
            min_dataset = -3.9697017669677734
            max_dataset = 6.039807319641113 
            x = feat.detach().numpy()
            x = (x-min_dataset) / (max_dataset - min_dataset)   
            X, Y, Z = np.nonzero(x)
            nonzero = x[np.nonzero(x)]
            c = nonzero.flatten()                      
            draw_rect(ax, bbox)
            
            scatter = ax.scatter(Y, Z, c=c, cmap='viridis', marker='o')
            ax.scatter([99], [99], color='r', marker='x')
            
            # plt.setp(ax.spines.values(), color='b', linewidth=2)
            # #plt.imshow(x.squeeze(0).detach(), vmin=0, vmax=1, cmap='Blues')
            # plt.imshow(x.squeeze(0).detach(), vmin=-3.9697017669677734, vmax=6.039807319641113, cmap='viridis')
            
            #print('saving', filename)
            plt.savefig(filename.replace('.npy', '_full.jpg'))
            plt.close()
            
            return                  
        
        
    
def extract_images(output_dir='output', image_shape=(256,704)):
    """ Processa imagens do Nuscenes para converter em BEV, identificar por objeto e exportar imagens
        e tensor de saída (formato width x height x 64)
    """
    
    H=900
    W=1600
    resize_lim=(0.193, 0.225)
    final_dim=(128, 352)
    bot_pct_lim=(0.0, 0.22)
    rot_lim=(-5.4, 5.4)
    rand_flip=True

    xbound=[-50.0, 50.0, 0.5]
    ybound=[-50.0, 50.0, 0.5]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[4.0, 45.0, 1.0]
    
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': cams,
        'Ncams': 5,
    }
    
    nusc = NuscenesReId(data_aug_conf=data_aug_conf, grid_conf=grid_conf)
    model = LSSTransformModified(grid_conf=grid_conf, data_aug_conf=data_aug_conf, outC=64)
    model.compile()
    
    os.makedirs(output_dir, exist_ok=True)
    object_dict = {}
    
    total = len(nusc.nusc.sample)
    for i, sample in enumerate(nusc.nusc.sample):
        print(f'Iniciando sample {i} de {total}')
        if len(object_dict.keys()) >= 10:
            break
        for sensor_name in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            cam_data = nusc.nusc.get('sample_data', sample['data'][sensor_name])
            cam_intrinsics = nusc.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['camera_intrinsic']
            original_img_path = os.path.join(DATASET_PATH, cam_data['filename'])
            image = cv2.imread(original_img_path)
            image_size = (image.shape[1], image.shape[0])  # (width, height)
                        
            ann_tokens = sample['anns']
            for ann_token in ann_tokens:
                ann = nusc.nusc.get('sample_annotation', ann_token)
                instance_token = ann['instance_token']
                
                visibility_token = int(ann['visibility_token'])
                
                if visibility_token < 4:
                    continue  # Ignora objetos que não tenham "boa" visibilidade
                
                category_name = ann['category_name']
                
                if 'vehicle' in category_name or 'pedestrian' in category_name:
                    data_path, boxes, camera_intrinsic = nusc.nusc.get_sample_data(sample['data'][sensor_name], BoxVisibility.ALL, [ann['token']])
                    
                    if len(boxes) == 0:
                        continue
                    
                    box = boxes[0]
                    
                    if nusc.is_fully_visible(box, image_size):
                        #Replicando código do render para desenhar um quadrado que cubra todo o objeto
                        corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
                        #Obtendo os cantos <X=A, >X=B, <Y=C e >Y=D
                        A = np.min(corners[0,:])
                        B = np.max(corners[0,:])
                        C = np.min(corners[1,:])
                        D = np.max(corners[1,:])                          
                                   
                        box_bev = nusc.box_to_bev(sample, ann)
                        
                        #Corta imagem na bounding box
                        #cropped_img = image.crop((A, C, B, D))
                        x_min, y_min = int(A), int(C)
                        x_max, y_max = int(B), int(D)
                        cropped_img = highlight_image_frame(image, x_min, x_max, y_min, y_max)
                        cropped_img = image[y_min:y_max, x_min:x_max]
                        area = (x_max - x_min)*(y_max - y_min)
                        
                        imgs, rots, trans, intrins, post_rots, post_trans = nusc.get_image_data(cam_data=cam_data)
                        bev_image = model.transform(imgs, rots, trans, intrins, post_rots, post_trans)                     
                        
                        if len(cropped_img) == 0 or area < 1000:
                            continue
                        
                        #Salva arquivo com base no instance_token, que identifica o objeto                     
                        if instance_token not in object_dict:
                            object_dict[instance_token] = {
                                'images': [],
                                'intrinsics': [],
                                'category': category_name,
                                'bbox': []
                            }                     
                                                
                        image_filename = f"{instance_token}_{len(object_dict[instance_token]['images'])}.jpg"
                        image_path = os.path.join(output_dir, image_filename)
                                                                            
                        for i, x in enumerate(bev_image): 
                            #print(x.shape)  
                            if box_bev is not None:     
                                filename = os.path.join(output_dir, f"{instance_token}_{len(object_dict[instance_token]['images'])}_{i}_bev.npy")                 
                                nusc.save_to_file(x, filename, box_bev)
    
                        
                        edges = np.array([[A, C], [A, D], [B, D], [B, C]])
                        pil_img = Image.open(original_img_path) 
                        nusc.save_to_file(pil_img, image_path, edges, True)
                            
                        #Salvando imagem com opencv
                        #cv2.imwrite(image_path, image)
                        #cv2.imwrite(os.path.join(output_dir, f"{instance_token}_{len(object_dict[instance_token]['images'])}_redim.jpg"), cropped_img)
                        #cv2.imwrite(os.path.join(output_dir, f"{instance_token}_{len(object_dict[instance_token]['images'])}_bev.jpg"), bev_image)
                        
                        object_dict[instance_token]['images'].append(image_path)
                        object_dict[instance_token]['intrinsics'].append(cam_intrinsics)
                        object_dict[instance_token]['bbox'].append(box_bev.tolist())
                        
    #Eliminando objetos com menos de 5 imagens
    total = len(object_dict.keys())
    filtered_dict = {}
    for i, token in enumerate(object_dict):
        print(f'Analisando objeto {i} de {total}')
        if len(object_dict[token]['images']) >= 4:
            filtered_dict[token] = object_dict[token] 
        else:
            #Remove imagens salvas do objeto não adicionado
            rm_list = [x for x in os.listdir(output_dir) if x.startswith(token)]
            for item in rm_list:    
                path = os.path.join(output_dir, item)            
                os.remove(path)  
    
    return filtered_dict




def resize_image_without_distortion(img, largura_destino, altura_destino):
    # Obtém as dimensões da imagem original
    altura_original, largura_original = img.shape[:2]

    # Calcula a razão de aspecto da imagem original e da imagem de destino
    razao_original = largura_original / altura_original
    razao_destino = largura_destino / altura_destino

    if razao_original > razao_destino:
        # A imagem é mais larga do que a proporção de destino
        nova_largura = largura_destino
        nova_altura = int(largura_destino / razao_original)
    else:
        # A imagem é mais alta do que a proporção de destino
        nova_altura = altura_destino
        nova_largura = int(altura_destino * razao_original)

    # Redimensiona a imagem mantendo a razão de aspecto original
    img_redimensionada = cv2.resize(img, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)

    # Calcula os valores de preenchimento (padding) para centralizar a imagem
    padding_horizontal = (largura_destino - nova_largura) // 2
    padding_vertical = (altura_destino - nova_altura) // 2

    # Aplica o preenchimento para alcançar as dimensões de destino
    img_com_padding = cv2.copyMakeBorder(img_redimensionada, 
                                         padding_vertical, altura_destino - nova_altura - padding_vertical, 
                                         padding_horizontal, largura_destino - nova_largura - padding_horizontal, 
                                         cv2.BORDER_CONSTANT, 
                                         value=[0, 0, 0])  # Preenche com preto, mas pode usar outra cor

    return img_com_padding

def highlight_image_frame(img:MatLike, x_min:int, x_max:int, y_min:int, y_max:int):
    highlighted_image = np.zeros((img.shape[0], img.shape[1],3))
    highlighted_image[y_min:y_max, x_min:x_max] = img[y_min:y_max, x_min:x_max]
    return highlighted_image


if __name__ == '__main__':    
    # Run the extraction process
    object_data = extract_images(output_dir='./downloads/datasets/nuscenes_reid/images')

    # Save the object data dictionary if needed
    import json
    with open('./downloads/datasets/nuscenes_reid/object_data.json', 'w') as f:
        json.dump(object_data, f, indent=4)