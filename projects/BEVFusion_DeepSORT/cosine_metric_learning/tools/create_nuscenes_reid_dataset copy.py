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
nusc = NuScenes(version='v1.0-trainval', dataroot=DATASET_PATH, verbose=True)

class LSSTransformModified(LiftSplatShoot):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LSSTransformModified, self).__init__(grid_conf, data_aug_conf, outC)
    


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


def is_fully_visible(box: Box, image_size: tuple) -> bool:
    corners = view_points(box.corners(), np.eye(4), normalize=True)[:2, :]
    min_corner = corners.min(axis=1)
    max_corner = corners.max(axis=1)
    if np.any(min_corner < 0) or np.any(max_corner > np.array(image_size)):
        return False
    return True

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

def LSS_Transform(img, rot, tran, intrin, post_rot, post_tran):  
    
    def sample_augmentation(data_aug_conf):
        H, W = data_aug_conf['H'], data_aug_conf['W']
        fH, fW = data_aug_conf['final_dim']

        resize = max(fH/H, fW/W)
        resize_dims = (int(W*resize), int(H*resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(data_aug_conf['bot_pct_lim']))*newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        return resize, resize_dims, crop, flip, rotate
      
    imgs = []
    rots = []
    trans = []
    intrins = []
    post_rots = []
    post_trans = []
    
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
    
    # augmentation (resize, crop, horizontal flip, rotate)
    resize, resize_dims, crop, flip, rotate = sample_augmentation(data_aug_conf)
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
    
    model = compile_model(grid_conf, data_aug_conf, outC=64)
    #Path to pre-trained model
    #modelf = '/lasi/cosine_metric_learning/tools/lss_transform/models/1channel/model525000.pt'
    #modelf = '/lasi/cosine_metric_learning/tools/lss_transform/models/3channel/model730000.pt'
    modelf = '/lasi/cosine_metric_learning/tools/lss_transform/models/64channel/model670000.pt'
    model.load_state_dict(torch.load(modelf))
    model.eval()
    
    #Adiciona dimensão que representa o número de câmeras, visto que foi utilizada apenas
    #uma câmera
    out = model(torch.stack(imgs).unsqueeze(0), torch.stack(rots).unsqueeze(0), 
                torch.stack(trans).unsqueeze(0), torch.stack(intrins).unsqueeze(0), 
                torch.stack(post_rots).unsqueeze(0), torch.stack(post_trans).unsqueeze(0))
    
    return out

def BEV_transform(image:MatLike, cam_intrinsics:np.array):
    import torch 
    from from_bevfusion.depth_lss import LSSTransform, DepthLSSTransform
    from mmdet.models.backbones import SwinTransformer
    from from_bevfusion.bevfusion_necks import GeneralizedLSSFPN
    
    print(f'Original shape: {image.shape}')    
    image = cv2.resize(image, (704,256))
    print(f'New shape: {image.shape}')
    image = np.array(image).astype(np.float32)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Reordenar e adicionar dimensões do batch
    print(image.shape)

    #SwinTransformer
    embed_dims=96
    depths=[2, 2, 6, 2]
    num_heads=[3, 6, 12, 24]
    window_size=7
    mlp_ratio=4
    qkv_bias=True
    qk_scale=None
    drop_rate=0.0
    attn_drop_rate=0.0
    drop_path_rate=0.2
    patch_norm=True
    out_indices=[1, 2, 3]
    with_cp=False
    convert_weights=True
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        '/home/lasi/Documents/bruno/bevfusion/bevfusion-mmdet3d/mmdetection3d/work_dirs/swint-nuimages-pretrained/swint-nuimages-pretrained.pth'
    )
    backbone = SwinTransformer(embed_dims=embed_dims, depths=depths, num_heads=num_heads, window_size=window_size,
                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                           attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, patch_norm=patch_norm,
                           out_indices=out_indices, with_cp=with_cp, convert_weights=convert_weights,
                           init_cfg=init_cfg)
    
    backbone.eval()
    with torch.no_grad():
        swin_features = backbone(image)
        # Combinar todas as saídas do FPN
        #combined_features = torch.cat(swin_features, dim=1)  # Concatena no eixo de canais
        
    print('\nBackbone: SwinTransformer')
    for i, feature in enumerate(swin_features):
        print(f"Saída do bloco {i}: shape {feature.shape}")
        
    #Image Neck
    in_channels=[192, 384, 768]
    out_channels=256
    start_level=0
    num_outs=3
    norm_cfg=dict(type='BN2d', requires_grad=True)
    act_cfg=dict(type='ReLU', inplace=True)
    upsample_cfg=dict(mode='bilinear', align_corners=False)
    neck = GeneralizedLSSFPN(in_channels=in_channels, out_channels=out_channels, start_level=start_level,
                             num_outs=num_outs, norm_cfg=norm_cfg, act_cfg=act_cfg,
                             upsample_cfg=upsample_cfg)
    neck.eval()
    image_neck = neck(swin_features)
    # Combinar todas as saídas do FPN
    print('\nNeck: GeneralizedLSSFPN')
    for i, feature in enumerate(image_neck):
        print(f"Saída do bloco {i}: shape {feature.shape}")

    #LSSTransform
    d = (60-1)/0.5
    in_channels=256
    out_channels=80
    image_size=[128, 352]
    feature_size=[32, 88]
    xbound=[-54.0, 54.0, 0.3]
    ybound=[-54.0, 54.0, 0.3]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[1.0, 60.0, 0.5]
    downsample=2
    
    
    lss_transform = LSSTransform(in_channels, out_channels, image_size, feature_size,
                             xbound, ybound, zbound, dbound, downsample=downsample)

    points = None
    # lidar2image = torch.eye(4).unsqueeze(0)  # (batch size, 4, 4)
    # camera2lidar = torch.eye(4).unsqueeze(0)  # (batch size, 4, 4)
    # img_aug_matrix = torch.eye(4).unsqueeze(0)  # (batch size, 4, 4)
    # lidar_aug_matrix = torch.eye(4).unsqueeze(0)  # (batch size, 4, 4)
    
    lidar2image = torch.eye(4).unsqueeze(0)  # (batch size, 4, 4)
    camera2lidar = torch.eye(4).unsqueeze(0).unsqueeze(0)  # (batch size, num_cams, 4, 4)
    img_aug_matrix = torch.eye(4).unsqueeze(0)  # (batch size, num_cams, 4, 4)
    lidar_aug_matrix = torch.eye(4).unsqueeze(0)  # (batch size, 4, 4)
    camera2lidar_trans = camera2lidar[..., :3, 3]

    # Metadados (mockados)
    metas = [{
        "img_shape": (256, 256, 3),
        "lidar2img": lidar2image,
        "lidar2cam": camera2lidar,
        "cam2img": cam_intrinsics,
    }]

    lss_transform.eval()
    
    print("Shape do tensor antes do reshape:", image_neck[-1].shape)

    # Verifique o número de elementos no tensor
    num_elements = image_neck[-1].numel()
    print(f"Número de elementos no tensor: {num_elements}")

    # Comparar com o número de elementos necessário para o shape desejado
    target_shape = (672600, 3)
    required_elements = target_shape[0] * target_shape[1]

    # if num_elements != required_elements:
    #     raise ValueError(f"Shape desejado {target_shape} é incompatível com o número de elementos {num_elements}. São necessários {required_elements} elementos.")

    # Ajuste o reshape de acordo com as dimensões corretas
    #img = image_neck[-1].view(target_shape)
    
    print(image_neck[-1].shape)
    img = image_neck[-1].unsqueeze(0)
    print(img.shape)
    print(f'Intrinsics: {cam_intrinsics.shape}\n{cam_intrinsics}')
    
    bev_output = lss_transform(
        img=img,
        points=points,
        lidar2image=lidar2image,
        camera_intrinsics=torch.from_numpy(cam_intrinsics).float(),
        camera2lidar=camera2lidar,
        img_aug_matrix=img_aug_matrix,
        lidar_aug_matrix=lidar_aug_matrix,
        metas=metas
    )
    #transform.eval()
    #lss_image = transform.forward(swin_features[-1], cam_intrinsic=cam_intrinsic)
    print("Características no espaço BEV:", bev_output.shape)
    
    # Exemplo de uso
    #img_shape = (image.shape[1], image.shape[0])

    # Inicializa o conversor
    #bev_converter = SimpleBEVConverter(img_shape, intrinsics)
    
    # camera2lidar_rots = img_features[..., :3, :3]
    # camera2lidar_trans = img_features[..., :3, 3]
    # bev_image = transform.get_geometry(camera2lidar_rots, camera2lidar_trans, intrinsics,
                                    #    camera2lidar_rots, camera2lidar_trans)

    # Converte para BEV
    #bev_image = bev_converter.lift_splat_shoot(image)
    #print("Imagem BEV gerada:", bev_image.shape)
    return bev_output

def extract_images(nusc:NuScenes, output_dir='output', image_shape=(256,704)):
    os.makedirs(output_dir, exist_ok=True)
    object_dict = {}
    
    total = len(nusc.sample)
    for i, sample in enumerate(nusc.sample):
        print(f'Iniciando sample {i} de {total}')
        if len(object_dict.keys()) >= 20:
            break
        for sensor_name in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            cam_data = nusc.get('sample_data', sample['data'][sensor_name])
            cam_intrinsics = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['camera_intrinsic']
            original_img_path = os.path.join(DATASET_PATH, cam_data['filename'])
            image = cv2.imread(original_img_path)
            image_size = (image.shape[1], image.shape[0])  # (width, height)
                        
            ann_tokens = sample['anns']
            for ann_token in ann_tokens:
                ann = nusc.get('sample_annotation', ann_token)
                instance_token = ann['instance_token']
                
                visibility_token = int(ann['visibility_token'])
                
                if visibility_token < 4:
                    continue  # Ignora objetos que não tenham "boa" visibilidade
                
                category_name = ann['category_name']
                
                # if instance_token in object_dict:
                #     if len(object_dict[instance_token]['images']) == 5:
                #         #Quando já houverem 5 imagens para o objeto, ignora o mesmo
                #         continue
                
                if 'vehicle' in category_name or 'pedestrian' in category_name:
                    data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample['data'][sensor_name], BoxVisibility.ALL, [ann['token']])
                    
                    if len(boxes) == 0:
                        continue
                    
                    box = boxes[0]
                    
                    if is_fully_visible(box, image_size):
                        #Replicando código do render para desenhar um quadrado que cubra todo o objeto
                        corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
                        #Obtendo os cantos <X=A, >X=B, <Y=C e >Y=D
                        A = np.min(corners[0,:])
                        B = np.max(corners[0,:])
                        C = np.min(corners[1,:])
                        D = np.max(corners[1,:])   
                        
                        def box_to_bev():
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
                            
                            dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
                            dx, bx, nx = dx.numpy(), bx.numpy(), nx.numpy()
                            egopose = nusc.get('ego_pose',
                                nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])
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

                        box_bev = box_to_bev()
                        
                        #Corta imagem na bounding box
                        #cropped_img = image.crop((A, C, B, D))
                        x_min, y_min = int(A), int(C)
                        x_max, y_max = int(B), int(D)
                        cropped_img = highlight_image_frame(image, x_min, x_max, y_min, y_max)
                        #cropped_img = image[y_min:y_max, x_min:x_max]
                        area = (x_max - x_min)*(y_max - y_min)
                        
                        #bev_image = BEV_transform(cropped_img, np.array(camera_intrinsic))
                        sens = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                        pil_img = Image.open(original_img_path)                    
                        rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
                        tran = torch.Tensor(sens['translation'])
                        intrins = torch.Tensor(camera_intrinsic)
                        post_rot = torch.eye(2)
                        post_tran = torch.zeros(2)
                        bev_image = LSS_Transform(pil_img, rot, tran, intrins, post_rot, post_tran)                      
                        
                        if len(cropped_img) == 0 or area < 1000:
                            continue
                        
                        #Salva arquivo com base no instance_token, que identifica o objeto                     
                        if instance_token not in object_dict:
                            object_dict[instance_token] = {
                                'images': [],
                                'intrinsics': [],
                                'category': category_name
                            }                     
                                                  
                        image_filename = f"{instance_token}_{len(object_dict[instance_token]['images'])}.jpg"
                        image_path = os.path.join(output_dir, image_filename)
                        
                        #Salvando imagem com matplotlib
                        val = 0.01
                        final_dim=(128, 352)
                        fH, fW = final_dim
                        fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
                        gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
                        gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
                        
                        def plot_image(img, filename, bbox=None, rgb=False):
                            def draw_rect(ax, selected_corners):
                                prev = selected_corners[-1]
                                for corner in selected_corners:
                                    ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color='b', linewidth=2)
                                    prev = corner
                            # with open(os.path.join(output_dir, f"{instance_token}_{len(object_dict[instance_token]['images'])}_{i}_bev.npy"), 'wb') as f:
                            #     conv_layer = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
                            #     x1 = conv_layer(x)
                            #     np.save(f, x1.detach().numpy())
                            plt.clf()
                            ax = plt.subplot(gs[0, :])
                            ax.get_xaxis().set_ticks([])
                            ax.get_yaxis().set_ticks([])
                            
                            if rgb:
                                plt.imshow(img)  
                                draw_rect(ax, bbox) 
                                plt.savefig(filename)
                                plt.close()
                                return                               
                                   
                            conv_layer = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
                            x = conv_layer(img)                            
                            draw_rect(ax, bbox)
                            plt.setp(ax.spines.values(), color='b', linewidth=2)
                            #plt.imshow(x.squeeze(0).detach(), vmin=0, vmax=1, cmap='Blues')
                            plt.imshow(x.squeeze(0).detach(), vmin=-3.9697017669677734, vmax=6.039807319641113, cmap='viridis')
                            
                            #print('saving', filename)
                            plt.savefig(filename)
                            plt.close()
                                                    
                        for i, x in enumerate(bev_image):   
                            if box_bev is not None:     
                                filename = os.path.join(output_dir, f"{instance_token}_{len(object_dict[instance_token]['images'])}_{i}_bev.jpg")                 
                                plot_image(x, filename, box_bev)
    
                        #Salvando imagem com opencv
                        edges = np.array([[A, C], [A, D], [B, D], [B, C]])
                        plot_image(pil_img, image_path, edges, True)
                            
                        #cv2.imwrite(image_path, image)
                        #cv2.imwrite(os.path.join(output_dir, f"{instance_token}_{len(object_dict[instance_token]['images'])}_redim.jpg"), cropped_img)
                        #cv2.imwrite(os.path.join(output_dir, f"{instance_token}_{len(object_dict[instance_token]['images'])}_bev.jpg"), bev_image)
                        
                        object_dict[instance_token]['images'].append(image_path)
                        object_dict[instance_token]['intrinsics'].append(cam_intrinsics)
                        
    #Eliminando objetos com menos de 5 imagens
    total = len(object_dict.keys())
    filtered_dict = {}
    for i, token in enumerate(object_dict):
        print(f'Analisando objeto {i} de {total}')
        if len(object_dict[token]['images']) >= 2:
            filtered_dict[token] = object_dict[token] 
        else:
            #Remove imagens salvas do objeto não adicionado
            for path in object_dict[token]['images']:
                os.remove(path)  
    
    return filtered_dict

# Run the extraction process
object_data = extract_images(nusc, output_dir='./downloads/datasets/nuscenes_reid/images')

# Save the object data dictionary if needed
import json
with open('./downloads/datasets/nuscenes_reid/object_data.json', 'w') as f:
    json.dump(object_data, f, indent=4)