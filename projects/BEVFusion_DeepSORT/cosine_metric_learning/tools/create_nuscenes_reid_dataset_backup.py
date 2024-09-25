from nuscenes.nuscenes import NuScenes

from PIL import Image
import matplotlib.pyplot as plt
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from typing import Tuple
import numpy as np

nusc = NuScenes(version='v1.0-trainval', dataroot='./downloads/datasets/nuscenes', verbose=True)

def render_box(img, axis, view, normalize, colors: Tuple = ('b', 'r', 'k'), linewidth=2):
  def draw_rect(selected_corners, color):
    prev = selected_corners[-1]
    for corner in selected_corners:
        axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
        prev = corner

  #Replicando código do render para desenhar um quadrado que cubra todo o objeto
  corners = view_points(box.corners(), view, normalize=normalize)[:2, :]
  #Obtendo os cantos <X=A, >X=B, <Y=C e >Y=D
  A = np.min(corners[0,:])
  B = np.max(corners[0,:])
  C = np.min(corners[1,:])
  D = np.max(corners[1,:])
  edges = np.array([[A, C], [A, D], [B, D], [B, C]])
  draw_rect(edges, colors[0])

  #Corta imagem na bounding box
  img = img.crop((A, C, B, D))
  axis.imshow(img)

#Exemplo executado no Google Colab
my_scene = nusc.scene[0]
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)
my_annotation_token = my_sample['anns'][1]
#######################
ann = nusc.get('sample_annotation', my_annotation_token)
sample = nusc.get('sample', ann['sample_token'])
print(my_sample)
print()
print(sample)
exit()
#data = nusc.get_sample_data(sample['data']['CAM_FRONT'])
data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample['data']['CAM_FRONT'])
index = 10
print(boxes[index])
print(boxes[10].corners())
box = boxes[index]
img = Image.open(data_path)
_, ax = plt.subplots(1, 1, figsize=(9, 16))

ax.imshow(img)
#boxes[10].render(ax, view=camera_intrinsic, normalize=True, colors=('0.5','0.5','0.5'))

render_box(img, ax, camera_intrinsic, True, colors=('0.5','0.5','0.5'))

#nusc.render_annotation(boxes[index].token)
#nusc.render_sample_data(sample['data']['CAM_FRONT'])