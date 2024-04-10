"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from fire import Fire

import tools.lss_transform.src as src
from src.explore import lidar_check, cumsum_check, eval_model_iou, viz_model_preds
from src.train import train


if __name__ == '__main__':
    Fire({
        'lidar_check': lidar_check,
        'cumsum_check': cumsum_check,

        'train': train,
        'eval_model_iou': eval_model_iou,
        'viz_model_preds': viz_model_preds,
    })