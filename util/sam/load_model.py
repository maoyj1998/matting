from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.coco import CocoDataset
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS


from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

from models.sghm.model.model import HumanSegment, HumanMatting

import torch
import yaml
import mmengine

# from utils.sam.util import *
from segment_anything import sam_model_registry, SamPredictor



def build_model(config=None):
    
    if config is None:
        config = config_register['default']
    else:
        config = yaml.safe_load(open(config))
    
    config_files = config['config_file']
    checkpoint_files = config['checkpoints']



    detector_config = config_files['detector']
    kptnet_config = config_files['kptnet']
    sam_config = config_files['sam']
    

    detector_checkpoint = checkpoint_files['detector']
    kptnet_checkpoint = checkpoint_files['kptnet']
    sam_checkpoint = checkpoint_files['sam']
    matting_checkpoint = checkpoint_files['matting']


    detector = init_detector(detector_config,detector_checkpoint,device='cpu')
    kptnet = init_pose_estimator(kptnet_config,kptnet_checkpoint,device='cpu')
    sam = sam_model_registry[sam_config](checkpoint=sam_checkpoint)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    
    mattingnet = HumanMatting(backbone='resnet50').eval()
    mattingnet.load_state_dict(torch.load(matting_checkpoint))



    return detector, kptnet, sam, mattingnet



model_registry = {
    'default': build_model
}

config_register = {
    'default':{

        'config_file':{
            'detector':'./configs/mmdet_config/yolox/yolox_l_8xb8-300e_coco.py',
            'kptnet':'./configs/mmpose_config/body_2d_keypoint/topdown_heatmap/coSco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
            'sam':'vit_h'
        },


        'checkpoints':{
            'detector':'./checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
            'kptnet':'./checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
            'sam':'checkpoints/sam_model/sam_vit_h_4b8939.pth'
        }
    }

}







