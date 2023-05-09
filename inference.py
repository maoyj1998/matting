from util.sam.model_interface import kpt_sam_interface
from util.sam.load_model import model_registry
from util.sam.visual import *
from segment_anything import SamPredictor

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.coco import CocoDataset
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from PIL import Image

from tqdm import tqdm
import os
import argparse
from PIL import Image
from matplotlib import pyplot as plt
import torch
import time
import mmcv
import numpy as np
from models.sghm import inference
import models.sghm.utils

def seed():
    torch.manual_seed(1234) 
    torch.cuda.manual_seed(1234) 
    torch.cuda.manual_seed_all(1234)  
    np.random.seed(1234)  


def save_masks(masks,scores,save_path,file_name='demo',save_score=False,save_best=False,save_rgb=False,):

    for i,mask in enumerate(masks):
        mask = mask.cpu().squeeze().numpy().astype(np.uint8)
        mask *= 255
        mask = Image.fromarray(mask) 
        if save_score:
            mask.save(os.path.join(save_path,file_name+'_'+str(i+1)+'_%.2f'%(scores[i].cpu().numpy().tolist())+'.png'))
        else:
            mask.save(os.path.join(save_path,file_name+'_'+str(i+1)+'.png'))


    if save_best:

        if masks.shape[0] == 3:
            best = scores.argmax()
            show_best_mask =  masks[best].cpu().numpy()
            best_mask = masks[best].cpu().numpy().astype(np.uint8)
        else:
            best_mask = masks[0].cpu().numpy().astype(np.uint8)
        

        best_mask *= 255
        best_mask = Image.fromarray(best_mask) 
        best_mask.save(os.path.join(save_path,file_name+'_best'+'.png'))

    if save_rgb:
        plt.figure(figsize=(15,15))
        plt.imshow(img)
        show_mask(show_best_mask,plt.gca(),random_color=True)
        plt.axis('off')
        plt.savefig(os.path.join(save_path,file_name+'_rgb'+'.png'))
        plt.close()

def gen_matting(img,matting_mask,sam_mask):
    matting_mask = np.squeeze(matting_mask)
    sam_mask = sam_mask.squeeze().cpu().numpy().astype('uint8')
    matting_mask = (matting_mask * 255).astype('uint8')
    
    matting_mask *= sam_mask

    rgba = np.concatenate([img,matting_mask[:,:,None]],axis=-1)
    rgba = Image.fromarray(rgba)
    return rgba
    


if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./demo.jpg')
    parser.add_argument('--config',type=str,default='./configs/default.yml')
    parser.add_argument('--model-type',type=str,default='default')
    parser.add_argument('--use_cuda',action='store_true',default=True)
    parser.add_argument('--save_path', type=str, default='./saved_path')
    parser.add_argument('--save_seg_masks', action='store_true', default=False)


    seed()

    args = parser.parse_args()
    config = args.config
    model_type = args.model_type
    use_cuda = args.use_cuda
    save_seg_masks = args.save_seg_masks
    device = 'cpu'




    detector, kptnet, sam, mattingnet = model_registry[model_type](config)

    if use_cuda:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    detector = detector.to(device)
    kptnet = kptnet.to(device)
    sam = sam.to(device)
    sam = SamPredictor(sam)
    mattingnet = mattingnet.to(device)




    img_path = args.img_path
    if not os.path.exists:
        raise Exception('Image file does not exist')
    

    img = mmcv.imread(img_path,channel_order='rgb')
    file_name = os.path.basename(img_path)


    masks,scores,num_obj = kpt_sam_interface(img,detector,sam,kptnet,use_box=True,use_nms=True,nms_thresh=0.4,mean_center=True,vis=False,vis_all=False,return_scores=True,use_best_mask=True)
    # save_masks(masks,scores,args.save_path,file_name,save_masks,save_masks,save_masks)
    pred_alpha, pred_mask = inference.single_inference(mattingnet, Image.fromarray(img))
    
    rgba = gen_matting(img,pred_alpha,masks,num_obj)
    rgba.save('./rgba.png')
    print(1)











    




        
