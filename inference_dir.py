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
from tqdm import tqdm
import cv2

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

def gen_matting(img,matting_mask,sam_mask,num_obj,area_threshould,scores):
    matting_mask = np.squeeze(matting_mask)
    scores = scores.squeeze().cpu().numpy()

    masks_fixed = []
    connected = []

    for i,mask in enumerate(sam_mask):
        mask = mask.cpu().squeeze().numpy()
        connect_region = holes_islands(mask)
        connected.append(connect_region)

        mask,_ = remove_small_regions(mask,1000,'holes')
        mask,_ = remove_small_regions(mask,1000,'islands')

        mask = mask.astype('uint8')
        masks_fixed.append(mask)
    
    connected = list(map(conected_score,connected))
    scores += np.array(connected)
    best = scores.argmax()
    


    sam_mask =  masks_fixed[best]

    matting_mask = (matting_mask * 255).astype('uint8')

    if num_obj is None:
        sam_mask = np.ones_like(matting_mask)
    
    matting_mask *= sam_mask
    sam_mask *= 255
    matting_mask = np.where(sam_mask!=0,sam_mask,matting_mask)
    # rgba = sam_mask
    rgba = np.concatenate([img,matting_mask[:,:,None]],axis=-1)
    rgba = Image.fromarray(rgba)
    return rgba
    

def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
):
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def conected_score(connected_region,num_obj=1):

    connected_region = int(connected_region / num_obj)
    if connected_region == 2:
        return 0.03
    elif connected_region == 3:
        return 0.01
    elif connected_region == 4:
        return 0.01
    elif connected_region == 5:
        return 0.001
    else:
        return -0.05

def holes_islands(mask):
    hole = (True ^ mask).astype(np.uint8)
    holes, _, _, _ = cv2.connectedComponentsWithStats(hole, 8)


    island = (False ^ mask).astype(np.uint8)
    islands, _, _, _ = cv2.connectedComponentsWithStats(island, 8)

    # background and the main part(person) are counted twicw
    return holes + islands - 2
if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./demo')
    parser.add_argument('--config',type=str,default='./configs/default.yml')
    parser.add_argument('--model-type',type=str,default='default')
    parser.add_argument('--use_cuda',action='store_true',default=True)
    parser.add_argument('--save_path', type=str, default='./res')
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


    img_dir = args.img_dir
    if not os.path.exists(img_dir):
        raise Exception('Image file does not exist')
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)


    files = os.listdir(img_dir)
    with tqdm(total=len(files)) as pbar:
        for f in files:
            img_path = os.path.join(img_dir,f)
            print(img_path)
            img = mmcv.imread(img_path,channel_order='rgb')
            file_name = os.path.basename(img_path)
            try:
                masks,scores,num_obj,max_area = kpt_sam_interface(img,detector,sam,kptnet,use_box=True,use_nms=True,nms_thresh=0.4,mean_center=True,vis=False,vis_all=False,return_scores=True,use_best_mask=False,return_num=True,box_only=True)

                pred_alpha, pred_mask = inference.single_inference(mattingnet, Image.fromarray(img))
                
                if num_obj == 0:
                    raise RuntimeError('no person in the image')


                rgba = gen_matting(img,pred_alpha,masks,num_obj,max_area*0.3,scores)
                rgba.save(os.path.join(args.save_path,file_name.split('.')[0] + '.png'))
                print("save to: " + os.path.join(args.save_path,file_name.split('.')[0] + '.png'))
            except KeyboardInterrupt as e:
                print(e)
            except:
                print(img_path)
            pbar.update()



# python inference_dir.py --img_dir /mnt/d/data/person/supervisely_person_clean_2667_img/images/ --save_path /mnt/d/data/person/supervisely_person_clean_2667_img/pipeline_connect








    




        
