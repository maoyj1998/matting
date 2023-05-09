from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.coco import CocoDataset
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS


import mmengine

# from utils.sam.util import *
from segment_anything import sam_model_registry, SamPredictor



from matplotlib import pyplot as plt


import numpy as np
import torch

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
import cv2

from functools import cmp_to_key


class mmres2sam(object):
    
    def __init__(self,mmres,threshould=0.8):
        if 'cuda' in str(mmres.pred_instances.bboxes.device):
            self.bboxes = mmres.pred_instances.bboxes.cpu().numpy()
            self.scores = mmres.pred_instances.scores.cpu().numpy()
            self.threshould = threshould
            self.labels = mmres.pred_instances.labels.cpu().numpy()
        else:
            self.bboxes = mmres.pred_instances.bboxes.numpy()
            self.scores = mmres.pred_instances.scores.numpy()
            self.threshould = threshould
            self.labels = mmres.pred_instances.labels.numpy()
    
    
    def preprocess(self,select_label=None):
        '''
            label 0 : person
        
        '''
        if select_label is None:
            return self.bboxes
        
        
        
        boxes = []
        for box, score, label in zip(self.bboxes,self.scores,self.labels):
            if label == select_label and score >= self.threshould:
#                 print(label,score)
                boxes.append(box)
        
        return np.array(boxes)
    





def gen_coord(box):
    coords = []
    width = box[2] - box[0]
    height = box[3] - box[1]
    # print(height)
    center = [box[0] + int(width/2), box[1] + int(height / 2)]
    coords.append(center)



    upper = center[1] - height / 2 * 0.35
    upper = int(upper)
    bound = center[1] + height / 2 * 0.35
    bound = int(bound) + 10
    # print(upper)
    # print(bound)
    gap = int((bound - upper)/2)
    
    # print(gap)
    for i in range(upper,bound,gap):
        # print(i)
        coords.append([center[0],i])
    return np.array(coords)




def mm_sam_all(image,det_model,predictor,point=False):
    result = inference_detector(det_model, image)

    input_point = None
    input_label = None
    trasnformed_coors = None

    det_res = mmres2sam(result,threshould=0.6)
    input_box = det_res.preprocess(select_label=0)


    predictor.set_image(image)
    

    if point:
        input_point = []
        for box in input_box:
            input_point.extend(gen_coord(box).tolist())
        input_point = np.array(input_point)    
        input_label = np.array([1]*len(input_point))



    
    if len(input_box) == 0:
        return torch.zeros([1,1,image.shape[0],image.shape[1]])
    
    if input_box.ndim == 1:
        input_box = input_box[None,:]
    
#     print(input_box.shape)
    input_box = torch.tensor(input_box).to(predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_box, image.shape[:2])

    if point:
        # input_point = torch.tensor(input_point).to(predictor.device)
        # trasnformed_coors = predictor.transform.apply_coords_torch(input_point,image.shape[:2])
        pass
    
#     masks, _, _ = predictor.predict(
#     point_coords=trasnformed_coors,
#     point_labels=input_label,
#     box=input_box,
#     multimask_output=True,
# )



    masks, _, _ = predictor.predict_torch(
        point_coords=trasnformed_coors,
        point_labels=input_label,
        boxes=transformed_boxes,
        multimask_output=True,
    )
    
    # if type(masks) == torch.tensor:
    try:
        return masks[:,0,...]
    except: 
        return torch.from_numpy(masks[0,...]).unsqueeze(0)



class mmpose2sam(object):
    
    def __init__(self,mmres,box_threshould=0.8,kpt_box_threshould=0.6,img_shape=None,nms_thresh=0.4,mean_center=False):
        if type(mmres) == mmengine.structures.instance_data.InstanceData:
            
            self.boxes = mmres.bboxes
            self.box_scores = mmres.bbox_scores

            self.kpts = mmres.keypoints
            self.kpt_scores = mmres.keypoint_scores
            
        else:
            try:
                self.boxes = mmres.pred_instances.bboxes.cpu().numpy()
                self.box_scores = mmres.pred_instances.bbox_scores.cpu().numpy()
                self.kpts = mmres.pred_instances.keypoints.cpu().numpy()
                self.kpt_scores = mmres.pred_instances.keypoint_scores.cpu().numpy()


            except:
                self.boxes = mmres.pred_instances.bboxes
                self.box_scores = mmres.pred_instances.bbox_scores

                self.kpts = mmres.pred_instances.keypoints
                self.kpt_scores = mmres.pred_instances.keypoint_scores

        
        
        self.box_threshould = box_threshould
        self.kpt_threshould = kpt_box_threshould
        self.img_width = img_shape[0]
        self.img_height = img_shape[1]
        self.thresh = nms_thresh
        self.mean_center = mean_center

    def preprocess(self,point_nms=False):
        
        # inner bbox
        if point_nms:
            self.nms_kpts()
        

        boxes = []
        kpts = []
        scores = []
        
        for i, bs in enumerate(self.box_scores):
            if bs >= self.box_threshould:
                
                box = self.boxes[i]
                boxes.append(box)
                temp = []
                
                for j, ks in enumerate(self.kpt_scores[i]):
                    if ks >= self.kpt_threshould:
                        
                        pt = self.kpts[i][j]
                        
                        if pt[0] < 0 or pt[1] < 0 or pt[1] > self.img_width or pt[0] > self.img_height:
                            continue
                        
                        temp.append(pt)
                        scores.append(ks)
                kpts.extend(temp)
        if len(kpts) == 0:
            return np.array([]), np.array([])
            
        if point_nms: # across bbox
            kpts,scores = self.nms_kpts_single(np.array(kpts),np.array(scores),[0,0,self.img_width,self.img_height],thresh=0.05)
       
        if self.mean_center:
            center = [[np.mean(kpts[:,0]),np.mean(kpts[:,1])]]
            kpts = np.concatenate([kpts,center],axis=0)

    
        return np.array(boxes), np.array(kpts)
    
    def nms_kpts_single(self,dets,scores,box,thresh=None):

 
        x = dets[:,0]
        y = dets[:,1]
        
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        
        if thresh is None:
            thresh = self.thresh * min(box_width,box_height)
        else:
            thresh = thresh * min(box_width,box_height)
        
    

        keep = []
        index = scores.argsort()[::-1]
        while index.size >0:
            i = index[0]       
            keep.append(i)


            x_2 = (x[i] - x[index[1:]]) ** 2    
            y_2 = (y[i] - y[index[1:]]) ** 2


            distance = np.sqrt(x_2 + y_2)

            idx = np.where(distance>=thresh)[0]
            index = index[idx+1]   # because index start from 1

        return dets[keep], scores[keep]
    
    
    def nms_kpts(self,):
        kpts = []
        scores = []
        for kpt, score, box in zip(self.kpts,self.kpt_scores,self.boxes):
            k,s = self.nms_kpts_single(kpt,score,box)
            kpts.append(k)
            scores.append(s)
        self.kpts = np.array(kpts)
        self.kpt_scores = np.array(scores)
    

    
    def __repr__(self,):
        print(self.kpts)
        print(self.boxes)
        return ''           


def return_connected_region(masks):
    connected_reigions = []
    masks = masks.cpu().numpy().astype('uint8')

    for mask in masks:
        connected_reigion, _= cv2.connectedComponents(mask)
        connected_reigions.append(connected_reigion)
    return connected_reigions



def kpt_sam(predictor,det_res,image,box=False,point_nms=False,vis=True,vis_all=False,use_best_mask=False,box_only=False,return_scores=False,return_num=False):
    predictor.set_image(image)
    
    input_point = None
    input_label = None
    trasnformed_coors = None
    scores = None
    num_obj = None
    max_area = None


    input_box, input_kpts = det_res.preprocess(point_nms=point_nms)
    
    



    if len(input_kpts) == 0:
        return torch.ones([image.shape[2],image.shape[0],image.shape[1]]),None, None


#     print(input_kpts)
    input_label = np.array([1]*len(input_kpts))
    
    
    if input_box.ndim == 1:
        input_box = input_box[None,:]
    
    if input_kpts.ndim == 2:
        input_kpts = input_kpts[None,:,:]
    
    areas = (input_box[:,2] - input_box[:,0]) * (input_box[:,3] - input_box[:,1])
    max_area = areas.max()
    
    
    

    input_box = torch.tensor(input_box).to(predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_box, image.shape[:2])
    # if transformed_boxes.ndim == 2 and transformed_boxes.shape[0] != 1:
    #     transformed_boxes = transformed_boxes.unsqueeze(0)
    
    input_kpts = torch.tensor(input_kpts).to(predictor.device)
    trasnformed_coors = predictor.transform.apply_coords_torch(input_kpts,image.shape[:2])
    
    input_label = torch.tensor(input_label).unsqueeze(0)





    multimask_output = True
    box_flag=False


    num_obj = transformed_boxes.shape[0]

    if num_obj > 1 or box_only: 
        trasnformed_coors = None
        if num_obj > 1:
            box_flag = True

    elif not box:
        transformed_boxes = None





    masks, scores, _ = predictor.predict_torch(
        point_coords=trasnformed_coors,
        point_labels=input_label,
        boxes=transformed_boxes,
        multimask_output=multimask_output,
        box_flag=box_flag
    )

    if num_obj > 1:
        index = scores.argsort(dim=1,descending=True).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,image.shape[0],image.shape[1])
        masks = torch.gather(masks,dim=1,index=index)
        scores = scores.sort(dim=1,descending=True)[0]
        scores = scores.mean(0)

        mask_ = torch.zeros([3,masks.shape[-2],masks.shape[-1]],device=scores.device).bool()
        for ii,mask in enumerate(masks):
            mask_ = mask_ | mask
        masks = mask_






    masks = masks.squeeze()
    scores = scores.squeeze()
    best_idx = scores.argmax()
    # connected_regions = return_connected_region(masks)
    # tuples = [(i,j,idx) for idx,(i,j) in enumerate(zip(connected_regions,scores))]

    # def conected_min_score_max(x,y):
    #     if x[0] < y[0]:
    #         return -1
    #     elif x[0] > y[0]:
    #         return 1
    #     else:
    #         if x[1] < y[1]:
    #             return 1
    #         elif x[1] > y[1]:
    #             return -1
    #         else:
    #             return 0
    # best_idx = sorted(tuples,key=cmp_to_key(conected_min_score_max))[0][-1]
    # best_idx = torch.tensor(best_idx,dtype=torch.long,device=masks.device)


    if vis:
        if vis_all:
            for i, (mask, score) in enumerate(zip(masks, scores)):

                plt.figure(figsize=(15,15))
                plt.imshow(image)
                show_mask(mask.cpu().numpy(), plt.gca())
                show_points(input_kpts.cpu().numpy(), input_label, plt.gca())
                for box in input_box:
                    show_box(box.cpu().numpy(),plt.gca())
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                plt.axis('off')
                plt.show()  
  
        else:
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(masks[best].cpu().numpy(), plt.gca())
            show_points(input_kpts.cpu().numpy(), input_label, plt.gca())
            for box in input_box:
                show_box(box.cpu().numpy(),plt.gca())
    #         show_points(input_point, input_label, plt.gca())
            # print(i,score)
            plt.title("Score: %f" % (scores[best].cpu().numpy()), fontsize=18)
            plt.axis('on')
            plt.show()  

    



    if use_best_mask:
        return masks[best_idx].unsqueeze(0), scores,num_obj,max_area
    else:
        return masks, scores,num_obj,max_area



def process_one_image(args, img_path, detector, pose_estimator, visualizer,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""
    '''
        args.det_cat_id  0
        args.bbox_thr 0.3
        args.nms_thr  0.3
        args.kpt_thr  0.3
    '''
    # predict bbox
    det_result = inference_detector(detector, img_path)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.6)]
    bboxes = bboxes[nms(bboxes, 0.7), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    # img = mmcv.imread(img_path, channel_order='rgb')

    out_file = None


    # visualizer.add_datasample(
    #     'result',
    #     img,
    #     data_sample=data_samples,
    #     draw_gt=False,
    #     draw_heatmap=False,
    #     draw_bbox=True,
    #     show_kpt_idx=True,
    #     skeleton_style='mmpose',
    #     show=True,
    #     wait_time=show_interval,
    #     out_file=out_file,
    #     kpt_thr=0.3)
    # visualizer.show()
    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def kpt_sam_interface(image,det_model,predictor,pose_model,use_box=False,use_nms=False,nms_thresh=0.4,vis=True,mean_center=False,vis_all=False,use_best_mask=False,box_only=False,return_scores=False,return_num=False):
    # image == mmcv.imread(image,channel_order='rgb')
    scores = None
    num_obj = None
    pred_instances = process_one_image(
            None,
            image,
            det_model,
            pose_model,
            None,
            show_interval=0)
    # print(pred_instances)
    # pred_instances_list = split_instances(pred_instances)
    if pred_instances is None:
        return torch.ones([image.shape[2],image.shape[0],image.shape[1]]),None,None
    else:
        det_res = mmpose2sam(pred_instances,kpt_box_threshould=0.4,img_shape=image.shape,nms_thresh=nms_thresh,mean_center=mean_center)


    masks,scores,num_obj,max_area = kpt_sam(predictor,det_res,image,use_box,use_nms,vis=vis,vis_all=vis_all,use_best_mask=use_best_mask,box_only=box_only,return_scores=return_scores)

    return masks, scores, num_obj, max_area


if __name__ == '__main__':
    #detector
    det_config = '/mnt/d/projects/mmpose/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py'
    det_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth '

    detector = init_detector(
        det_config, det_ckpt, device='cuda')
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    config_file = '/mnt/d/projects/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
    checkpoint_file = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
    model = init_pose_estimator(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'



    sam_checkpoint = "/mnt/d/projects/segment-anything/saved_model/sam_vit_b_01ec64.pth"
    device = "cuda"
    model_type = "vit_b"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)


    img = '/mnt/d/data/matting/RealWorldPortrait-636/images/02_0030_input.jpg'
    img = mmcv.imread(img,channel_order='rgb')


    kpt_sam_interface(img,detector,predictor,model,use_box=False,use_nms=True,nms_thresh=0.4,vis=True)