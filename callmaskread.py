import random
import string
import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import torch
import cv2
import numpy as np
import mmcv
import time


if not os.path.exists('../associations.txt'):
    print("associations.txt does not exist")
    exit(0)

# initialize
print(sys.version)
print('torch -V: ', torch.__version__)
config = '/home/xyz/newworkspace/mmdetection/configs/scnet/scnet_r50_fpn_1x_coco.py'
# Setup a checkpoint file to load
checkpoint = '/home/xyz/rebuid/checkpoints/scnet_r50_fpn_1x_coco-c3f09857.pth'
# initialize the detector
model = init_detector(config, checkpoint, device='cuda:0')
ii=0
with open("../associations.txt","r") as ass:
    for ass_data in ass:
        if len(ass_data)<50 :
            break
        common_name="/home/xyz/rebuid/rgbd_dataset_freiburg2_desk_with_person/"

        rgb_input_name=common_name+ass_data[18:43]
        depth_input_name=common_name+ass_data[62:89]

        result = inference_detector(model, rgb_input_name)

        img = cv2.imread(rgb_input_name)


        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)


        adepth = cv2.imread(depth_input_name,-1)


        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > 0.7)[0]#!!!jingzhundu

            for i in inds:
                i = int(i)
                if labels[i]!=0:
                    continue;
                sg = segms[i]
                if isinstance(sg, torch.Tensor):
                    sg = sg.detach().cpu().numpy()
                mask = sg.astype(bool)
                img[mask]=255
                adepth[mask]=0


        cv2.imwrite('/home/xyz/rebuid/rgbd_dataset_freiburg2_desk_with_person/adepth/'+ass_data[68:89], adepth)
        cv2.imwrite('/home/xyz/rebuid/rgbd_dataset_freiburg2_desk_with_person/argb/'+ass_data[22:43], img)

        print(ii)
        ii=ii+1
exit(0)
