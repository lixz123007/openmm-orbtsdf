import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import torch
import cv2
import numpy as np
import mmcv
import time
def printHello():
    print(sys.version)
    print('torch -V: ', torch.__version__)
    print("hello world!")


def masktry():
    t0=time.time();
   # print(sys.version)
    #print('torch -V: ', torch.__version__)
    config = '/home/finch/catkin_ws/src/ORB_SLAM2/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = '/home/finch/catkin_ws/src/ORB_SLAM2/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # initialize the detector
    config = '/home/finch/catkin_ws/src/ORB_SLAM2/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = '/home/finch/catkin_ws/src/ORB_SLAM2/mmdetection/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
    # initialize the detector
    model = init_detector(config, checkpoint, device='cuda:0')

    t1=time.time()
    img = '/home/finch/SLAM/src/orbtsdf/mmdetection/demo/demo.jpg'# test a single image
    img = '/home/finch/catkin_ws/src/ORB_SLAM2/b1.jpg'# test a single image
    #img = '/home/finch/test/coser2.jpeg'
    result = inference_detector(model, img)
    t2=time.time()

    # for i in range(100):
    #     t1=time.time()
            
    #     if i%2 == 1:
    #         img = '/home/finch/Pictures/qushan.jpeg'# test a single image
    #     if i%2== 0:
    #         img = '/home/finch/catkin_ws/src/ORB_SLAM2/b1.jpg'# test a single image
    #     #img = '/home/finch/test/coser2.jpeg'
    #     result = inference_detector(model, img)
    #     t2=time.time()
    #     print(i,t2-t1)

    img = cv2.imread(img)
    img = img.copy()
    img3=np.zeros((img.shape), np.uint8)
    t3=time.time()
    # t1=time.time()
    #print(result)
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

    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > 0.7)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        for i in inds:
            i = int(i)
            if labels[i]!=0 and labels[i]!=15:
                continue;
            color_mask = color_masks[labels[i]]
            sg = segms[i]
            if isinstance(sg, torch.Tensor):
                sg = sg.detach().cpu().numpy()
            mask = sg.astype(bool)
            img3[mask]=img3[mask]+color_mask
            #img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # t2=time.time()
    img3 = cv2.cvtColor(img3,cv2.COLOR_RGB2GRAY)
    # t3=time.time()
    #res=img3
    #return 
    # res=b''
    # for i in range(img3.shape[0]):
    #     for j in range(img3.shape[1]):
    #         # print(img3[i,j])
    #         res=res+b"%d,"%(img3[i,j])

    res=b''
    for i in range(0,len(labels)):
        if(labels[i]!=0 and labels[i]!=15):
            continue
        res=res+b"%d %d %d %d %d %d,"%(bboxes[i][4]*100,labels[i],bboxes[i][0],bboxes[i][1],bboxes[i][2],bboxes[i][3])


    # t4=time.time()
    cv2.imwrite('black.jpg',img3)
    t5=time.time()
    # print(time.time())
    # print(t5-t4)
    # print(t4-t3)
    # print(t3-t2)
    # print(t2-t1)
    print("pythontime:"+str(t5-t0))
    print("once: "+str(time.time()))
    #print(img3)
    #print(res)
    #print("bbox_result:  ")
    #print(len(bboxes))
    #print(bboxes)
    #print(labels)
    #print(bbox_result)
    #print("segm_result:  ")
    #print(type(img3))
    #print(segm_result)
    #print("once!")
    return res#print(result)
    # show the results
    #show_result_pyplot(model, img, result, score_thr=0.3)
    #img2 = model.show_result(img, result, score_thr=0.8, show=False)
    #cv2.imwrite('yzma2.jpg',img2)


#masktry()