from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import cv2
import sys
from tqdm import tqdm
import time
import math
import datetime

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def make_fundamental(joint_1,joint_2):
    j1_batch_size=joint_1.shape[0]
    j2_batch_size=joint_2.shape[0]

    F_data=[]

    for i in range(j1_batch_size):
        F, mask = cv2.findFundamentalMat(joint_1[i],joint_2[i],cv2.FM_LMEDS)
        F_data.append(F)
    F_data=np.array(F_data)
    #F_data=torch.Tensor(F_data)
    return F_data
"""
def padding_heatmap(heatmap):
    padding=nn.ZeroPad2d(1)
    pad_heatmap=padding(heatmap)
    return pad_heatmap
"""
def make_a_b_c(heatmap_size,matrix):
    """
    heatmap_size[0] => x
    heatmap_size[1] => y
    To make function 'ax+by+c=0' 
    pixel = [x,y,1]
    matrix = [batch_size, Fundamental Matrix]
    Batch Multiplication
    """
    #print("heatmap :",heatmap_size)
    pixel=[]
    for i in range(heatmap_size[0]):
        for j in range(heatmap_size[1]):
            pixel.append([i,j,1])
    pixel=np.array([pixel])
    batch_matrix=matrix.shape[0]
    new_pixel=np.repeat(pixel,repeats=batch_matrix,axis=0)

    batch_a_b_c=np.matmul(new_pixel,matrix)

    return batch_a_b_c

def a_b_c_pixel(a_b_c,heatmap_size,conv_heatmap,num_joint):
    """
    a_b_c => [batch, num_pixel, 3(a,b,c)]
    heatmap_size => [x, y]
    conv_heatmap => [batch, num, y, x]
    f = fundmental(y,x)
    epipolar line => x*f
    y plane
    """
    #print("a_b_c :",a_b_c.shape)
    batch_size=a_b_c.shape[0]
    #print("heatmap :",heatmap_size)
    #print("conv :",conv_heatmap.shape)
    pixel_x=np.array([[i for i in range(heatmap_size[0])]])
    pixel_x=np.repeat(pixel_x,repeats=batch_size,axis=0)
    pixel_x=np.transpose(pixel_x)
    a_b_c=np.transpose(a_b_c)
    i_box=[]
    for i in pixel_x:
        y=-(a_b_c[0]*i+a_b_c[2])/a_b_c[1]
        y=np.array(y,dtype=np.int16)
        i_box.append(y)
        #print(y.shape) # 3072, 2
    i_box=np.array(i_box)
    i_box=np.transpose(i_box,(1,0,2))
    zero_heatmap=torch.zeros(batch_size,num_joint,heatmap_size[1],heatmap_size[0]).cuda()
    
    zero_heatmap=torch.transpose(zero_heatmap,1,3)
    zero_heatmap=torch.transpose(zero_heatmap,0,2)
    zero_heatmap=torch.transpose(zero_heatmap,0,1)
    conv_heatmap=torch.transpose(conv_heatmap,1,3)
    conv_heatmap=torch.transpose(conv_heatmap,0,2)
    conv_heatmap=torch.transpose(conv_heatmap,0,1)
    

    start=time.time()
    for n,i in enumerate(i_box):
        x_p=n//heatmap_size[1]
        y_p=n%heatmap_size[1]
        for n1, j in enumerate(i):
            for k in j:
                if 0<k<63:
                    zero_heatmap[n1][k]+=conv_heatmap[x_p][y_p]

                    
    end=time.time()
    sec=(end-start)
    print("time_Calculation_Fused_heatmap: ",datetime.timedelta(seconds=sec))
    zero_heatmap=torch.transpose(zero_heatmap,0,2)
    zero_heatmap=torch.transpose(zero_heatmap,1,3)
    zero_heatmap=torch.transpose(zero_heatmap,2,3)
    
    return zero_heatmap

class Module(nn.Module):
    def __init__(self,cfg,model1,model2,**kwargs):
        super(Module,self).__init__()
        #self.pretrained_state=torch.load(cfg.MODEL.PRETRAINED)
        
        self.model1=model1
        self.model2=model2
        """
        channel-wise convolution
        """
        self.Conv1=nn.Conv2d(17,17,3,1,1,groups=17,bias=False).cuda()
        self.Conv2=nn.Conv2d(17,17,3,1,1,groups=17,bias=False).cuda()
        nn.init.uniform_(self.Conv1.weight,0,0.1) # Weight => 0~0.1
        nn.init.uniform_(self.Conv2.weight,0,0.1) # Weight => 0~0.1

        self.sigmoid=nn.Sigmoid()

        self.heatmap_size =cfg.MODEL.HEATMAP_SIZE
        self.num_joint = cfg.MODEL.NUM_JOINTS

    def forward(self,x,y):
        x1=self.model1(x)
        x2=self.model2(y)
        list_x1,m=get_max_preds(x1.cpu().detach().numpy())
        list_x2,m1=get_max_preds(x2.cpu().detach().numpy())
        x1=x1.cuda()
        x2=x1.cuda()
        conv_x1=self.Conv1(x1)
        conv_x2=self.Conv2(x2)

        Fundmental_Batch_1=make_fundamental(list_x1,list_x2)
        Fundmental_Batch_2=make_fundamental(list_x2,list_x1)

        b_a_b_c_1=make_a_b_c(self.heatmap_size,Fundmental_Batch_1)
        b_a_b_c_2=make_a_b_c(self.heatmap_size,Fundmental_Batch_2)

        m_heatmap1=a_b_c_pixel(b_a_b_c_1,self.heatmap_size,conv_x2,self.num_joint)
        m_heatmap2=a_b_c_pixel(b_a_b_c_2,self.heatmap_size,conv_x1,self.num_joint)

        m_heatmap1=self.sigmoid(m_heatmap1)
        m_heatmap2=self.sigmoid(m_heatmap2)

        x_out1=(x1+m_heatmap1)/2
        x_out2=(x2+m_heatmap2)/2
        return x_out1, x_out2 , x1, x2

def get_pose_net(cfg,model1,model2,**kwargs):
    module=Module(cfg,model1,model2,**kwargs)

    return module