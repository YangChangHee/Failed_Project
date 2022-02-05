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

class Module(nn.Module):
    def __init__(self,cfg,model1,model2,**kwargs):
        super(Module,self).__init__()
        #self.pretrained_state=torch.load(cfg.MODEL.PRETRAINED)
        
        self.model1=model1
        self.model2=model2
        self.Conv1_1=nn.Conv2d(17,34,3,1,1)
        self.Conv2_1=nn.Conv2d(34,48,3,1,1)
        self.Conv3_1=nn.Conv2d(48,48,3,1,1)
        self.Conv4_1=nn.Conv2d(48,17,3,1,1)
        self.Conv1_2=nn.Conv2d(17,34,3,1,1)
        self.Conv2_2=nn.Conv2d(34,48,3,1,1)
        self.Conv3_2=nn.Conv2d(48,48,3,1,1)
        self.Conv4_2=nn.Conv2d(48,17,3,1,1)
        self.relu=nn.ReLU()

    def forward(self,x,y):
        x=self.model1(x)
        y=self.model2(y)
        x1=self.relu(self.Conv1_1(x))
        y1=self.relu(self.Conv1_2(y))
        x1=self.relu(self.Conv2_1(x1))
        y1=self.relu(self.Conv2_2(y1))
        x1=self.relu(self.Conv3_1(x1))
        y1=self.relu(self.Conv3_2(y1))
        x1=self.relu(self.Conv4_1(x1))
        y1=self.relu(self.Conv4_2(y1))
        x1=(y+x1)/2
        y1=(x+y1)/2
        return x, y, x1, y1

def get_pose_net(cfg,model1,model2,**kwargs):
    module=Module(cfg,model1,model2,**kwargs)

    return module