from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import convolve
import time
import math
import datetime

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

logger = logging.getLogger(__name__)

class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train,Cam_num ,transform=None):
        self.Cam_num=Cam_num
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set # Train or Test

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP # 뒤집다.
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB


        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1
        self.TEST_MODE = cfg.DATASET.TEST_MODE

        self.transform = transform
        self.db = []
        self.db1 =[]

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.db)

    def half_body_transform(self, joints, joints_vis):
        upper_joints= []
        lower_joints= []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0]>0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints)>2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints\
                if len(lower_joints) > 2 else upper_joints
        
        if len(selected_joints)<2:
            return None, None
        
        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w= right_bottom[0] - left_top[0]
        h= right_bottom[1] - left_top[1]

        if w>self.aspect_ratio *h:
            h=w*1.0 / self.aspect_ratio
        elif w<self.aspect_ratio *h:
            w= h*self.aspect_ratio
        
        scale = np.array(
            [
                w*1.0/ self.pixel_std,
                h*1.0/ self.pixel_std
            ],
            dtype=np.float32
        )

        scale=scale*1.5

        return center, scale

            
    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        db_rec1 = copy.deepcopy(self.db1[idx])

        image_file= db_rec['image']
        image_file1= db_rec1['image']
        
        filename= db_rec['filename'] if 'filename' in db_rec else ''
        filename1= db_rec1['filename'] if 'filename' in db_rec1 else ''
        
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        imgnum1 = db_rec1['imgnum'] if 'imgnum' in db_rec1 else ''

        #print(image_file)

        if self.data_format =='zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            data_numpy1 = zipreader.imread(
                image_file1, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            data_numpy1 = cv2.imread(
                image_file1, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        
        if self.color_rgb:
            data_numpy= cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            data_numpy1= cv2.cvtColor(data_numpy1, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))
        """"""
        joints =db_rec['joints_3d']
        joints1 =db_rec1['joints_3d']

        joints_vis= db_rec['joints_3d_vis']
        joints_vis1= db_rec1['joints_3d_vis']
        ######
        """
        c = db_rec['center']
        c1 = db_rec1['center']

        s = db_rec['scale']
        s1 = db_rec1['scale']
        
        score = db_rec['score'] if 'score' in db_rec else 1
        r=0
        score1 = db_rec1['score'] if 'score' in db_rec1 else 1
        r1=0

        if self.is_train:
            if (np.sum(joints_vis[:,0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(joints,joints_vis)

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            if (np.sum(joints_vis1[:,0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body1, s_half_body1 = self.half_body_transform(joints1,joints_vis1)

                if c_half_body1 is not None and s_half_body1 is not None:
                    c1, s1 = c_half_body1, s_half_body1

            sf = self.scale_factor
            rf = self.rotation_factor

            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
            
            s1 = s1 * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r1 = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0
            
            if self.flip and random.random() <= 0.5:
                data_numpy1 = data_numpy1[:, ::-1, :]
                joints1, joints_vis1 = fliplr_joints(
                    joints1, joints_vis1, data_numpy1.shape[1], self.flip_pairs)
                c1[0] = data_numpy1.shape[1] - c1[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        trans1 = get_affine_transform(c1, s1, r1, self.image_size)
        
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        input1 = cv2.warpAffine(
            data_numpy1,
            trans1,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        """
        input = cv2.resize(
            data_numpy,
            (int(self.image_size[0]), int(self.image_size[1])),
            interpolation=cv2.INTER_AREA)
        input1 = cv2.resize(
            data_numpy1,
            (int(self.image_size[0]), int(self.image_size[1])),
            interpolation=cv2.INTER_AREA)
        if self.transform:
            input = self.transform(input)
        if self.transform:
            input1 = self.transform(input1)
        """
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        
        for i in range(self.num_joints):
            if joints_vis1[i, 0] > 0.0:
                joints1[i, 0:2] = affine_transform(joints1[i, 0:2], trans1)
        """
        target, target_weight = self.generate_target(joints, joints_vis)
        target1, target_weight1 = self.generate_target(joints1, joints_vis1)

        #target_h = self.fundheatmap(joints,joints1,target1)
        #target_h1 = self.fundheatmap(joints1,joints,target)

        #target_h=(target_h+target)/2
        #target_h1=(target_h1+target1)/2

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        target1 = torch.from_numpy(target1)
        target_weight1 = torch.from_numpy(target_weight1)

        

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
        }
        meta1 = {
            'image': image_file1,
            'filename': filename1,
            'imgnum': imgnum1,
            'joints': joints1,
            'joints_vis': joints_vis1,
        }

        return input,input1, target,target1, target_weight,target_weight1, meta,meta1


    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def fundheatmap(self,joints_1,joints1_1,heatmap_1):
        fund_sigma=0.04 # plus_1 3 0.04 => max 4.4x
        conv_m=np.ones((self.num_joints,1,1),dtype=np.float32)*fund_sigma

        Z_p=np.zeros((heatmap_1.shape[0],heatmap_1.shape[1],heatmap_1.shape[2]),dtype=np.float32)
        
        
        for n,(i,j) in enumerate(zip(conv_m,heatmap_1)):
            #j1=np.pad(j,((1,1),(1,1)),'constant',constant_values=0)
            Z_p[n]=convolve(j,i)

        i_h_ratio=self.image_size/self.heatmap_size
        joints_1=joints_1/i_h_ratio[0]
        joints1_1=joints1_1/i_h_ratio[0]
        F,mask=cv2.findFundamentalMat(joints_1,joints1_1,cv2.FM_LMEDS)
        joints_1=joints_1*i_h_ratio[0]
        joints1_1=joints1_1*i_h_ratio[0]
        
        pixel = []
        for i in range(self.heatmap_size[0]):
            for j in range(self.heatmap_size[1]):
                pixel.append([i,j,1])
        pixel=np.array(pixel)
        a_b_c=np.dot(pixel,F) # [num_pixel, 3]
        pixel_x=np.array([i for i in range(self.heatmap_size[0])]) # [1, 48]
        a_b_c=np.transpose(a_b_c) #[3, num_pixel]
        i_box=[]
        for i in pixel_x:
            y=-(a_b_c[0]*i+a_b_c[2])/a_b_c[1]
            y=np.array(y,dtype=np.int16)
            i_box.append(y)
        i_box=np.array(i_box) # 48, 3072
        zero_heatmap=np.zeros((self.num_joints,self.heatmap_size[1],self.heatmap_size[0]),dtype=np.float32)
        zero_heatmap=np.transpose(zero_heatmap)
        Z_p=np.transpose(Z_p)
        i_box=np.transpose(i_box)

        #start=time.time()
        for n,i in enumerate(i_box):
            x_p=n//self.heatmap_size[1]
            y_p=n%self.heatmap_size[1]
            index_coordi=np.where((0<i)&(i<63))
            for k in index_coordi[0]:
                zero_heatmap[k][i[k]]+=Z_p[x_p][y_p]
        #end=time.time()
        #sec=(end-start)
        #print("time_Calculation_Fused_heatmap: ",datetime.timedelta(seconds=sec))
        zero_heatmap=np.transpose(zero_heatmap)
        def sigmoid(x):
            return 1/ (1+np.exp(-x))
        zero_heatmap=sigmoid(zero_heatmap)

        return zero_heatmap


        

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)
            #target_weight * joints_weight index ! 마다 곱셈

        return target, target_weight