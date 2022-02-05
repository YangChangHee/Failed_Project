from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json_tricks as json
import numpy as np
import sys
from tqdm import tqdm
import time
from dataset.CamJointsDataset import JointsDataset
from nms.nms import oks_nms
from nms.nms import soft_oks_nms


logger = logging.getLogger(__name__)

class new_h36mDataset(JointsDataset):
    '''
    "keypoints":{
        0: "Pelvis"
        1: "R_Hip"
        2: "R_Knee"
        3: "R_Ankle"
        4: "L_Hip"
        5: "L_Ankle"
        6: "Torso"
        7: "Neck"
        8: "Nose"
        9: "Head"
        10: "L_Shoulder"
        11: "L_Elbow"
        12: "L_Wrist"
        13: "R_Shoulder"
        14: "R_Elbow"
        15: "R_Wrist"
        16: "Thorax"
    },
    "skeleton":(
        ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), 
        (2, 3), (0, 4), (4, 5), (5, 6) )
    )
    '''
    def __init__(self, cfg, root, image_set, is_train, Cam_num, transform=None):
        super().__init__(cfg, root, image_set, is_train, Cam_num, transform)
        self.Cam_num=Cam_num
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS # False
        self.oks_thre = cfg.TEST.OKS_THRE 
        self.in_vis_thre= cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE # 'data/h36m/annotations/'
        self.train_set_s=cfg.DATASET.TRAIN_SET_S
        self.test_set_s=cfg.DATASET.TEST_SET_S
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX #TRUE
        self.image_width= cfg.MODEL.IMAGE_SIZE[0]
        self.image_height=cfg.MODEL.IMAGE_SIZE[1]
        self.data_root=cfg.DATASET.ROOT
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.last_sub=0
        self.last_acids=0
        self.last_subis=0
        self.TEST_MODE=cfg.DATASET.TEST_MODE

        if self.TEST_MODE==True:
            if is_train==True:
                self.coco = self._get_ann_file_keypoint_train_T_TEST()
            else:
                self.coco = self._get_ann_file_keypoint_train_F_TEST()
        else:
            if is_train==True:
                self.coco = self._get_ann_file_keypoint_train_T()
            else:
                self.coco = self._get_ann_file_keypoint_train_F()

        self.classes=['__background__']+['person']

        self.image_set_index=self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_image:{}'.format(self.num_images))


        self.num_joints=17

        

        self.parent_ids=None
        self.upper_body_ids=(6,7,8,9,10,11,12,13,14,15,16)
        self.lower_body_ids=(0,1,2,3,4,5)

        self.joints_weight=np.array(
            [
                1., 1., 1.2, 1.5, 1., 1.5, 1., 1., 1., 1.,
                1., 1.2, 1.5, 1., 1.2, 1.5, 1.
            ],
            dtype=np.float32
        ).reshape((self.num_joints,1))

        self.db, self.db1=self._get_db()
        print(len(self.db),len(self.db1))

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)
            self.db1 = self.select_data(self.db1)

        logger.info('=> load {}_{} samples'.format(len(self.db),len(self.db1)))



    def _get_ann_file_keypoint_train_T_TEST(self):
        print("_get_ann_file_keypoint_train_T_TEST")
        coco=COCO()
        for i in self.train_set_s:
            print(i)
            with open(self.bbox_file+'Human36M_subject{}_data.json'.format(i)) as f:
                json_data=json.load(f)
            new_json_data={'images':None,'annotations':None}
            new_json_data['images']=json_data['images'][:32]
            new_json_data['annotations']=json_data['annotations'][:32]

            
            new_json_data1={'images':None,'annotations':None}
            new_json_data1['images']=json_data['images'][1383:1415]
            new_json_data1['annotations']=json_data['annotations'][1383:1415]
            """
            Train_subject_1
            cam1=> 0~1382
            cam2=> 1383~
            """
            for k,v in new_json_data.items():
                coco.dataset[k]=v
            for k,v in new_json_data1.items():
                coco.dataset[k]+=v

        coco.createIndex()
        return coco


    def _get_ann_file_keypoint_train_T(self):
        print("_get_ann_file_keypoint_train_T")
        coco=COCO()
        for i in self.train_set_s:
            print(i)
            with open(self.bbox_file+'Human36M_subject{}_data.json'.format(i)) as f:
                json_data=json.load(f)
            
            if len(coco.dataset)==0:
                for k,v in json_data.items():
                    coco.dataset[k]=v
            else:
                for k,v in json_data.items():
                    coco.dataset[k]+=v

        coco.createIndex()
        #print(coco.dataset['images'][1383])
        #sys.exit()
        return coco


    def _get_ann_file_keypoint_train_F_TEST(self):
        print("_get_ann_file_keypoint_train_F_TEST")
        coco=COCO()
        for i in self.test_set_s:
            with open(self.bbox_file+'Human36M_subject{}_data.json'.format(i)) as f:
                json_data=json.load(f)
            new_json_data={'images':None,'annotations':None}
            new_json_data['images']=json_data['images'][:32]
            new_json_data['annotations']=json_data['annotations'][:32]

            
            new_json_data1={'images':None,'annotations':None}
            new_json_data1['images']=json_data['images'][2356:2388]
            new_json_data1['annotations']=json_data['annotations'][2356:2388]

            """
            TEST_subject_9
            cam1=> 0~2355
            cam2=> 2356~
            """            
            for k,v in new_json_data.items():
                coco.dataset[k]=v
            for k,v in new_json_data1.items():
                coco.dataset[k]+=v

        coco.createIndex()
        return coco



    def _get_ann_file_keypoint_train_F(self):
        print("_get_ann_file_keypoint_train_F")
        coco=COCO()
        for i in self.test_set_s:
            with open(self.bbox_file+'Human36M_subject{}_data.json'.format(i)) as f:
                json_data=json.load(f)
            
            if len(coco.dataset)==0:
                for k,v in json_data.items():
                    coco.dataset[k]=v
            else:
                for k,v in json_data.items():
                    coco.dataset[k]+=v

        coco.createIndex()
        return coco

    def _load_image_set_index(self):
        print("_load_image_set_index")
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            gt_db,gt_db1=self._load_coco_keypoint_annotations()
        else:
            gt_db,gt_db1=self._load_coco_person_detection_result()
        return gt_db, gt_db1

    def _load_coco_keypoint_annotations(self):
        print("_load_coco_keypoint_annotations")
        gt_db_cam1=[]
        gt_db_cam2=[]
        gt_db_cam3=[]
        gt_db_cam4=[]
        kpy_npy=None
        for n,index in enumerate(tqdm(self.image_set_index)):
            time.sleep(0.000000000000001)
            check=self.coco.loadImgs(index)[0]
            sub=check['subject']
            aids=check['action_idx']
            subids=check['subaction_idx']
            camids=check['cam_idx']
            fids=check['frame_idx']

            # 2d keypoint gt
            if self.last_sub != sub:
                kpy_npy1=np.load(self.data_root+'hm36_np/S_{}_A_{}_SA_{}_in.npy'.format(sub,aids,subids))
                kpy_npy=kpy_npy1.view()
            else:
                if self.last_acids != aids:
                    kpy_npy1=np.load(self.data_root+'hm36_np/S_{}_A_{}_SA_{}_in.npy'.format(sub,aids,subids))
                    kpy_npy=kpy_npy1.view()
                else:
                    if self.last_subis != subids:
                        kpy_npy1=np.load(self.data_root+'hm36_np/S_{}_A_{}_SA_{}_in.npy'.format(sub,aids,subids))
                        kpy_npy=kpy_npy1.view()

            self.last_sub=sub
            self.last_acids=aids
            self.last_subis=subids
            keypoints_data=kpy_npy[fids][camids-1]

            if camids ==1:
                gt_db_cam1.extend(self._load_coco_keypoint_annotation_kernel(index, keypoints_data))
            elif camids == 2:
                gt_db_cam2.extend(self._load_coco_keypoint_annotation_kernel(index, keypoints_data))
            elif camids == 3:
                gt_db_cam3.extend(self._load_coco_keypoint_annotation_kernel(index, keypoints_data))
            elif camids == 4:
                gt_db_cam4.extend(self._load_coco_keypoint_annotation_kernel(index, keypoints_data))
        if self.Cam_num==[1,2]:
            return gt_db_cam1, gt_db_cam2
        elif self.Cam_num==[3,4]:
            return gt_db_cam3, gt_db_cam4


    def _load_coco_keypoint_annotation_kernel(self, index,keypoints_data):
        im_ann=self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height= im_ann['height']

        annIds= self.coco.getAnnIds(imgIds=index)
        objs=self.coco.loadAnns(annIds)

        valid_objs =[]
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))

            if x2>= x1 and y2>=y1:
                obj['clean_bbox']=[x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs=valid_objs

        rec=[]
        for obj in objs:

            joints_3d = np.zeros((self.num_joints,3),dtype=np.float)
            joints_3d_vis=np.zeros((self.num_joints,3),dtype=np.float)
            t_vis1 = np.array(obj['keypoints_vis'],dtype=np.int)
            for ipt in range(self.num_joints):
                joints_3d[ipt,0] = keypoints_data[ipt][0]
                joints_3d[ipt,1] = keypoints_data[ipt][1]
                joints_3d[ipt,2] = 0
                t_vis=t_vis1[ipt]

                joints_3d_vis[ipt,0]=t_vis
                joints_3d_vis[ipt,1]=t_vis
                joints_3d_vis[ipt,2]=0

            center,scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })
            #print(rec)
            
        return rec

    def _box2cs(self,box):
        x,y,w,h = box[:4]
        return self._xywh2cs(x,y,w,h)


    # 확인
    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        check1=self.coco.loadImgs(index)[0]

        file_name = check1['file_name']

        image_path = os.path.join(
            self.data_root, 'images/', file_name)

        return image_path

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                *args, **kwargs):
        rank = cfg.RANK

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make{}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_{}.json'.format(
                self.image_set, rank)
        )

        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': img_path[idx][-38:-4]
            })

        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        num_joints=self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts=[]
        for img in kpts.keys():
            img_kpts=kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s=n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)
        """
        if 'test' not in self.image_set:
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0
        """

    def _write_coco_keypoint_results(self, keypoints, res_file):
        #'cat_id': self._class_to_coco_ind[cls],
        data_pack = [
            {
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)


    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        #cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float
            )

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': 1,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results



    



"""
활용할 것 /human_pose/HRNet/make_model_test/untitle1

coco 자체를 사용하는 것이 아니여서 gt_db에 넣는거임
cam 1, 2를 dataset으로 넣어서 훈련시켜야겟다.

cam 3, 4를 dataset으로 넣어서 훈련 => 마지막 앙상블 방법을 이용!
"""