from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import numpy as np
import random
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import sys
import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function1 import train
from core.function1 import validate
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
import dataset
import models
import matplotlib.pyplot as plt




def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model1 = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    model2 = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    pretrained_state=torch.load('output/h36m/pose_hrnet/w32_256x192_adam_lr1e-3/checkpoint.pth')
    model1=torch.nn.DataParallel(model1)
    model2=torch.nn.DataParallel(model2)
    model1.load_state_dict(pretrained_state['state_dict'])
    model2.load_state_dict(pretrained_state['state_dict'])
    model1=model1.module.cpu()
    model2=model2.module.cpu()
    """
    for child in model1.children():
        for param in child.parameters():
            param.requires_grad = False
    
    for child in model2.children():
        for param in child.parameters():
            param.requires_grad = False
    """
    #print(model1)
    # 3:50??? ?????? train ??????
    
    
    """
    model3 = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    model4 = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    """

    model_f = eval('models.'+cfg.MODEL.NAME2+'.get_pose_net')(
        cfg, model1, model2
    )
    """
    for name, param in model_f.named_parameters():
        print('param.requires_grad: ', param.requires_grad)
    """


    this_dir = os.path.dirname(__file__)

    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    dump_input1 = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )

    logger.info(get_model_summary(model_f, dump_input,dump_input1))

    #print(model_f(dump_input)[0].shape,model_f(dump_input)[1].shape)

    model = torch.nn.DataParallel(model_f, device_ids=cfg.GPUS).cuda()

    dump_input=dump_input.cuda()
    dump_input1=dump_input1.cuda()

    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True, [1,2],
        transforms.Compose([transforms.ToTensor(), normalize,])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    

    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, [1,2],
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_pref=0.0
    best_model=False
    last_epoch=-1
    optimizer = get_optimizer(cfg,model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        optimizer.step()
        lr_scheduler.step()

        train(cfg, train_loader, model, criterion, optimizer, epoch,
            final_output_dir, tb_log_dir, writer_dict)
        
        validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            #'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()



# python tools/new_module.py --cfg experiments/h36m/hrnet/new_w32_256x192_adam_lr1e-3.yaml
# => Freeze Pre-Trained Model
# python tools/new_module.py --cfg experiments/h36m/hrnet/new_w32_256x192_adam_lr1e-3_Model.yaml
# => Trained Pre-Trained Model

if __name__ == '__main__':
    main()