# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import time
import datetime
import math

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input,input1, target,target1, target_weight,target_weight1, meta,meta1,h_heatmap,h_heatmap1) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs,outputs1,o_heat1s,o_heats= model(input,input1)
        #o=o.cpu().detach().numpy()
        #plt.imsave('/home/newuser/human_pose/HRNet/HRNet-Human-Pose-Estimation-master/lib/core/test_output_file/1.jpg',o[0][0])
        target = target.cuda(non_blocking=True)
        target1 = target1.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        target_weight1 = target_weight1.cuda(non_blocking=True)
        h_heatmap = h_heatmap.cuda(non_blocking=True)
        h_heatmap1 = h_heatmap1.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            loss += criterion(outputs1[0], target1, target_weight1)
            loss += criterion(o_heats[0], h_heatmap, target_weight)
            loss += criterion(o_heat1s[0], h_heatmap1, target_weight1)
            
            for output,output1,o_heat, o_heat1 in zip(outputs[1:],outputs1[1:],o_heats[1:],o_heat1s[1:]):
                loss += criterion(output, target, target_weight)
                loss += criterion(output1, target1, target_weight1)
                loss += criterion(o_heat, h_heatmap, target_weight)
                loss += criterion(o_heat1, h_heatmap1, target_weight1)
                
        else:
            output = outputs
            output1 = outputs1
            o_heat = o_heats
            o_heat1 = o_heat1s
            loss = criterion(output, target, target_weight)
            loss += criterion(output1, target1, target_weight1)
            loss += criterion(o_heat, h_heatmap, target_weight )
            loss += criterion(o_heat1, h_heatmap1,target_weight1)
            ####$


        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        start=time.time()
        print('zero_optimizer')
        optimizer.zero_grad()
        print('backward_loss')
        loss.backward()
        print('optimizer_step')
        optimizer.step()
        end=time.time()
        sec=(end-start)
        print("time_Calculation_Loss: ",datetime.timedelta(seconds=sec))

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        _, avg_acc1, cnt1, pred1 = accuracy(output1.detach().cpu().numpy(),
                                         target1.detach().cpu().numpy())
        acc.update(avg_acc, cnt)
        acc1.update(avg_acc1, cnt1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss_cam {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy_cam1 {acc.val:.3f} ({acc.avg:.3f})\t' \
                  'Accuracy_cam2 {acc1.val:.3f} ({acc1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc,
                      acc1=acc1)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer.add_scalar('train_acc1', acc1.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)
            save_debug_images(config, input1, meta1, target1, pred1*4, output1,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input,input1, target,target1, target_weight,target_weight1, meta,meta1,h_heatmap,h_heatmap1) in enumerate(val_loader):
            # compute output
            outputs,outputs1,o_heat1s,o_heats= model(input,input1)
            if isinstance(o_heat1s, list):
                o_heats = o_heats[-1]
                o_heat1s = o_heat1s[-1]
            else:
                o_heats = o_heats
                o_heat1s = o_heat1s

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                input_flipped1 = np.flip(input1.cpu().numpy(), 3).copy()
                input_flipped1 = torch.from_numpy(input_flipped1).cuda()
                outputs_flipped,outputs_flipped1,o_flipped_heat1s,o_flipped_heats = model(input_flipped,input_flipped1)

                if isinstance(o_flipped_heat1s, list):
                    o_flipped_heats = o_flipped_heats[-1]
                    o_flipped_heat1s = o_flipped_heat1s[-1]
                    
                else:
                    o_flipped_heats = o_flipped_heats
                    o_flipped_heat1s = o_flipped_heat1s

                o_flipped_heats = flip_back(o_flipped_heats.cpu().numpy(),
                                           val_dataset.flip_pairs)
                o_flipped_heats = torch.from_numpy(o_flipped_heats.copy()).cuda()
                o_flipped_heat1s = flip_back(o_flipped_heat1s.cpu().numpy(),
                                           val_dataset.flip_pairs)
                o_flipped_heat1s = torch.from_numpy(o_flipped_heat1s.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    o_flipped_heats[:, :, :, 1:] = \
                        o_flipped_heats.clone()[:, :, :, 0:-1]
                    o_flipped_heat1s[:, :, :, 1:] = \
                        o_flipped_heat1s.clone()[:, :, :, 0:-1]

                o_heats = (o_heats + o_flipped_heats) * 0.5
                o_heat1s = (o_heat1s + o_flipped_heat1s) * 0.5

            h_heatmap = h_heatmap.cuda(non_blocking=True)
            h_heatmap1 = h_heatmap1.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(o_heat1s, h_heatmap1, target_weight)
            loss += criterion(o_heats, h_heatmap, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(o_heat1s.cpu().numpy(),
                                             h_heatmap1.cpu().numpy())
            _, avg_acc1, cnt1, pred1 = accuracy(o_heats.cpu().numpy(),
                                             h_heatmap.cpu().numpy())

            acc.update(avg_acc, cnt)
            acc1.update(avg_acc1, cnt1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, o_heats.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss_Cam_12 {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy_cam1 {acc.val:.3f} ({acc.avg:.3f})\t'\
                      'Accuracy_cam2 {acc1.val:.3f} ({acc1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc,acc1=acc1)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, h_heatmap1, pred*4, o_heat1s,
                                  prefix)
        ## name_values, pref_indicator ???????????????
        # name_values, perf_indicator
        val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        """
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)
        """

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc_cam',
                acc.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc_cam',
                acc1.avg,
                global_steps
            )
            """
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            """
            writer_dict['valid_global_steps'] = global_steps + 1

    #return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
