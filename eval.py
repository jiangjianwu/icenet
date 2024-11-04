"""
Training Script
Date: 2023.11.04
Author: fengbuxi@glut.edu.cn
"""
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from utils.score import Metric
from utils.logger import setup_logger
from utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler
from utils.visualize import get_color_pallete
from dataloaders.dataloader import IndoorContextDataSet
from models import get_model
from global_config import nclass, task_num, pretrained_tasks
from train import parse_args

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        data_kwargs = {'base_size': args.base_size, 'crop_size': args.crop_size}
        val_dataset = IndoorContextDataSet(root=args.data_root, split='val', transform=input_transform, **data_kwargs)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_model(
            version=args.version,
            nclass=nclass,
            backbone=args.backbone,
            pretrained_base=args.pretrained_base,
            norm_layer=BatchNorm2d,
            jpu=args.jpu,
            pth_path=args.pth_path).to(self.device)
        if self.args.device != 'cpu' and len(self.args.gpus) > 1: # 多GPU训练
            self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(args.checkpoints, map_location=args.device))
        
        # evaluation metrics
        self.metric = []
        for i in range(task_num):
            self.metric.append(Metric(nclass[i]))

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def eval(self):
        # Reset mertric
        for i in range(task_num):
            self.metric[i].reset()

        self.model.eval()
        model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (images, labels, filenames) in enumerate(self.val_loader):
            images = images.to(args.device)
            labels = [i.to(self.device) for i in labels]

            with torch.no_grad():
                outputs = model(images)
            
            # ================ M X N ==================
            for j in range(pretrained_tasks, task_num):
                self.metric[j].update(outputs[j], labels[j])
                AllAcc, ClsAcc, IoU, mIoU = self.metric[j].get()
                logger.info("Sample-Task{:d}: {:d}, Validation AllAcc: {:.3f}, mIoU: {:.3f}".format(j, i + 1, AllAcc, mIoU))
                logger.info("Sample-Task{:d}: {:d}, ClassAcc: {}, IoU: {}".format(j, i + 1, ClsAcc.__str__(), IoU.__str__())) 

        synchronize()

if __name__ == '__main__':
    args = parse_args()
    args.distributed = False

    # Pretrained model path
    args.save_dir = os.path.join('running', '2024-08-14_V1.0') # 此处必须替换为要验证的模型权重路径
    args.log_dir = os.path.join(args.save_dir, 'logs')
    args.checkpoints = os.path.join(args.save_dir, 'checkpoint', 'mtcnet_best.pth')
    args.visual = os.path.join(args.save_dir, 'visual')
    logger = setup_logger("MTCNet", args.log_dir, get_rank(), filename='eval_log.txt', mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()