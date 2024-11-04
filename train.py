#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
训练脚本

@File        :   train.py
@Time        :   2024-08-12 10:45:00
@Author      :   Jiang Jianwu
@Contact     :   fengbuxi@glut.edu.cn
@Affiliation :   Guilin University of Technology
@Version     :   1.0
"""
import os
import sys
import argparse
import time
import datetime
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision import transforms

from utils.distributed import get_rank, make_data_sampler, make_batch_data_sampler, synchronize
from utils.logger import setup_logger
from utils.lr_scheduler import WarmupPolyLR
from utils.loss import MixSoftmaxCrossEntropyLoss
from utils.score import Metric
from dataloaders.dataloader import IndoorContextDataSet
from models import get_model
from global_config import nclass, task_num, pretrained_tasks, data_root

def parse_args():
    parser = argparse.ArgumentParser(description='MTCNet Training With Pytorch')
    parser.add_argument('--version', type=str, default='1.0', help='Model Version')
    parser.add_argument('--remark', type=str, default='001', help='Remark')
    # training hyper params
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50',
                                 'resnet101', 'resnet152', 'densenet121',
                                 'densenet161', 'densenet169', 'densenet201'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 50)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--power', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-3, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0, help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3, help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear', help='method of warmup')
    parser.add_argument('--base-size', type=int, default=520, help='base image size')
    parser.add_argument('--crop-size', type=int, default=480, help='crop image size')
    parser.add_argument('--workers', '-j', type=int, default=12, metavar='N', help='dataloader threads')
    parser.add_argument('--pretrained_base', action='store_true', default=True, help='Backbone Pretrained')
    parser.add_argument('--pretrained', action='store_true', default=False, help='Backbone Pretrained')
    parser.add_argument('--jpu', action='store_true', default=True, help='Backbone Pretrained')
    # cuda setting
    parser.add_argument('--gpus', type=str, default='0', help='GPU')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default='', help='put the path to resuming file if needed~/.torch/models/psp_resnet50_ade20k.pth')
    parser.add_argument('--save-epoch', type=int, default=10, help='save model every checkpoint-epoch')
    parser.add_argument('--log-iter', type=int, default=10, help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1, help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False, help='skip validation during training')
    parser.add_argument('--save_pred', action='store_true', default=True, help='save pred result during training')
    parser.add_argument('--pths', type=str, default='2024-08-14_V1.0', help='')
    parser.add_argument('--pth_path', type=str, default='mtcnet.pth', help='BaseModel')
    
    args = parser.parse_args()

    args.data_root = data_root

    # 设置训练采用的GPU情况
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        torch.backends.cudnn.enable =True
        torch.backends.cudnn.benchmark = True
        args.device = 'cuda:{}'.format(args.gpus.split(',')[0]) # 默认将第一个GPU作为主GPU
    else:
        args.device = 'cpu'

    return args

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.args.gpus = [int(i) for i in args.gpus.split(',')]
        self.version = args.version

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        
        # dataset and dataloader
        data_kwargs = {'base_size': args.base_size, 'crop_size': args.crop_size}
        train_dataset = IndoorContextDataSet(root=args.data_root, split='train', transform=input_transform, **data_kwargs)
        val_dataset = IndoorContextDataSet(root=args.data_root, split='val', transform=input_transform, **data_kwargs)
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                           pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_model(
            version=args.version,
            nclass=nclass,
            backbone=args.backbone,
            pretrained_base=args.pretrained_base,
            norm_layer=BatchNorm2d,
            jpu=args.jpu,
            pth_path=args.pth_path).to(self.device)
        # resume checkpoint if needed
        if args.resume != '':
            if os.path.exists(args.resume):
                _, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))
            else:
                print('Load checkpoint failed')
                sys.exit()

        # create criterion
        self.criterion = MixSoftmaxCrossEntropyLoss().to(self.device)
        
        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': args.lr})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': args.lr * 10})
        self.optimizer = torch.optim.SGD(params_list,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=args.power,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)
        
        if self.args.device != 'cpu' and len(self.args.gpus) > 1: # 多GPU训练
            self.model = nn.DataParallel(self.model)
            cudnn.enable = True
            cudnn.benchmark = True

        # evaluation metrics
        self.metric = []
        for i in range(task_num):
            self.metric.append(Metric(nclass[i]))

        self.best_pred = 0.0
        
    def train(self):
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()

        for iteration, (images, labels, _) in enumerate(self.train_loader, self.args.start_epoch):
            iteration = iteration + 1
            self.lr_scheduler.step()

            images = images.to(self.device)
            labels = [i.to(self.device) for i in labels]

            outputs = self.model(images)

            loss = []
            for i in range(pretrained_tasks, task_num):
                loss_dict = self.criterion(outputs[i:i+1], labels[i])
                loss.append(sum(loss for loss in loss_dict.values()))
            
            losses = sum(loss)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Loss_dict: {} || Cost Time: {} || Estimated Time: {}".format(
                    iteration, max_iters, self.optimizer.param_groups[0]['lr'],
                    losses,
                    ' '.join([str(round(i.item() ,2)) for i in loss]),
                    str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation()
                self.model.train()

        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info("Total training time: {} ({:.4f}s / it)".format(total_training_str, total_training_time / max_iters))

    def validation(self):
        is_best = False
        for i in range(task_num):
            self.metric[i].reset()
        
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (images, labels, _) in enumerate(self.val_loader):
            images = images.to(self.device)
            labels = [i.to(self.device) for i in labels]

            with torch.no_grad():
                outputs = model(images)
            
            # 精度验证
            new_pred = []
            for j in range(pretrained_tasks, task_num):
                self.metric[j].update(outputs[j], labels[j])
                AllAcc, ClsAcc, IoU, mIoU = self.metric[j].get()
                logger.info("Sample-Task{:d}: {:d}, Validation AllAcc: {:.3f}, mIoU: {:.3f}".format(j, i + 1, AllAcc, mIoU))
                logger.info("Sample-Task{:d}: {:d}, ClassAcc: {}, IoU: {}".format(j, i + 1, ClsAcc.__str__(), IoU.__str__()))
                new_pred.append((AllAcc + mIoU) / 2)

        if max(new_pred) > self.best_pred:
            is_best = True
            self.best_pred = max(new_pred)
        save_checkpoint(self.model, self.args, is_best)
        synchronize()

def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.checkpoint)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'mtcnet.pth'
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'mtcnet_best.pth'
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)

if __name__ == '__main__':
    args = parse_args()

    # 生成运行目录
    save_dir = './running/'
    times = str(datetime.datetime.now().strftime('%Y-%m-%d'))
    if args.remark != '':
        timestr = times + "_v{}".format(args.version) + "_" + args.remark
    else:
        timestr = times + "_v{}".format(args.version)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    save_dir = Path(save_dir).joinpath('{}/'.format(timestr))
    save_dir.mkdir(exist_ok=True)
    log_dir = save_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir = save_dir.joinpath('checkpoint/')
    checkpoint_dir.mkdir(exist_ok=True)
    visual_dir = save_dir.joinpath('visual/')
    visual_dir.mkdir(exist_ok=True)
    args.checkpoint = checkpoint_dir.__str__()
    args.log_dir = log_dir
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        args.num_gpus = len(args.gpus.split(','))
    else:
       args.num_gpus = 1
    args.distributed = False

    args.lr = args.lr * args.num_gpus

    logger = setup_logger("MTCNet", args.log_dir, get_rank(), filename='train_log.txt')
    logger.info("Using {} GPUs".format(args.num_gpus))
    logger.info("Parameters:")
    logger.info(args)
    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
