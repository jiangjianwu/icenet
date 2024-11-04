#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
MtcNet基础模型

@File        :   mtcbase.py
@Time        :   2024-08-12 21:30:30
@Author      :   Jiang Jianwu
@Contact     :   fengbuxi@glut.edu.cn
@Affiliation :   Guilin University of Technology
@Version     :   1.0
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from global_config import classes_dict, nclass, pretrained_tasks, task_num, tasks_on
from .segbase import SegBaseModel


class MTCNetBase(SegBaseModel):
    def __init__(self, nclass, backbone='resnet50', pretrained_base=True, norm_layer=nn.BatchNorm2d, jpu=True, **kwargs):
        super(MTCNetBase, self).__init__(sum(nclass),backbone, jpu, pretrained_base, **kwargs)
        # Common feature
        self.head = MTCHead(nclass, norm_layer=norm_layer, **kwargs)

        self.__setattr__('exclusive', ['head'])

        # tasks
        self.tasks = nn.ModuleList()
        if pretrained_tasks > 0:
            self.task_num = pretrained_tasks
        else:
            self.task_num = task_num

        # Multi task Module
        for i in range(self.task_num):
            if len(classes_dict[i].keys()) > 2:
                out_channel = 120
            else:
                out_channel = 60

            self.tasks.append(MTCModule(
                nclass=nclass[i], kernal_size=5, pool_size=2, out_channel=out_channel, drop_p=0.4, inplace=False))


# ===================基础模块========================


def _MTC1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels, **
                   ({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )


class _PyramidPooling(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _MTC1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _MTC1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _MTC1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _MTC1x1Conv(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)),
                              size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)),
                              size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)),
                              size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)),
                              size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)


class MTCHead(nn.Module):
    def __init__(self, nclass=[12, 2, 2, 2], norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(MTCHead, self).__init__()
        self.MTC = _PyramidPooling(
            2048, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )
        # Multi task output
        self.out = nn.ModuleList()
        task_num = len(nclass)
        for i in range(task_num):
            self.out.append(nn.Conv2d(512, nclass[i], 1))

    def forward(self, x):
        x = self.MTC(x)
        x = self.block(x)
        res = []
        task_num = len(nclass)
        for i in range(task_num):
            res.append(self.out[i](x))
        return tuple(res)


class MTCModule(nn.Module):
    def __init__(self, nclass, kernal_size=5, pool_size=2, out_channel=120, drop_p=0.4, inplace=False, **kwargs):
        super(MTCModule, self).__init__()
        self.nclass = nclass
        self.hidden_channel = kernal_size * 2 + pool_size
        self.block = nn.Sequential(
            nn.Conv2d(nclass, nclass, kernal_size),
            nn.ReLU(inplace),
            nn.MaxPool2d(pool_size, pool_size),
            nn.Conv2d(nclass, nclass, kernal_size),
            nn.ReLU(inplace),
            nn.MaxPool2d(pool_size, pool_size)
        )
        self.out = nn.Sequential(
            nn.Linear(nclass * self.hidden_channel *
                      self.hidden_channel, out_channel),
            nn.Linear(out_channel, int(out_channel / 2)),
            nn.Dropout(drop_p),
            nn.Linear(int(out_channel / 2), nclass)
        )

    def forward(self, x):
        x = self.block(x)
        x = x.view(-1, self.nclass * self.hidden_channel * self.hidden_channel)
        x = self.out(x)
        return x
