#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Multi-task scenario-level cognitive information extraction model

@File        :   mtcnet.py
@Time        :   2024-08-11 20:02:43
@Author      :   Jiang Jianwu
@Contact     :   fengbuxi@glut.edu.cn
@Affiliation :   Guilin University of Technology
@Version     :   1.0
"""
from .mtcbase import *

def get_pretrained_net(backbone='resnet50', pretrained_base=True, norm_layer=nn.BatchNorm2d, jpu=True, pretrained=True, pth_path='mtcnet.pth'):
    model = MTCNetBase(nclass[:pretrained_tasks], backbone, pretrained_base, norm_layer, jpu)
    if pretrained:
        base_model_path = os.path.expanduser('~') + "/.torch/models/" + pth_path
        model.load_state_dict(torch.load(base_model_path), strict=False)
    return model


class MTCNet(MTCNetBase):
    def __init__(self, nclass, backbone='resnet50', pretrained_base=True, norm_layer=nn.BatchNorm2d, jpu=True, **kwargs):
        super(MTCNet, self).__init__(nclass, backbone, pretrained_base, norm_layer, jpu, **kwargs) # 声明新的Base
        if pretrained_tasks > 0: # 调用旧的Base，主要是获取权重
            self.pretrained = get_pretrained_net(backbone='resnet50', pretrained_base=pretrained_base, norm_layer=norm_layer, jpu=jpu, pretrained=True, pth_path=kwargs['pth_path'])

        self.head = MTCHead(nclass, norm_layer=norm_layer, **kwargs)
        self.__setattr__('exclusive', ['head'])
        self.task_num = len(nclass)

        # Multi task Module
        for i in range(self.task_num):
            if len(classes_dict[i].keys()) > 2:
                out_channel = 120
            else:
                out_channel = 60

            if pretrained_tasks > 0:  # 已经有训练好的模型了
                if i > pretrained_tasks-1:
                    self.tasks.append(MTCModule(nclass=nclass[i], kernal_size=5, pool_size=2, out_channel=out_channel, drop_p=0.4, inplace=False))

    def forward(self, x):
        outputs = []
        if pretrained_tasks > 0:
            _, _, c3, c4 = self.pretrained.base_forward(x)
        else:
            _, _, c3, c4 = self.base_forward(x)
        x = self.head(c4)
        i = 0
        for xx in x:
            if i in tasks_on:
                if i < pretrained_tasks:
                    t = self.pretrained.tasks[i](xx)
                else:
                    t = self.tasks[i](xx)
            else:
                t= -1
            outputs.append(t)
            i += 1

        return tuple(outputs)
