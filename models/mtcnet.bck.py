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
        if pretrained_tasks > 0:  # 获取基础模型中的ResNet50
            # self.pretrained_net = get_pretrained_net(backbone='resnet50', pretrained_base=pretrained_base, norm_layer=norm_layer, jpu=jpu, pretrained=True, pth_path='mtcnet.pth')
            super(MTCNet, self).__init__(nclass, backbone, jpu, pretrained_base, norm_layer, **kwargs)
        else: # 重新获取ResNet50
            super(MTCNet, self).__init__(nclass, backbone, jpu, pretrained_base, norm_layer, **kwargs)
            # Common feature
            self.head = MTCHead(nclass, norm_layer=norm_layer, **kwargs)

            self.__setattr__('exclusive', ['head'])

            # tasks
            self.tasks = nn.ModuleList()
            self.task_num = task_num

            # Multi task Module
            for i in range(self.task_num):
                if len(classes_dict[i].keys()) > 2:
                    out_channel = 120
                else:
                    out_channel = 60

                if pretrained_tasks > 0:  # 已经有训练好的模型了
                    if i < pretrained_tasks:
                        self.tasks.append(self.pretrained_net.tasks[i])
                    else:
                        self.tasks.append(MTCModule(
                            nclass=nclass[i], kernal_size=5, pool_size=2, out_channel=out_channel, drop_p=0.4, inplace=False))
                else:  # 还未开始训练
                    self.tasks.append(MTCModule(
                        nclass=nclass[i], kernal_size=5, pool_size=2, out_channel=out_channel, drop_p=0.4, inplace=False))

    def forward(self, x):
        outputs = []
        if pretrained_tasks > 0:
            _, _, c3, c4 = self.pretrained_net.base_forward(x)
        else:
            _, _, c3, c4 = self.base_forward(x)
            x = self.head(c4)
            i = 0
            for xx in x:
                t = self.tasks[i](xx)
                outputs.append(t)
                i += 1

        return tuple(outputs)
