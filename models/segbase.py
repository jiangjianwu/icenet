"""Base Model for Semantic Segmentation"""

import torch.nn as nn
from .nn import JPU
from .base_models.resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, backbone='resnet50', jpu=False, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated = False if jpu else True
        self.aux = False
        self.nclass = nclass
        if 'pth_path' in kwargs.keys():
            del(kwargs['pth_path'])
        if backbone == 'resnet50':
            if pretrained_base: # 预训练模型的类别数为1000
                self.pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, nclass = 1000, **kwargs)
            else:
                self.pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, nclass = nclass, **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet152':
            self.pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.jpu = JPU([512, 1024, 2048], width=512, **kwargs) if jpu else None

    def base_forward(self, x): # 
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x) # [2, 3, 480, 480]
        x = self.pretrained.bn1(x) # [2, 128, 240, 240]
        x = self.pretrained.relu(x) # [2, 128, 240, 240]
        x = self.pretrained.maxpool(x) # [2, 128, 120, 120]
        c1 = self.pretrained.layer1(x) # [2, 256, 120, 120]
        c2 = self.pretrained.layer2(c1) # [2, 512, 60, 60]
        c3 = self.pretrained.layer3(c2) # [2, 1024, 60, 60]
        c4 = self.pretrained.layer4(c3) # [2, 2048, 60, 60]

        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred
