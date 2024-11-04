#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
模型加载

@File        :   __init__.py
@Time        :   2024-08-11 22:04:00
@Author      :   Jiang Jianwu
@Contact     :   fengbuxi@glut.edu.cn
@Affiliation :   Guilin University of Technology
@Version     :   1.0
"""
from .mtcnet import MTCNet

def get_model(version = '1.0', **kwargs):
    if version == '1.0': # 多任务融合的场景级认知信息提取模型 | 可配置 + 多任务
        return MTCNet(**kwargs)
    else:
        return MTCNet(**kwargs)
