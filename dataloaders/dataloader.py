#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
数据集加载

@File        :   dataloader.py
@Time        :   2024-08-11 20:19:13
@Author      :   Jiang Jianwu
@Contact     :   fengbuxi@glut.edu.cn
@Affiliation :   Guilin University of Technology
@Version     :   1.0
"""
import sys
sys.path.append('../')
import os
import torch
import numpy as np
from PIL import Image
from .segbase import SegmentationDataset
from global_config import classes_dict, pretrained_tasks

__all__ = ['IndoorContextDataSet']

class IndoorContextDataSet(SegmentationDataset):
    '''场景级别数据集加载'''
    def __init__(self, root='../datasets/', split='train', transform=None, **kwargs):
        super(IndoorContextDataSet, self).__init__(root, split, split, transform, **kwargs)
        assert os.path.exists(root), "The data set path does not exist !"
        self.task_num = len(classes_dict)
        self.mode = split
        self.images, self.labels = self._get_data_pairs(root, split)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in folders of:" + root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        
        label = []
        for i in range(self.task_num):
            label.append(self.labels[i][index])
        
        if self.mode == 'train':
            img = self._sync_transform_no_mask(img)
        elif self.mode == 'val':
            img = self._val_sync_transform_no_mask(img)
        else:
            assert self.mode == 'testval'
            img = self._img_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32') - 1)

    def __len__(self):
        return len(self.images)

    def _get_data_pairs(self, folder, split='train'):
        img_paths = [] # img paths
        labels = [] # labels
        if split == 'train':
            img_folder = os.path.join(folder, 'train')
        else:
            img_folder = os.path.join(folder, 'test')
            
        # 确定任务数量
        task_num = self.task_num
        # 生成task_num个labels变量
        tmp_labels = {}
        for i in range(task_num):
            tmp_labels['labels_{}'.format(i)] = []
            
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                if os.path.isfile(imgpath):
                    tmp = basename.split('_')[:-1] # 不包含后缀名和序号， 长度需要与task_num一致
                    # assert len(tmp) == task_num , "请检查数据集格式，数据集标签类别与给定的类别数不一致！文件名为：" + filename
                    img_paths.append(imgpath)
                    for i in range(task_num):
                        if i < pretrained_tasks: # 额外任务无标签
                            tmp_labels['labels_{}'.format(i)].append(0)
                        else:
                            tmp_labels['labels_{}'.format(i)].append(classes_dict[i][tmp[i-pretrained_tasks]])
                else:
                    print('cannot find the image:', imgpath)
                    
        for i in range(task_num):
            labels.append( tmp_labels['labels_{}'.format(i)])
        
        return img_paths, labels


if __name__ == '__main__':
    from torchvision import transforms
    input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    data_kwargs = {'transform': input_transform, 'base_size': 520, 'crop_size': 480}
    train_dataset = IndoorContextDataSet(
        root='E:/DataSets/indoor_context_data/',
        split='test',
        **data_kwargs
    )
    print(train_dataset)
    
