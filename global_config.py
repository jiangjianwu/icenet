#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Global Configuration

@File        :   global_config.py
@Time        :   2024-08-11 22:12:26
@Author      :   Jiang Jianwu
@Contact     :   fengbuxi@glut.edu.cn
@Affiliation :   Guilin University of Technology
@Version     :   1.0
"""
# Dataset Path
data_root = '/path/to/your/dir/'

# Task switch allows you to manually control the state of the task, it does not work during training but only during application.
tasks_on = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Number of tasks already trained
pretrained_tasks = 0   

# Category dictionary !!! Very important configuration !!!
classes_dict = [
    {
        'bedroom': 0,
        'parlor': 1,
        'kitchen': 2,
        'studyroom': 3,
        'bathroom': 4,
        'balcony': 5,
        'office': 6,
        'conference': 7,
        'supermarket': 8,
        'mall': 9
    },
    # {
    #     'night': 0,
    #     'day': 1
    # },
    # {
    #     'nobody': 0,
    #     'somebody': 1
    # },
    # {
    #     'impassable': 0,
    #     'passable': 1
    # },
    # {
    #     'stone': 0,
    #     'wooden': 1
    # },
    # {
    #     'dry': 0,
    #     'wet': 1
    # },
]


nclass = []
for classes in classes_dict:
    nclass.append(len(classes.keys()))
task_num = len(nclass)

