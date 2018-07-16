# coding=utf-8
# Copyright 17-12-21 下午3:46. All Rights Reserved.
# Author: Li Ning 
# E-mail: ningli2017@gmail.com
# File: config.py 
# Time: 17-12-21 下午3:46

config = {
    'vggf_model_path': 'model_zoo/imagenet-vgg-f.mat',
    'image_size': [224, 224, 3],
    'nuswide': {'image': 'data/nuswide/Img.h5',
                'label': 'data/nuswide/Lab.h5'},
    'coco': {'image': 'data/coco/Img.h5',
             'label': 'data/coco/Lab.h5'},
    'imagenet': {'image': 'data/imagenet/Img.h5',
                 'label': 'data/imagenet/Lab.h5'}
}
