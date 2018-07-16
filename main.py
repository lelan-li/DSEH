# coding=utf-8
# Copyright 17-12-21 下午3:46. All Rights Reserved.
# Author: Li Ning 
# E-mail: ningli2017@gmail.com
# File: main.py 
# Time: 17-12-21 下午3:46
import numpy as np
from config import config
from DSEH import DSEH

args = {
    'bit': 32,  # 16, 32, 48, 64
    'gpu': '2, 3',
    'dataset_name': 'nuswide',  # nuswide imagnet coco

    'max_memory_per_gpu': 0.9,
    'max_epoch': 200,
    'lr': {
        'nuswide': {
            'label': [np.power(0.1, x) for x in np.arange(3.0, 100, 0.5)],
            'image': [np.power(0.1, x) for x in np.arange(4.0, 100, 0.5)],
        },
        'coco': {
            'label': [np.power(0.1, x) for x in np.arange(3.0, 100, 0.5)],  
            'image': [np.power(0.1, x) for x in np.arange(3.5, 100, 0.5)],
        },
        'imagenet': {
            'label': [np.power(0.1, x) for x in np.arange(3.0, 100, 0.5)],
            'image': [np.power(0.1, x) for x in np.arange(4.5, 100, 0.5)]},
    },
    's': {
        'nuswide': {
            'label_fea': 0.999,
            'label_hash': 0.999,
            'image': 0.999,
        },
        'coco': {
            'label_fea': 0.8,
            'label_hash': 0.8,
            'image': 0.999,
        },
        'imagenet': {
            'label_fea': 5.,
            'label_hash': 5.,
            'image': 5.,
        }},
    'gamma': {
        'nuswide': {
            'label_hash_label': 1.,
            'label_pairwise_hash': 1.,
            'label_quantization': 1.,
            'label_pairwise_feature': 1.,

            'image_hash': 1.,
            'image_pairwise_feature': 1.,
        },
        'coco': {
            'label_hash_label': 1.,  
            'label_pairwise_hash': 1.,
            'label_quantization': 1.,
            'label_pairwise_feature': 1.,

            'image_hash': 1.,
            'image_pairwise_feature': 1.,
        },
        'imagenet': {
            'label_hash_label': 0.1,
            'label_pairwise_hash': 1.,
            'label_quantization': 1.,
            'label_pairwise_feature': 0.,

            'image_hash': 1.,
            'image_pairwise_feature': 1.,
        }},
    'batch_size_image': 128,
    'batch_size_label': {
        'nuswide': 32,
        'coco': 32,
        'imagenet': 32,
    },
    'batch_size_val': 256,
    'net_name': 'vggf',  

    'test_rate': 1,
    'save_rate': 1,
    'log_rate': 1,
    'record_rate': 1,
    'record_save_rate': 10,
    'print_tag_rate': 10,

    'debug': False,

    'record_file': {'epoch': [],
                    'all_loss': [],
                    'lr': [],
                    'loss_target_hash': [],
                    'loss_target_class': [],
                    'S_hash_map': [],
                    'S_val_acc': [],
                    'S_train_acc': [],
                    'T_hash_map': [],
                    'T_val_acc': [],
                    'T_train_acc': [],
                    'T_batch_train_acc': [],
                    'S_batch_train_acc': [],
                    'time': []},
}

model = DSEH(config, args)
model.main()

