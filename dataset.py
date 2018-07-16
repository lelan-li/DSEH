# coding=utf-8
# Copyright 17-11-29 下午9:59. All Rights Reserved.
# Author: Li Ning
# E-mail: ningli2017@gmail.com
# File: dataset.py
# Time: 17-11-29 下午9:59

import random
import h5py
import numpy as np


class DataSet:
    def __init__(self, config, args):
        # init
        self.config = config
        self.args = args
        # dataset
        self.dataset_name = self.args['dataset_name']
        print('start load data.')
        with h5py.File(self.config[self.dataset_name]['image'], 'r') as image:
            self.image_train = image['ImgTrain'][:]  # 10500
            self.image_retrieval = image['ImgDataBase'][:]  # 177821
            self.image_query = image['ImgQuery'][:]  # 2100

        with h5py.File(self.config[self.dataset_name]['label'], 'r') as label:
            self.label_train = label['LabTrain'][:]  # [10500, 21]
            self.label_retrieval = label['LabDataBase'][:]
            self.label_query = label['LabQuery'][:]

        self.n_train = len(self.image_train)
        self.n_query = len(self.image_query)
        self.n_dataset = len(self.image_retrieval)
        self.n_retrieval = self.n_dataset
        print('load data done!')
        # args
        self.batch_size = self.args['batch_size_image']
        self.batch_size_label = self.args['batch_size_label'][self.dataset_name]
        self.n_class = self.label_query.shape[1]

    def batch_image(self):
        batch_index = random.sample(range(self.n_train), self.batch_size)
        batch_image = self.image_train[batch_index, ...]
        batch_label = self.label_train[batch_index]
        return batch_image, batch_label, batch_index

    def batch_label(self):
        batch_index = random.sample(range(self.n_train), self.batch_size_label)
        batch_label = self.label_train[batch_index]
        return batch_label, batch_index

