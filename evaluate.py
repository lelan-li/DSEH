# coding=utf-8
# Copyright 17-12-21 下午4:18. All Rights Reserved.
# Author: Li Ning 
# E-mail: ningli2017@gmail.com
# File: evaluate.py 
# Time: 17-12-21 下午4:18

import numpy as np


class HashCodeEvaluate:
    def __init__(self, query_code, retrival_code, query_label, retrieval_label):
        """

        :type query_code: {-1,+1}^{mxq}
        :type retrival_code: {-1,+1}^{nxq}
        :type query_label: {0,1}^{mxl}
        :type retrieval_label: {0,1}^{nxl}
        """
        self.query_code = query_code
        self.retrival_code = retrival_code
        self.query_label = query_label
        self.retrieval_label = retrieval_label

    @property
    def map(self):
        num_query = self.query_label.shape[0]
        _map = 0
        for i in xrange(num_query):
            similar_mat = (np.dot(self.query_label[i, :], self.retrieval_label.transpose()) > 0).astype(np.float32)
            similar_sum = np.sum(similar_mat)
            if similar_sum == 0:
                continue
            hamm_dist = self._calc_hamming_dist(self.query_code[i, :], self.retrival_code)
            hash_sort = np.argsort(hamm_dist)
            similar_mat = similar_mat[hash_sort]
            right_sort = np.linspace(1, similar_sum, similar_sum)

            real_sort = np.asarray(np.where(similar_mat == 1)) + 1.0
            _map = _map + np.mean(right_sort / real_sort)
        _map = _map / num_query
        return _map

    def map_top_n(self, top_n=5000):
        n_query = self.query_code.shape[0]
        sim = np.dot(self.retrival_code, self.query_code)
        ids = np.argsort(-sim, axis=0)
        ap_x = []
        ground_truth = np.sum(np.dot(self.retrieval_label, self.query_label.T), axis=0)
        self.query_label[self.query_label == 0] = -1
        for i in range(n_query):
            label = self.query_label[i, :]
            if ground_truth[i] > 0:
                idx = ids[:, i]
                imatch = np.sum(self.retrieval_label[idx[0:top_n], :] == label, axis=1) > 0
                relevant_num = np.sum(imatch)
                Lx = np.cumsum(imatch)
                Px = Lx.astype(float) / np.arange(1, top_n + 1, 1)
                if relevant_num != 0:
                    ap_x.append(np.sum(Px * imatch) / relevant_num)
                else:
                    ap_x.append(np.sum(Px * imatch) / 1)
        return np.mean(np.array(ap_x))

    @staticmethod
    def _calc_hamming_dist(b_1, b_2):
        bit = b_2.shape[1]
        hamming_dist = 0.5 * (bit - np.dot(b_1, b_2.transpose()))
        return hamming_dist




