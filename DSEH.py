# coding=utf-8
# Copyright 17-12-21 下午3:46. All Rights Reserved.
# Author: Li Ning 
# E-mail: ningli2017@gmail.com
# File: mutilabel_hash.py 
# Time: 17-12-21 下午3:46

import os
import random
import numpy as np
from tqdm import tqdm

import scipy.io as sio
from PIL import Image
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from dataset import DataSet
from evaluate import HashCodeEvaluate
from net_vggf import VggF, LabelNetVgg


class DSEH:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        # dataset
        self.dataset = DataSet(self.config, self.args)
        self.n_class = self.dataset.n_class
        self.n_train = self.dataset.n_train
        self.n_query = self.dataset.n_query
        self.n_retrieval = self.dataset.n_retrieval
        self.n_dataset = self.dataset.n_dataset
        self.dataset_name = self.dataset.dataset_name
        # train args
        self.bit = self.args['bit']
        self._lr_image = self.args['lr'][self.dataset_name]['image']
        self._lr_label = self.args['lr'][self.dataset_name]['label']
        self.max_epoch = self.args['max_epoch']
        self.image_size = self.config['image_size']
        self.batch_size_image = self.args['batch_size_image']
        self.batch_size_label = self.args['batch_size_label'][self.dataset_name]
        self.batch_size_val = self.args['batch_size_val']
        # net args
        self.gamma = self.args['gamma'][self.dataset_name]
        self.s = self.args['s'][self.dataset_name]
        self.epoch_label = 0
        self.epoch_image = 0
        self.net_name = self.args['net_name']
        # record
        self.record_file = self.args['record_file']
        # net init
        self.lr = tf.placeholder('float32', (), name='lr')
        self.label_keep_prop = tf.placeholder('float32', (), name='label_keep_prob')
        self.image_keep_prob = tf.placeholder('float32', (), name='image_keeep_prob')
        self.image_net_input = tf.placeholder('float32', [None] + self.image_size, name='image_net_input')
        self.label_net_input = tf.placeholder('float32', [None, self.n_class], name='label_net_input')
        image_net = VggF(self.n_class, self.image_keep_prob, self.config, self.args)
        label_net = LabelNetVgg(self.n_class, self.label_keep_prop, self.config, self.args)

        self.n_feature_label_net = 512  # label_net.layer_node_2
        self.n_feature_image_net = self.n_feature_label_net

        self.label_net, self.label_regular_loss = label_net(self.label_net_input)
        self.image_net, self.image_mean_one, self.image_weights = image_net(self.image_net_input)
        self.image_mean = np.repeat(self.image_mean_one[np.newaxis, :, :, :], self.batch_size_image, axis=0).astype(
            np.float32)
        self.image_mean_val = np.repeat(self.image_mean_one[np.newaxis, :, :, :], self.batch_size_val, axis=0).astype(
            np.float32)

        self.image_hash = self.image_net['image_hash']
        self.image_hash_sigmoid = self.image_net['image_hash_sigmoid']
        self.image_hash_tanh = self.image_net['image_hash_tanh']
        self.image_label = self.image_net['image_label']
        self.image_label_sigmoid = self.image_net['image_label_sigmoid']
        self.image_feature = self.image_net['image_feature']

        self.label_hash = self.label_net['label_hash']
        self.label_hash_label = self.label_net['label_hash_label']
        self.label_hash_label_sigmoid = self.label_net['label_hash_label_sigmoid']
        self.label_label = self.label_net['label_label']
        self.label_feature = self.label_net['label_feature']
        self.similar_mat = self.clac_similar(self.dataset.label_train, self.dataset.label_train)
        # label net
        self.batch_label_similar_mat = tf.placeholder('float32', [self.n_train, self.batch_size_label],
                                                      name='batch_label_similar_mat')
        self._label_all_hash_code = np.random.randn(self.n_train, self.bit)
        self.label_all_hash_code = tf.placeholder('float32', [self.n_train, self.bit], name='all_hash_code')
        self.batch_label_label = tf.placeholder('float32', [self.batch_size_label, self.n_class],
                                                name='batch_label_label')
        self._all_label_sign_hash = np.random.randn(self.n_train, self.bit)
        self._label_all_feature = np.random.randn(self.n_train, self.n_feature_label_net)
        self.label_all_feature = tf.placeholder('float32', [self.n_train, self.n_feature_label_net],
                                                name='label_all_feature')
        self.batch_label_ones = tf.constant(np.ones([self.batch_size_label, 1], 'float32'))
        # image_net
        self._image_all_feature = np.random.randn(self.n_train, self.n_feature_image_net)
        self.image_all_feature = tf.placeholder('float32', [self.n_train, self.n_feature_image_net],
                                                name='image_all_feature')
        self.batch_image_label = tf.placeholder('float32', [self.batch_size_image, self.n_class],
                                                name='batch_image_label')
        self.batch_image_hash = tf.placeholder('float32', [self.batch_size_image, self.bit],
                                               name='batch_image_hash')
        self.batch_image_sigmoid = tf.placeholder('float32', [self.batch_size_image, self.bit],
                                                  name='batch_image_sigmoid')
        self.batch_image_feature = tf.placeholder('float32', [self.batch_size_image, self.n_feature_image_net],
                                                  name='batch_image_feature')
        self.batch_image_similar_mat = tf.placeholder('float32', [self.n_train, self.batch_size_image],
                                                      name='batch_label_similar_mat')
        self.image_all_hash_code = tf.placeholder('float32', [self.n_train, self.bit], name='image_all_hash_code')
        # net build
        self.build_graph()


    def build_graph(self):

        self.label_loss()
        self.loss_all_label = self.label_loss()
        optimizer_label = tf.train.AdamOptimizer(self.lr)
        gradient_label = optimizer_label.compute_gradients(self.loss_all_label)
        clip_gradient_label = [(self.clip_if_not_none(grad), val) for grad, val in gradient_label]
        self.train_label_op = optimizer_label.apply_gradients(clip_gradient_label)

        if self.dataset_name == 'imagenet':
            self.image_loss()
            self.loss_all_image = self.image_loss()
            optimizer_image = tf.train.AdamOptimizer(self.lr)
            gradient_image = optimizer_image.compute_gradients(self.loss_all_image)
            clip_gradient_image = [(self.clip_if_not_none(grad), val) for grad, val in gradient_image]
            self.train_image_op = optimizer_image.apply_gradients(clip_gradient_image)
        else:
            var_net = tf.trainable_variables()[10: 24]
            var_self = tf.trainable_variables()[24:]
            opt_net = tf.train.AdamOptimizer(0.1 * self.lr)
            opt_self = tf.train.AdamOptimizer(self.lr)
            self.image_loss()
            self.loss_all_image = self.image_loss()
            gradient_image = tf.gradients(self.loss_all_image, var_net + var_self)
            grads_net = gradient_image[:len(var_net)]
            grads_self = gradient_image[len(var_net):]
            train_op_net = opt_net.apply_gradients(zip(grads_net, var_net))
            train_op_self = opt_self.apply_gradients(zip(grads_self, var_self))
            self.train_image_op = tf.group(train_op_net, train_op_self)

    def main(self):
        gpuconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=self.args['max_memory_per_gpu']))
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args['gpu']
        with tf.Session(config=gpuconfig) as self.sess:
            if self.args['debug']:
                self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            tf.global_variables_initializer().run()
            for self.big_epoch in range(3):
                self.train_label()
                self.train_image()

    def train_label(self):
        if self.big_epoch != 0:
            image_input_dir = 'label_hash_' + self.dataset_name + '/' + str(self.bit) + '_image_hash.mat'
            self._label_all_hash_code = sio.loadmat(image_input_dir)['image_hash']
            self._label_all_feature = sio.loadmat(image_input_dir)['image_feature']
        if self.dataset_name == 'imagenet':
            n = 4 - self.big_epoch * 1
        elif self.dataset_name == 'nuswide':
            n = 30 - self.big_epoch * 10
        elif self.dataset_name == 'coco':
            n = 40 - self.big_epoch * 10

        self.save_all_label_hash()
        self.test_label()
        for self.epoch_label in tqdm(xrange(n)):

            self.lr_i = self._lr_label[int(self.epoch_label / 10 + self.big_epoch)]

            for i_batch in xrange(int(self.n_train / self.batch_size_label)):
                batch_label, batch_index = self.dataset.batch_label()

                batch_label_tanh = np.copy(batch_label)
                batch_label_tanh[batch_label_tanh == 1] = 1.  # -1.0
                batch_label_tanh[batch_label_tanh == 0] = 0.

                _batch_hash, _batch_feature = self.sess.run([self.label_hash, self.label_feature],
                                                            feed_dict={self.label_net_input: batch_label_tanh,
                                                                       self.label_keep_prop: 1.0})
                self._label_all_hash_code[batch_index] = _batch_hash
                self._label_all_feature[batch_index] = _batch_feature
                _batch_similar_mat = self.similar_mat[:, batch_index]

                _ = self.sess.run(
                    [self.train_label_op],
                    feed_dict={self.lr: self.lr_i,
                               self.label_keep_prop: 1.0,
                               self.label_net_input: batch_label_tanh,
                               self.label_all_hash_code: self._label_all_hash_code,
                               self.label_all_feature: self._label_all_feature,
                               self.batch_label_similar_mat: _batch_similar_mat,
                               self.batch_label_label: batch_label})
            self._all_label_sign_hash = np.sign(self._label_all_hash_code)
            if self.epoch_label == 0 or (self.epoch_label + 1) % 10 == 0 or (self.epoch_label + 1) == n:
                self.test_label()
                self.save_all_label_hash()

    def train_image(self):
        image_input_dir = 'label_hash_' + self.dataset_name + '/' + str(self.bit) + '_label_hash.mat'
        self._all_label_sign_hash = sio.loadmat(image_input_dir)['label_hash']
        self._all_label_feature = sio.loadmat(image_input_dir)['label_feature']

        n = 40
        for self.epoch_image in tqdm(xrange(int(n))):
            self.lr_i = self._lr_image[int(self.epoch_image / 10. + self.big_epoch)]
            for i_batch in xrange(int(self.n_train / self.batch_size_image)):
                batch_image, batch_label, batch_index = self.dataset.batch_image()
                batch_image = self.path_to_image_mirror(batch_image)
                batch_image -= self.image_mean
                _batch_feature = self.sess.run([self.image_feature],
                                               feed_dict={self.image_keep_prob: 0.5,
                                                          self.image_net_input: batch_image})
                self._image_all_feature[batch_index] = _batch_feature
                _batch_similar_mat = self.similar_mat[:, batch_index]
                _image_hash_sigmoid = np.copy((self._all_label_sign_hash[batch_index]))
                _image_hash_sigmoid[_image_hash_sigmoid >= 0] = 1.0
                _image_hash_sigmoid[_image_hash_sigmoid < 0] = 0.0
                _image_hash_tanh = np.copy((self._all_label_sign_hash[batch_index]))
                _image_feature = np.copy((self._all_label_feature[batch_index]))
                _ = self.sess.run(
                    [self.train_image_op],
                    feed_dict={self.lr: self.lr_i,
                               self.image_keep_prob: 0.5,
                               self.image_net_input: batch_image,
                               self.batch_image_hash: _image_hash_tanh,
                               self.batch_image_sigmoid: _image_hash_sigmoid * 0.999,
                               self.batch_image_feature: _image_feature,
                               self.image_all_feature: self._all_label_feature,
                               self.image_all_hash_code: self._all_label_sign_hash,
                               self.batch_image_similar_mat: _batch_similar_mat,
                               self.batch_image_label: batch_label})
            if self.epoch_image == 0 or self.epoch_image % 10 == 0 or self.epoch_image == (n - 1):
                self.test_image()
            self.lr_i *= 0.9
        self.save_all_image_hash()

    def label_loss(self):
        label_label_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.label_label, labels=self.batch_label_label))
        self.label_label_loss = tf.div(label_label_loss, self.batch_size_label * self.n_class)

        label_hash_label_loss = tf.nn.l2_loss(self.batch_label_label - self.label_hash_label_sigmoid)
        self.label_hash_label_loss = tf.div(label_hash_label_loss, self.batch_size_label * self.n_class)

        hi_t_hj_1 = 1.0 / 2.0 * tf.matmul(self.label_all_hash_code, tf.transpose(self.label_hash))
        pairwise_loss_hash = -tf.reduce_sum(  # TODO
            tf.multiply(self.batch_label_similar_mat * self.s['label_hash'], hi_t_hj_1) - tf.log(
                1.0 + tf.exp(hi_t_hj_1)))
        self.label_pairwise_hash_loss = tf.div(pairwise_loss_hash, float(self.n_train * self.batch_size_label))

        all_one = tf.constant(np.ones([self.batch_size_label, self.bit]))
        self.label_quantization_loss = 0.01 * tf.reduce_sum(
            tf.losses.absolute_difference(all_one, tf.abs(self.label_hash)))

        hi_t_hj_2 = 1.0 / 2.0 * tf.matmul(self.label_all_feature, tf.transpose(self.label_feature))
        pairwise_loss_feature = -tf.reduce_sum(
            tf.multiply(self.batch_label_similar_mat * self.s['label_fea'], hi_t_hj_2) - tf.log(
                1.0 + tf.exp(hi_t_hj_2)))

        self.label_pairwise_feature_loss = tf.div(pairwise_loss_feature,
                                                  float(self.n_train * self.batch_size_label))
        self.label_pairwise_feature_loss = tf.div(pairwise_loss_feature, float(self.n_train * self.batch_size_label))

        label_loss = (self.gamma['label_hash_label'] * self.label_hash_label_loss +
                      self.gamma['label_pairwise_hash'] * self.label_pairwise_hash_loss +
                      self.gamma['label_quantization'] * self.label_quantization_loss +
                      self.gamma['label_pairwise_feature'] * self.label_pairwise_feature_loss)
        return label_loss

    def image_loss(self):
        image_label_loss = tf.nn.l2_loss(self.image_label_sigmoid - self.batch_image_label)

        self.image_label_loss = tf.div(tf.reduce_sum(image_label_loss), float(self.batch_size_image * self.n_class))

        image_hash_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=10. * self.image_hash, labels=self.batch_image_sigmoid)
        self.image_hash_loss = tf.div(tf.reduce_sum(image_hash_loss), float(self.batch_size_image * self.bit))

        self.image_hash_abs_loss = tf.losses.absolute_difference(self.image_hash_sigmoid, self.batch_image_sigmoid)

        all_one = tf.constant(np.ones([self.batch_size_image, self.bit]))
        image_quantization_loss = tf.reduce_sum(
            tf.losses.absolute_difference(all_one, tf.abs(self.image_hash_tanh)))

        self.image_quantization_loss = 0.005 * image_quantization_loss

        hi_t_hj_2 = 1.0 / 2.0 * tf.matmul(self.image_all_feature, tf.transpose(self.image_feature))
        pairwise_loss_feature = -tf.reduce_sum(  # TODO
            tf.multiply(self.batch_image_similar_mat * 0.999, hi_t_hj_2) - tf.log(1.0 + tf.exp(hi_t_hj_2)))
        self.image_pairwise_feature_loss = tf.div(pairwise_loss_feature, float(self.n_train * self.batch_size_image))

        image_loss = (0.0 * self.image_label_loss +
                      self.gamma['image_hash'] * self.image_hash_loss +
                      0.0 * self.image_quantization_loss +
                      self.gamma['image_pairwise_feature'] * self.image_pairwise_feature_loss +
                      0.0 * self.image_hash_abs_loss)

        return image_loss

    def test_image(self):
        hash_query, label_query = self.generate_image_hash('query')
        hash_retrieval, label_retrieval = self.generate_image_hash('retrieval')
        hash_train, label_train = self.generate_image_hash('train')
        hash_other = np.vstack((hash_train, hash_retrieval))
        label_other = np.vstack((self.dataset.label_train, self.dataset.label_retrieval))

        hash_eval = HashCodeEvaluate(hash_query, hash_other, self.dataset.label_query, label_other)
        query_label_acc = self.clac_acc(label_query, self.dataset.label_query)
        hash_name = ('hash_code_' + self.dataset_name + '/new_' + self.dataset_name + '_' +
                     str(self.bit) + '_bit_' + str(self.epoch_label) + '_epoch_' + str(hash_eval.map) + '.mat')
        sio.savemat(hash_name, {
            'query_label': self.dataset.label_query, 'query_hash': hash_query,
            'retri_label': label_other, 'retri_hash': hash_other})

        print('Hash_map: %.5f     query_label_acc: %.5f' %
              (hash_eval.map, query_label_acc))

    def test_label(self):
        hash_query, hash_label_query, label_query = self.generate_label_hash('query')
        hash_retrieval, hash_label_retrieval, label_retrieval = self.generate_label_hash('retrieval')
        hash_train, hash_label_train, label_train = self.generate_label_hash('train')

        hash_eval = HashCodeEvaluate(hash_query, hash_retrieval, self.dataset.label_query, self.dataset.label_retrieval)
        hash_train_eval = HashCodeEvaluate(hash_train, hash_train, self.dataset.label_train, self.dataset.label_train)
        query_label_acc = self.clac_acc(label_query, self.dataset.label_query)
        retrieval_label_acc = self.clac_acc(label_retrieval, self.dataset.label_retrieval)
        query_hash_label_acc = self.clac_acc(hash_label_query, self.dataset.label_query)
        retrieval_hash_label_acc = self.clac_acc(hash_label_retrieval, self.dataset.label_retrieval)
        train_hash_label_acc = self.clac_acc(hash_label_train, self.dataset.label_train)

        print('Hash_map: %.5f    query_label_acc: %.5f    retrieval_label_acc:%.5f '
              '  hash_label_q_acc:%.5f   hash_label_r_acc:%.5f    train_map: %.5f   train_hash_label_acc:%.5f' %
              (hash_eval.map, query_label_acc, retrieval_label_acc, query_hash_label_acc, retrieval_hash_label_acc,
               hash_train_eval.map, train_hash_label_acc))

    def generate_label_hash(self, tag):
        n = eval('self.n_' + tag)
        index = np.linspace(0, n - 1, n).astype(int)
        hash_ = np.zeros([n, self.bit], dtype=np.float32)
        hash_label = np.zeros([n, self.n_class], dtype=np.float32)
        label = np.zeros([n, self.n_class], dtype=np.float32)
        n_batch = int(n / self.batch_size_val) + 1

        for _i in range(n_batch):
            ind = index[_i * self.batch_size_val: min((_i + 1) * self.batch_size_val, n)]
            batch_label = eval('self.dataset.label_' + tag)[ind]

            label_hash_batch, label_hash_label_batch, label_label_batch = self.sess.run([
                self.label_hash,
                self.label_hash_label_sigmoid,
                self.label_label],
                feed_dict={self.label_net_input: batch_label,
                           self.label_keep_prop: 1.0})
            label[ind, :] = label_label_batch
            hash_[ind, :] = label_hash_batch
            hash_label[ind, :] = label_hash_label_batch
        hash_ = np.sign(hash_)

        return hash_, hash_label, label

    def generate_image_hash(self, tag):
        n = eval('self.n_' + tag)
        index = np.linspace(0, n - 1, n).astype(int)
        hash_ = np.zeros([n, self.bit], dtype=np.float32)
        label = np.zeros([n, self.n_class], dtype=np.float32)
        n_batch = int(n / self.batch_size_val) + 1
        for _i in range(n_batch):
            ind = index[_i * self.batch_size_val: min((_i + 1) * self.batch_size_val, n)]
            batch_image = eval('self.dataset.image_' + tag)[ind]
            batch_image = self.path_to_image(batch_image)
            if _i != n_batch - 1:
                batch_image -= self.image_mean_val
            else:
                mean_pixel = np.repeat(self.image_mean_one[np.newaxis, :, :, :], len(ind), axis=0).astype(np.float32)
                batch_image -= mean_pixel
            hash_batch, label_batch = self.sess.run([
                self.image_hash_tanh,
                self.image_label_sigmoid],
                feed_dict={self.image_keep_prob: 1.0,
                           self.image_net_input: batch_image})
            label[ind, :] = label_batch
            hash_[ind, :] = hash_batch
        hash_ = np.sign(hash_)
        return hash_, label

    def save_all_label_hash(self):
        index_train = np.linspace(0, self.n_train - 1, self.n_train).astype(int)
        hash_train = np.zeros([self.n_train, self.bit], dtype=np.float32)

        feature_train = np.zeros([self.n_train, self.n_feature_label_net], dtype=np.float32)
        n_batch = int(self.n_train / self.batch_size_val) + 1

        for _i in range(n_batch):
            ind = index_train[_i * self.batch_size_val: min((_i + 1) * self.batch_size_val, self.n_train)]
            batch_label = self.dataset.label_train[ind]
            label_hash_batch, label_feature_batch = self.sess.run([
                self.label_hash,
                self.label_feature],
                feed_dict={self.label_net_input: batch_label,
                           self.label_keep_prop: 1.})
            hash_train[ind, :] = label_hash_batch
            feature_train[ind, :] = label_feature_batch
        hash_train = np.sign(hash_train)
        if not os.path.exists('label_hash_' + self.dataset_name + '/'):
            os.mkdir('label_hash_' + self.dataset_name + '/')
        hash_name = 'label_hash_' + self.dataset_name + '/' + str(self.bit) + '_label_hash.mat'
        sio.savemat(hash_name, {'label_hash': np.array(hash_train), 'label_feature': np.array(feature_train)})
        self.print_hash_class(hash_train)

    def save_all_image_hash(self):
        index_train = np.linspace(0, self.n_train - 1, self.n_train).astype(int)
        hash_train = np.zeros([self.n_train, self.bit], dtype=np.float32)

        feature_train = np.zeros([self.n_train, self.n_feature_image_net], dtype=np.float32)
        n_batch = int(self.n_train / self.batch_size_val) + 1

        for _i in range(n_batch):
            ind = index_train[_i * self.batch_size_val: min((_i + 1) * self.batch_size_val, self.n_train)]
            batch_image = self.path_to_image(self.dataset.image_train[ind])
            image_hash_batch, image_feature_batch = self.sess.run([
                self.image_hash,
                self.image_feature],
                feed_dict={self.image_net_input: batch_image,
                           self.image_keep_prob: 1.})
            hash_train[ind, :] = image_hash_batch
            feature_train[ind, :] = image_feature_batch
        hash_train = np.sign(hash_train)
        if not os.path.exists('label_hash_' + self.dataset_name + '/'):
            os.mkdir('label_hash_' + self.dataset_name + '/')
        hash_name = 'label_hash_' + self.dataset_name + '/' + str(self.bit) + '_image_hash.mat'
        sio.savemat(hash_name, {'image_hash': np.array(hash_train), 'image_feature': np.array(feature_train)})
        self.print_hash_class(hash_train)

    @staticmethod
    def clac_similar(label_1, label_2):
        similar_mat = (np.dot(label_1, label_2.transpose()) > 0).astype(int)
        return similar_mat

    @staticmethod
    def clac_acc(label_output, label):
        acc_1 = np.multiply(label_output, label)
        acc_0 = np.multiply(1. - label_output, 1. - label)
        acc = (acc_1 + acc_0).sum() / float(label.shape[0] * label.shape[1])
        return acc

    @staticmethod
    def print_hash_class(hash_code):
        hash_code = [tuple(i) for i in hash_code]
        print('hash_class: %s' % len(set(hash_code)))

    @staticmethod
    def path_to_image_mirror(images_path):
        images = []
        for path in images_path:
            image = Image.open(path)  # .resize((224, 224))
            if random.randint(0, 1):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = np.asarray(image).astype(np.float32)
            image = image[np.newaxis, ...]
            images.append(image)
        images = np.concatenate(images, axis=0)
        return images

    @staticmethod
    def path_to_image(images_path):
        images = []
        for path in images_path:
            image = Image.open(path)
            image = np.asarray(image).astype(np.float32)
            image = image[np.newaxis, ...]
            images.append(image)
        images = np.concatenate(images, axis=0)
        return images

    @staticmethod
    def clip_if_not_none(gradient):
        if gradient is None:
            return
        return tf.clip_by_value(gradient, -1, 1)
