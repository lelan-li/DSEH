# coding=utf-8
# Copyright 17-12-21 上午11:55. All Rights Reserved.
# Author: Li Ning 
# E-mail: ningli2017@gmail.com
# File: net_vggf.py
# Time: 17-12-21 上午11:55
import tensorflow as tf
import scipy.io as sio


class VggF:
    def __init__(self, n_class, keep_prob, config, args):
        self.model_path = config['vggf_model_path']
        self.data = sio.loadmat(self.model_path)
        self.bit = args['bit']
        self.n_class = n_class
        self.keep_prob = keep_prob
        self.layers = (
            'conv1', 'relu1', 'norm1', 'pool1', 'conv2', 'relu2', 'norm2', 'pool2', 'conv3', 'relu3',
            'conv4', 'relu4', 'conv5', 'relu5', 'pool5',
            'fc6', 'relu6', 'fc7', 'relu7')
        self.weights = self.data['layers'][0]
        self.mean = self.data['meta']['normalization'][0][0][0][0][2]  # [124.29, 123.96, 115.22]
        self.net = {}
        self.my_weights = {}
        self.n_fc7 = 4096
        self.n_feature = 512

    def __call__(self, input_image):
        current = tf.convert_to_tensor(input_image, dtype='float32')
        for i, name in enumerate(self.layers[:]):
            if name.startswith('conv'):
                with tf.name_scope(name):
                    kernels, bias = self.weights[i][0][0][2][0]

                    bias = bias.reshape(-1)
                    pad = self.weights[i][0][0][4][0]
                    stride = self.weights[i][0][0][5][0]
                    current = self._conv_layer(current, kernels, bias, pad, stride, self.my_weights, self.net,
                                               name)
            elif name.startswith('relu'):
                with tf.name_scope(name):
                    current = tf.nn.relu(current)
                if name == 'relu6' or name == 'relu7':
                    current = tf.nn.dropout(current, keep_prob=self.keep_prob)
            elif name.startswith('pool'):
                with tf.name_scope(name):
                    stride = self.weights[i][0][0][4][0]
                    pad = self.weights[i][0][0][5][0]
                    area = self.weights[i][0][0][3][0]
                    current = self._pool_layer(current, stride, pad, area, self.net, name)
            elif name.startswith('fc'):
                with tf.name_scope(name):
                    kernels, bias = self.weights[i][0][0][2][0]

                    bias = bias.reshape(-1)
                    current = self._full_conv(current, kernels, bias, self.my_weights, self.net, name)
            elif name.startswith('norm'):
                with tf.name_scope(name):
                    current = tf.nn.local_response_normalization(current, depth_radius=2, bias=2.000, alpha=0.0001,
                                                                 beta=0.75)
            self.net[name] = current

        with tf.name_scope('image_feature'):
            w_fc8 = tf.truncated_normal([1, 1, self.n_fc7, self.n_feature], stddev=1.0) * 0.01
            b_fc8 = tf.random_normal([self.n_feature], stddev=1.0) * 0.01

            fc8 = self._full_conv(self.net['relu7'], w_fc8, b_fc8, self.my_weights, self.net, 'image_feature')
            fc8_relu = tf.nn.relu(fc8)
            self.net['image_feature'] = tf.squeeze(fc8_relu)

        with tf.name_scope('image_label'):
            w_fc10 = tf.truncated_normal([1, 1, self.n_feature, self.n_class], stddev=1.0) * 0.01
            b_fc10 = tf.random_normal([self.n_class], stddev=1.0) * 0.01

            self.net['label'] = self._full_conv(
                fc8_relu, w_fc10, b_fc10, self.my_weights, self.net, 'image_label')
            self.net['image_label'] = tf.squeeze(self.net['label'])
            self.net['image_label_sigmoid'] = tf.squeeze(tf.sigmoid(self.net['label']))

        with tf.name_scope('image_hash'):
            w_fc9 = tf.truncated_normal([1, 1, self.n_feature, self.bit], stddev=1.0) * 0.01
            b_fc9 = tf.random_normal([self.bit], stddev=1.0) * 0.01
            # TODO 1 change fc8_relu -> fc8
            self.net['hash'] = self._full_conv(fc8, w_fc9, b_fc9, self.my_weights, self.net, 'image_hash')
            self.net['image_hash'] = tf.squeeze(self.net['hash'])
            self.net['image_hash_sigmoid'] = tf.squeeze(tf.sigmoid(self.net['hash']))
            self.net['image_hash_tanh'] = tf.squeeze(tf.tanh(self.net['hash']))

        return self.net, self.mean, self.my_weights

    @staticmethod
    def _conv_layer(input_tensor, weights, bias, pad, stride, my_weight, net, name):
        input_tensor = tf.pad(input_tensor, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
        w = tf.Variable(weights, name=name + '_w', dtype='float32')
        b = tf.Variable(bias, name=name + '_b', dtype='float32')

        my_weight[name + '_w'] = w
        my_weight[name + '_b'] = b

        conv = tf.nn.conv2d(input_tensor, w, strides=[1, stride[0], stride[1], 1], padding='VALID', name=name)
        net[name] = tf.nn.bias_add(conv, b, name=name + '_add')
        return net[name]

    @staticmethod
    def _full_conv(input_tensor, weights, bias, my_weight, net, name):
        w = tf.Variable(weights, name=name + '_w', dtype='float32')
        b = tf.Variable(bias, name=name + '_b', dtype='float32')

        my_weight[name + '_w'] = w
        my_weight[name + '_b'] = b

        conv = tf.nn.conv2d(input_tensor, w, strides=[1, 1, 1, 1], padding='VALID', name=name)
        net[name] = tf.nn.bias_add(conv, b, name=name + '_add')
        return net[name]

    @staticmethod
    def _pool_layer(input_tensor, stride, pad, area, net, name):
        input_tensor = tf.pad(input_tensor, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
        net[name] = tf.nn.max_pool(input_tensor, ksize=[1, area[0], area[1], 1], strides=[1, stride[0], stride[1], 1],
                                   padding='VALID')
        return net[name]


class LabelNetVgg:
    def __init__(self, n_class, keep_prob, config, args):
        self.config = config
        self.args = args
        self.n_class = n_class
        self.keep_prob = keep_prob
        self.bit = self.args['bit']
        self.layer_node_1 = 2048
        self.layer_node_2 = 512
        self.label_net = {}

    def __call__(self, label_net_input):
        with tf.name_scope('label_input'):
            label_net_input = tf.reshape(label_net_input, [-1, 1, self.n_class, 1])

        with tf.name_scope('label_fc1'):
            w_fc1 = tf.truncated_normal([1, self.n_class, 1, self.layer_node_1], stddev=1.0) * 0.01
            b_fc1 = tf.random_normal([1, self.layer_node_1], stddev=1.0) * 0.01
            self.label_net['fc1W'] = tf.Variable(w_fc1)
            self.label_net['fc1b'] = tf.Variable(b_fc1)
            self.label_net['conv1'] = tf.nn.conv2d(label_net_input, self.label_net['fc1W'], strides=[1, 1, 1, 1],
                                                   padding='VALID')
            w1_plus_b1 = tf.nn.bias_add(self.label_net['conv1'], tf.squeeze(self.label_net['fc1b']))
            relu1 = tf.nn.relu(w1_plus_b1)
            norm1 = tf.nn.local_response_normalization(relu1, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
            norm1 = tf.nn.dropout(norm1, self.keep_prob)

        with tf.name_scope('label_fc2'):
            w_fc2 = tf.truncated_normal([1, 1, self.layer_node_1, self.layer_node_2], stddev=1.0) * 0.01
            b_fc2 = tf.random_normal([1, self.layer_node_2], stddev=1.0) * 0.01
            self.label_net['fc2W'] = tf.Variable(w_fc2)
            self.label_net['fc2b'] = tf.Variable(b_fc2)
            self.label_net['conv2'] = tf.nn.conv2d(norm1, self.label_net['fc2W'], strides=[1, 1, 1, 1], padding='VALID')
            fc2 = tf.nn.bias_add(self.label_net['conv2'], tf.squeeze(self.label_net['fc2b']))
            relu2 = tf.nn.relu(fc2)
            self.label_net['label_feature'] = tf.squeeze(relu2)

        with tf.name_scope('label_hash'):
            w_fc3 = tf.truncated_normal([1, 1, self.layer_node_2, self.bit], stddev=1.0) * 0.01
            b_fc3 = tf.random_normal([1, self.bit], stddev=1.0) * 0.01
            self.label_net['fc3W'] = tf.Variable(w_fc3)
            self.label_net['fc3b'] = tf.Variable(b_fc3)
            self.label_net['conv3'] = tf.nn.conv2d(relu2, self.label_net['fc3W'], strides=[1, 1, 1, 1], padding='VALID')
            output_h = tf.nn.bias_add(self.label_net['conv3'], tf.squeeze(self.label_net['fc3b']))

            self.label_net['label_hash'] = tf.squeeze(tf.nn.tanh(output_h))

        with tf.name_scope('label_hash_label'):
            w_fc4 = tf.truncated_normal([1, 1, self.bit, self.n_class], stddev=1.0) * 0.01
            b_fc4 = tf.random_normal([1, self.n_class], stddev=1.0) * 0.01
            self.label_net['fc5W'] = tf.Variable(w_fc4)
            self.label_net['fc5b'] = tf.Variable(b_fc4)
            self.label_net['conv5'] = tf.nn.conv2d(tf.nn.tanh(output_h), self.label_net['fc5W'], strides=[1, 1, 1, 1],
                                                   padding='VALID')
            hash_label_ = tf.nn.bias_add(self.label_net['conv5'], tf.squeeze(self.label_net['fc5b']))
            self.label_net['label_hash_label'] = tf.squeeze(hash_label_)
            self.label_net['label_hash_label_sigmoid'] = tf.squeeze(tf.nn.sigmoid(hash_label_))

        with tf.name_scope('label_label'):
            w_fc4 = tf.truncated_normal([1, 1, self.layer_node_2, self.n_class], stddev=1.0) * 0.01
            b_fc4 = tf.random_normal([1, self.n_class], stddev=1.0) * 0.01
            self.label_net['fc4W'] = tf.Variable(w_fc4)
            self.label_net['fc4b'] = tf.Variable(b_fc4)
            self.label_net['conv4'] = tf.nn.conv2d(relu2, self.label_net['fc4W'], strides=[1, 1, 1, 1], padding='VALID')
            label_ = tf.nn.bias_add(self.label_net['conv4'], tf.squeeze(self.label_net['fc4b']))
            self.label_net['label_label'] = tf.squeeze(tf.nn.sigmoid(label_))
        self.label_net['w2_mean'] = tf.reduce_mean(output_h)
        self.label_net['w2_max'] = tf.reduce_max(output_h)
        self.regular_l2_loss = (tf.nn.l2_loss(self.label_net['fc1W']) +
                                tf.nn.l2_loss(self.label_net['fc2W']) +
                                tf.nn.l2_loss(self.label_net['fc3W']) +
                                tf.nn.l2_loss(self.label_net['fc5W']) +
                                tf.nn.l2_loss(self.label_net['fc4W']))

        return self.label_net, self.regular_l2_loss


class LabelNetVgg_1:
    def __init__(self, n_class, keep_prob, config, args):
        self.config = config
        self.args = args
        self.n_class = n_class
        self.keep_prob = keep_prob
        self.bit = self.args['bit']
        self.batch_size = self.args['batch_size']
        self.layer_node_1 = 2048
        self.layer_node_2 = 2048
        self.layer_node_3 = 512
        self.label_net = {}

    def __call__(self, label_net_input):
        with tf.name_scope('label_input'):
            label_net_input = tf.reshape(label_net_input, [-1, 1, self.n_class, 1])

        with tf.name_scope('label_fc1'):
            w_fc1 = tf.truncated_normal([1, self.n_class, 1, self.layer_node_1], stddev=1.0) * 0.01
            b_fc1 = tf.random_normal([1, self.layer_node_1], stddev=1.0) * 0.01
            self.label_net['fc1W'] = tf.Variable(w_fc1)
            self.label_net['fc1b'] = tf.Variable(b_fc1)
            self.label_net['conv1'] = tf.nn.conv2d(label_net_input, self.label_net['fc1W'], strides=[1, 1, 1, 1],
                                                   padding='VALID')
            w1_plus_b1 = tf.nn.bias_add(self.label_net['conv1'], tf.squeeze(self.label_net['fc1b']))
            relu1 = tf.nn.relu(w1_plus_b1)
            norm1 = tf.nn.local_response_normalization(relu1, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
            norm1 = tf.nn.dropout(norm1, self.keep_prob)

        with tf.name_scope('label_fc2'):
            w_fc2 = tf.truncated_normal([1, 1, self.layer_node_1, self.layer_node_2], stddev=1.0) * 0.01
            b_fc2 = tf.random_normal([1, self.layer_node_2], stddev=1.0) * 0.01
            self.label_net['fc2W'] = tf.Variable(w_fc2)
            self.label_net['fc2b'] = tf.Variable(b_fc2)
            self.label_net['conv2'] = tf.nn.conv2d(norm1, self.label_net['fc2W'], strides=[1, 1, 1, 1],
                                                   padding='VALID')
            w2_plus_b2 = tf.nn.bias_add(self.label_net['conv2'], tf.squeeze(self.label_net['fc2b']))
            relu2 = tf.nn.relu(w2_plus_b2)
            norm2 = tf.nn.local_response_normalization(relu2, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
            norm2 = tf.nn.dropout(norm2, self.keep_prob)

        with tf.name_scope('label_fc3'):
            w_fc3 = tf.truncated_normal([1, 1, self.layer_node_2, self.layer_node_3], stddev=1.0) * 0.01
            b_fc3 = tf.random_normal([1, self.layer_node_3], stddev=1.0) * 0.01
            self.label_net['fc3W'] = tf.Variable(w_fc3)
            self.label_net['fc3b'] = tf.Variable(b_fc3)
            self.label_net['conv3'] = tf.nn.conv2d(norm2, self.label_net['fc3W'], strides=[1, 1, 1, 1], padding='VALID')
            fc3 = tf.nn.bias_add(self.label_net['conv3'], tf.squeeze(self.label_net['fc3b']))
            relu3 = tf.nn.relu(fc3)
            self.label_net['label_feature'] = tf.squeeze(relu3)

        with tf.name_scope('label_hash'):
            w_fc4 = tf.truncated_normal([1, 1, self.layer_node_3, self.bit], stddev=1.0) * 0.01
            b_fc4 = tf.random_normal([1, self.bit], stddev=1.0) * 0.01
            self.label_net['fc4W'] = tf.Variable(w_fc4)
            self.label_net['fc4b'] = tf.Variable(b_fc4)
            self.label_net['conv4'] = tf.nn.conv2d(fc3, self.label_net['fc4W'], strides=[1, 1, 1, 1], padding='VALID')
            output_h = tf.nn.bias_add(self.label_net['conv4'], tf.squeeze(self.label_net['fc4b']))
            self.label_net['label_hash'] = tf.squeeze(tf.nn.tanh(output_h))

        with tf.name_scope('label_hash_label'):
            w_fc5 = tf.truncated_normal([1, 1, self.bit, self.n_class], stddev=1.0) * 0.01
            b_fc5 = tf.random_normal([1, self.n_class], stddev=1.0) * 0.01
            self.label_net['fc5W'] = tf.Variable(w_fc5)
            self.label_net['fc5b'] = tf.Variable(b_fc5)
            self.label_net['conv5'] = tf.nn.conv2d(tf.nn.tanh(output_h), self.label_net['fc5W'], strides=[1, 1, 1, 1],
                                                   padding='VALID')
            hash_label_ = tf.nn.bias_add(self.label_net['conv5'], tf.squeeze(self.label_net['fc5b']))
            self.label_net['label_hash_label'] = tf.squeeze(hash_label_)
            self.label_net['label_hash_label_sigmoid'] = tf.squeeze(tf.nn.sigmoid(hash_label_))

        with tf.name_scope('label_label'):
            w_fc6 = tf.truncated_normal([1, 1, self.layer_node_3, self.n_class], stddev=1.0) * 0.01
            b_fc6 = tf.random_normal([1, self.n_class], stddev=1.0) * 0.01
            self.label_net['fc6W'] = tf.Variable(w_fc6)
            self.label_net['fc6b'] = tf.Variable(b_fc6)
            self.label_net['conv6'] = tf.nn.conv2d(relu3, self.label_net['fc6W'], strides=[1, 1, 1, 1], padding='VALID')
            label_ = tf.nn.bias_add(self.label_net['conv6'], tf.squeeze(self.label_net['fc6b']))
            self.label_net['label_label'] = tf.squeeze(tf.nn.sigmoid(label_))
        self.label_net['w2_mean'] = tf.reduce_mean(output_h)
        self.label_net['w2_max'] = tf.reduce_max(output_h)
        self.regular_l2_loss = (tf.nn.l2_loss(self.label_net['fc1W']) +
                                tf.nn.l2_loss(self.label_net['fc2W']) +
                                tf.nn.l2_loss(self.label_net['fc3W']) +
                                tf.nn.l2_loss(self.label_net['fc5W']) +
                                tf.nn.l2_loss(self.label_net['fc4W']))

        return self.label_net, self.regular_l2_loss

