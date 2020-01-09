import pickle
import random
import json
import numpy as np
import tensorflow as tf
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class NameMapping:
    # Network name mappings
    @staticmethod
    def default_mapping_(name):
        # print(name, name[:-2])
        return name[:-2]

    @staticmethod
    def imagenet_mapping_(name):
        variable_dict = {
            "Conv1/composite_function/kernel:0": "block1_conv1/kernel",
            "Conv1/composite_function/bias:0": "block1_conv1/bias",
            "Conv2/composite_function/kernel:0": "block1_conv2/kernel",
            "Conv2/composite_function/bias:0": "block1_conv2/bias",
            "Conv3/composite_function/kernel:0": "block2_conv1/kernel",
            "Conv3/composite_function/bias:0": "block2_conv1/bias",
            "Conv4/composite_function/kernel:0": "block2_conv2/kernel",
            "Conv4/composite_function/bias:0": "block2_conv2/bias",
            "Conv5/composite_function/kernel:0": "block3_conv1/kernel",
            "Conv5/composite_function/bias:0": "block3_conv1/bias",
            "Conv6/composite_function/kernel:0": "block3_conv2/kernel",
            "Conv6/composite_function/bias:0": "block3_conv2/bias",
            "Conv7/composite_function/kernel:0": "block3_conv3/kernel",
            "Conv7/composite_function/bias:0": "block3_conv3/bias",
            "Conv8/composite_function/kernel:0": "block4_conv1/kernel",
            "Conv8/composite_function/bias:0": "block4_conv1/bias",
            "Conv9/composite_function/kernel:0": "block4_conv2/kernel",
            "Conv9/composite_function/bias:0": "block4_conv2/bias",
            "Conv10/composite_function/kernel:0": "block4_conv3/kernel",
            "Conv10/composite_function/bias:0": "block4_conv3/bias",
            "Conv11/composite_function/kernel:0": "block5_conv1/kernel",
            "Conv11/composite_function/bias:0": "block5_conv1/bias",
            "Conv12/composite_function/kernel:0": "block5_conv2/kernel",
            "Conv12/composite_function/bias:0": "block5_conv2/bias",
            "Conv13/composite_function/kernel:0": "block5_conv3/kernel",
            "Conv13/composite_function/bias:0": "block5_conv3/bias",
            "FC14/W:0": "fc1/kernel",
            "FC14/bias:0": "fc1/bias",
            "FC15/W:0": "fc2/kernel",
            "FC15/bias:0": "fc2/bias",
            "FC16/W:0": "predictions/kernel",
            "FC16/bias:0": "predictions/bias"
        }
        return variable_dict[name]


# ================== VGG Network Model ==================
class Model():
    def __init__(self, learning_rate=1e-3):
        '''
        For encode images super parameters:
            1. learning rate
            2. L1 penalty
            3. Lambda control gate threshold
        '''
        self.learning_rate = learning_rate
        self.use_batch_norm = False
        self.with_bias = True
        self.vgg_type = 'D'
        self.dataset_name = 'cifar'
        self.verbose = True

        self.batch = 0

    '''
    Wrap Functions:
        1. model.build_model()
        2. model.restore_model()
    '''

    '''
    Encode and save for a batch of data
    vgg_type: C, D
    Structures of different types please refere to
        https://img2018.cnblogs.com/blog/1365470/201903/1365470-20190307200142975-686030661.png
    '''
    def restore_model(self, graph, checkpoint_path, name_mapping):
        savedVariable = {}

        if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 0:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=graph, config=config)
        else:
            self.sess = tf.Session(graph=graph)

        self.writer.add_graph(self.sess.graph)
        self.sess.run(self.init)

        with graph.as_default():
            for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                variable = i
                name = i.name
                if name == 'pl:0'\
                        or 'Resize' in name:
                    continue
                if name[-11:] == '/Momentum:0':
                    continue
                elif name[:4] == 'FC16':
                    continue
                savedVariable[name_mapping(name)] = variable
            for k, v in savedVariable.items():
                print(k, v)
            saver = tf.train.Saver(savedVariable)
            # saver = tf.train.Saver(max_to_keep = None)
            saver.restore(self.sess, checkpoint_path)
            # print("Restored successfully!")

            # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            print(tf.trainable_variables())
            self.saver = tf.train.Saver()

    '''
    Build VGG Network with Control Gate Lambdas
    '''

    def build_model(self, graph, label_count):
        with graph.as_default():
            '''
            Place Holders:
                1. input_x: data
                2. input_y: original predicted labels
                3. learning rate
                4. drop keeping probability: no drop layer actually
                5. whether in training mode: 
            '''
            self.xs = tf.placeholder("float", shape=[None, 32, 32, 3], name='xs')
            self.ys_orig = tf.placeholder("float", shape=[None, label_count], name='ys_orig')
            self.ys_true_argmax = tf.argmax(self.ys_orig, 1)
            self.lr = tf.placeholder("float", shape=[], name='lr')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.is_training = tf.placeholder("bool", shape=[], name='is_training')
            kernel_size = {'C': 1, 'D': 3}[self.vgg_type]

            '''
            VGG Network Model Construction with Control Gates 
            '''
            current = self.xs
            with tf.variable_scope("Resize", reuse=tf.AUTO_REUSE):
                current = tf.image.resize_images(current, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
            with tf.variable_scope("Conv1", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 3, 64, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv2", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 64, 64, 3, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv3", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 64, 128, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv4", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 128, 128, 3, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv5", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 128, 256, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv6", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 256, 256, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv7", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 256, 256, kernel_size, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv8", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 256, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv9", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv10", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, kernel_size, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv11", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv12", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv13", reuse=tf.AUTO_REUSE):
                print(kernel_size)
                current = self.batch_activ_conv(current, 512, 512, kernel_size, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
                if self.vgg_type == 'C':
                    current = tf.reshape(current, [-1, 512])
                elif self.vgg_type == 'D':
                    current = tf.reshape(current, [-1, 25088])
            with tf.variable_scope("FC14", reuse=tf.AUTO_REUSE):
                if self.vgg_type == 'C':
                    current = self.batch_activ_fc(current, 512, 4096, self.is_training)
                elif self.vgg_type == 'D':
                    current = self.batch_activ_fc(current, 25088, 4096, self.is_training)
            with tf.variable_scope("FC15", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_fc(current, 4096, 4096, self.is_training)
            with tf.variable_scope("FC16", reuse=tf.AUTO_REUSE):
                Wfc = self.weight_variable_xavier([4096, label_count], name='W')
                bfc = self.bias_variable([label_count])
                self.ys_pred = tf.matmul(current, Wfc) + bfc

            self.ys_pred_softmax = tf.nn.softmax(self.ys_pred)
            self.ys_pred_argmax = tf.argmax(self.ys_pred, 1)

            '''
            Loss Definition
            '''
            # prediction = tf.nn.softmax(ys_)
            # conv_value = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in convValues])
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.ys_pred, labels=self.ys_orig
            ))
            self.total_loss = self.cross_entropy

            '''
            Optimizer
            '''
            self.train_step = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True).minimize(self.total_loss)

            '''
            Check whether correct
            '''
            correct_prediction = tf.equal(self.ys_true_argmax, self.ys_pred_argmax)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            self.init = tf.global_variables_initializer()

            '''
            SummWriter
            '''
            self.writer = tf.summary.FileWriter('log')
            self.loss_summ = tf.summary.scalar(name='loss', tensor=self.total_loss)
            self.acc_summ = tf.summary.scalar(name='acc', tensor=self.accuracy)

    def train_model(self, input_images, input_labels):
        # if self.epoch == 5: self.learning_rate /= 10
        if self.batch == 400: self.learning_rate /= 10
        if self.batch == 800: self.learning_rate /= 10
        _, loss, acc, ys_pred_argmax, loss_summ = self.sess.run(
            [self.train_step, self.total_loss, self.accuracy, self.ys_pred_argmax, self.loss_summ], 
            feed_dict = {
                self.xs: input_images,
                self.ys_orig : input_labels, 
                self.lr : self.learning_rate, 
                self.keep_prob : 1.0, 
                self.is_training : True
            })
        self.batch += 1
        # print("train:", ys_pred_argmax)
        print("Loss: ",np.array(loss))
        self.writer.add_summary(loss_summ, self.batch)

    def test_accuracy(self, test_images, test_labels):
        accuracy, ys_pred_argmax, ys_true_argmax, ys_pred, acc_summ = self.sess.run(
            [self.accuracy, self.ys_pred_argmax, self.ys_true_argmax, self.ys_pred, self.acc_summ], 
            feed_dict={
            self.xs: test_images,
            self.ys_orig: test_labels, 
            self.lr : 0.1,
            self.is_training: False,
            self.keep_prob: 1.0,
        }) 
        self.writer.add_summary(acc_summ, self.batch)
        return accuracy

    def save_ckpt(self, name=None):
        print('saving model')
        if name is None:
            self.saver.save(self.sess, 'ckpt/model.ckpt', self.batch)
        else:
            self.saver.save(self.sess, 'ckpt/{}.ckpt'.format(name))


    '''
    Close Session
    '''

    def close_sess(self):
        self.sess.close()

    '''
    Helper Builder Functions: to build model more conveniently
    '''

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer(),
                               trainable=True)

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                               trainable=True)

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name=name, initializer=initial, trainable=True)

    def conv2d(self, input, in_features, out_features, kernel_size):
        W = self.weight_variable_msra([kernel_size, kernel_size, in_features, out_features], name='kernel')
        conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
        if self.with_bias:
            return conv + self.bias_variable([out_features])
        return conv

    def batch_activ_conv(self, current, in_features, out_features, kernel_size, is_training, keep_prob):
        with tf.variable_scope("composite_function", reuse=tf.AUTO_REUSE):
            current = self.conv2d(current, in_features, out_features, kernel_size)
            if self.use_batch_norm:
                current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training,
                                                       updates_collections=None, trainable=True)
            current = tf.nn.relu(current)
        return current

    def batch_activ_fc(self, current, in_features, out_features, is_training):
        Wfc = self.weight_variable_xavier([in_features, out_features], name='W')
        bfc = self.bias_variable([out_features])
        current = tf.matmul(current, Wfc) + bfc
        if self.use_batch_norm:
            current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=True)
        current = tf.nn.relu(current)
        return current

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='VALID')

    '''
    Helper Data Processing Functions
    '''

    # calculate the means and stds for the whole dataset per channel
    def measure_mean_and_std(self, images):
        means = []
        stds = []
        for ch in range(images.shape[-1]):
            means.append(np.mean(images[:, :, :, ch]))
            stds.append(np.std(images[:, :, :, ch]))
        return means, stds

    # normalization for per channel
    def normalize_images(self, images):
        images = images.astype('float64')
        means, stds = self.measure_mean_and_std(images)
        for i in range(images.shape[-1]):
            images[:, :, :, i] = ((images[:, :, :, i] - means[i]) / stds[i])
        return images
