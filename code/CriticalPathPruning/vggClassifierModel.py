import pickle
import random
# from decimal import *
import json
import keras
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class ClassifierModel():
    '''
    Fine Tune Model: (Basically doing transfer learning)
        1. Last FC layer neurons number would be the target class number + 1
        2. Need to transform the original label to targeted class label
    '''

    def __init__(self,vgg_type="C"):
        '''
        Set hyperparameters
        '''
        self.learning_rate = 0.1
        self.epoch = 0
        self.prune_ratio = 0.9
        self.vgg_type = vgg_type

        '''
        For one input image :
            1. Store all the gates infomation
            2. Store all the gates values 
        '''
        self.AllGateVariables = dict()
        self.AllGateVariableValues = list()


        # self.target_id = target_class_id
        self.target_number = 2

        # self.target_class_id = target_class_id  # assign the trim class id

        self.graph = tf.Graph()
        self.build_model(self.graph)
        # print("restored the pretrained model......")
        # self.restore_model(self.graph)

    def from_checkpoint(self, filename):
        # If GPU is needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        # Else if CPU needed
        # self.sess = tf.Session(graph = graph)
        self.sess.run(self.init)
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, filename)
            print("Restored successfully!")

    '''
    Test Accuracy
    '''

    def predict(self, test_images, test_labels):

        ys_pred_argmax = self.sess.run(
            self.ys_pred_argmax, feed_dict={
                self.xs: test_images,
                # self.ys_orig: test_labels,
                self.lr: 0.1,
                self.is_training: False,
                self.keep_prob: 1.0,
            })

        return ys_pred_argmax

    '''
    Fine tune training
    '''

    def save_model(self, classifier_id, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = "classifier%d.ckpt" % classifier_id
        filename = os.path.join(output_dir, filename)
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    '''
    Build VGG Network with Control Gate Lambdas
    '''

    def build_model(self, graph, label_count=100):
        with graph.as_default():
            '''
            Place Holders:
                1. input_x: data
                2. input_y: original predicted labels
                3. learning rate
                4. drop keeping probability: no drop layer actually
                5. whether in training mode: always False
                6. penalty: regularization
            '''
            self.xs = tf.placeholder("float", shape=[None, 32, 32, 3])
            # self.ys_orig = tf.placeholder("float", shape=[None, self.target_number])
            self.lr = tf.placeholder("float", shape=[])
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder("bool", shape=[])
            # weight_decay = 5e-4
            weight_decay = 0
            kernel_size = {'C': 1, 'D': 3}[self.vgg_type]

            '''
            VGG Network Model Construction with Control Gates 
            '''
            current = self.xs
            if self.vgg_type == "D":
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
            with tf.variable_scope("FC14", reuse = tf.AUTO_REUSE):
                if self.vgg_type=="C":
                    current = self.batch_activ_fc(current, 512, 4096, self.is_training)
                else:
                    current = self.batch_activ_fc(current, 25088, 4096, self.is_training)
            with tf.variable_scope("FC15", reuse=tf.AUTO_REUSE):
                current = self.batch_activ_fc(current, 4096, 4096, self.is_training)
            with tf.variable_scope("FC16", reuse=tf.AUTO_REUSE):
                Wfc = self.weight_variable_xavier([4096, self.target_number], name='W')
                bfc = self.bias_variable([self.target_number])
                self.ys_pred = tf.matmul(current, Wfc) + bfc

            self.ys_pred_argmax = tf.argmax(self.ys_pred, 1)
            self.init = tf.global_variables_initializer()

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

    def gate_variable(self, length, name='gate'):
        initial = tf.constant([1.0] * length)
        v = tf.get_variable(name=name, initializer=initial, trainable=False)
        self.AllGateVariables[v.name] = v
        self.AllGateVariableValues.append(v)
        return v

    def conv2d(self, input, in_features, out_features, kernel_size, with_bias=False):
        W = self.weight_variable_msra([kernel_size, kernel_size, in_features, out_features], name='kernel')
        conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
        gate = self.gate_variable(out_features)
        conv = tf.multiply(conv, tf.abs(gate))
        if with_bias:
            return conv + self.bias_variable([out_features])
        return conv

    def batch_activ_conv(self, current, in_features, out_features, kernel_size, is_training, keep_prob):
        with tf.variable_scope("composite_function", reuse=tf.AUTO_REUSE):
            current = self.conv2d(current, in_features, out_features, kernel_size)
            if self.vgg_type=="C":
                current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training,
                                                   updates_collections=None, trainable=True)
            # convValues.append(current)
            current = tf.nn.relu(current)
            # current = tf.nn.dropout(current, keep_prob)
        return current

    def batch_activ_fc(self, current, in_features, out_features, is_training):
        Wfc = self.weight_variable_xavier([in_features, out_features], name='W')
        bfc = self.bias_variable([out_features])
        current = tf.matmul(current, Wfc) + bfc
        # gate = self.gate_variable(out_features)
        # current = tf.multiply(current, tf.abs(gate))
        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None,
                                               trainable=True)
        current = tf.nn.relu(current)
        return current

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='VALID')