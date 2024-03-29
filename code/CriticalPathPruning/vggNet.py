import pickle
import random
# from decimal import *
import json
import keras
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class NameMapping:
    # Network name mappings
    @staticmethod
    def default_mapping_(name):
        # print(name, name[:-2])
        return name[:-2]

    @staticmethod
    def identity_mapping_(name):
        return name

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


# ================== VGG Network Model with Control Gates ==================
class Model():
    def __init__(self, learning_rate=0.1, L1_loss_penalty=0.02, threshold=0.1, entropy_penalty=1):
        '''
        For one input image :
            1. Store all the gates infomation
            2. Store all the gates values
        '''
        self.AllGateVariables = dict()
        self.AllGateVariableValues = list()

        '''
        For encode images super parameters:
            1. learning rate
            2. L1 penalty
            3. Lambda control gate threshold
        '''
        self.learning_rate = learning_rate
        self.L1_loss_penalty = L1_loss_penalty
        self.threshold = threshold
        self.entropy_penalty = entropy_penalty

    '''
    Wrap Functions:
        1. model.build_model()
        2. model.restore_model()
        3. model.encode_input()
    '''

    def compute_encoding(self, data_input, checkpoint_path="vggNet/augmentation.ckpt-120",
                         name_mapping=None, output_size=100):
        if name_mapping is None:
            name_mapping = NameMapping.default_mapping_

        self.AllGateVariableValues.clear()
        self.AllGateVariables.clear()

        print("Restoring network...")
        graph = tf.Graph()
        self.build_model(graph, output_size)
        self.restore_model(graph, checkpoint_path, name_mapping)
        print("Computing CDRP...")
        generatedGates = self.encode_input(data_input)
        self.close_sess()

        return generatedGates

    '''
    Encode and save for a batch of data
    vgg_type: C, D
    Structures of different types please refere to
        https://img2018.cnblogs.com/blog/1365470/201903/1365470-20190307200142975-686030661.png
    '''

    def encode_class_data(self, class_id, train_images, checkpoint_path="vggNet/augmentation.ckpt-120",
                          name_mapping=None, use_batch_norm=True, with_bias=False, vgg_type='C', output_size=100,
                          dataset_name='cifar', verbose=True,save_folder = None):
        self.use_batch_norm = use_batch_norm
        self.with_bias = with_bias
        self.vgg_type = vgg_type
        self.dataset_name = dataset_name
        self.verbose = verbose

        # filename = "./innerclass/class"+str(class_id)+"gate.json"
        for i in range(len(train_images)):
            print('{}/{}'.format(i, len(train_images)))
            # generatedGate is a list of dicts{layername:xx, shape:xx, lambda:xx}
            generatedGate = self.compute_encoding(train_images[i].reshape((1, 32, 32, 3)),
                                                  checkpoint_path, name_mapping, output_size)
            picname = "class" + str(class_id) + "-pic" + str(i)
            if save_folder == None:
                if dataset_name == 'imagenet':
                    save_folder = "imagenet-ImageEncoding"
                else:
                    save_folder = "ImageEncoding"
            if dataset_name == 'imagenet':
                jsonpath = "./"+save_folder+"/" + picname + ".json"
            else:
                jsonpath = "./"+save_folder+"/" + picname + ".json"
            with open(jsonpath, 'w') as f:
                json.dump(generatedGate, f, sort_keys=True, indent=4, separators=(',', ':'))

    '''
    Restore the original network weights
    '''

    def restore_model(self, graph, checkpoint_path, name_mapping):
        savedVariable = {}

        if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 0:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=graph, config=config)
        else:
            self.sess = tf.Session(graph=graph)

        self.sess.run(self.init)

        with graph.as_default():
            for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                variable = i
                name = i.name
                if name == 'pl:0'\
                        or name in self.AllGateVariables\
                        or 'Resize' in name:
                    continue
                if len(name) >= 8 and name[-11:] == '/Momentum:0':
                    name_prefix = name[:-11]
                    name_prefix += ':0'
                    if name_prefix in self.AllGateVariables:
                        continue
                print(name)
                savedVariable[name_mapping(name)] = variable
            saver = tf.train.Saver(savedVariable)
            # saver = tf.train.Saver(max_to_keep = None)
            saver.restore(self.sess, checkpoint_path)
            # print("Restored successfully!")

        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

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
                5. whether in training mode: always False
                6. penalty: regularization
            '''
            # if self.vgg_type == 'C':
            self.xs = tf.placeholder("float", shape=[None, 32, 32, 3])
            # elif self.vgg_type == 'D':
            #     self.xs = tf.placeholder("float", shape=[None, 224, 224, 3])
            self.ys_orig = tf.placeholder("float", shape=[None, label_count])
            self.lr = tf.placeholder("float", shape=[])
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder("bool", shape=[])
            self.penalty = tf.placeholder(tf.float32)
            kernel_size = {'C': 1, 'D': 3}[self.vgg_type]

            '''
            VGG Network Model Construction with Control Gates 
            '''
            current = self.xs
            if self.dataset_name == 'imagenet':
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
            '''
            Loss Definition
            '''
            # prediction = tf.nn.softmax(ys_)
            # conv_value = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in convValues])
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.ys_pred, labels=self.ys_orig
            ))
            l1_loss = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in self.AllGateVariableValues])
            self.l1_loss = l1_loss * self.penalty
            self.total_loss = self.l1_loss + self.cross_entropy * self.entropy_penalty

            '''
            Optimizer
            '''
            self.train_step = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True).minimize(self.total_loss)

            '''
            Check whether correct
            '''
            correct_prediction = tf.equal(tf.argmax(self.ys_orig, 1), tf.argmax(self.ys_pred, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            self.init = tf.global_variables_initializer()

    '''
    Given a image input
    Produce a lambda code
    '''

    def encode_input(self, input_data):
        generateGate = dict()

        learning_rate = self.learning_rate
        L1_loss_penalty = self.L1_loss_penalty
        threshold = self.threshold

        label_orig = self.sess.run(self.ys_pred_softmax, feed_dict={
            self.xs: input_data,
            self.lr: learning_rate,
            self.keep_prob: 1.0,
            self.is_training: False,
            self.penalty: L1_loss_penalty
        })

        tmpLoss = 1000
        for epoch in range(150):
            if epoch == 50:
                learning_rate /= 10
                # L1_loss_penalty *= 10

            self.sess.run(self.train_step, feed_dict={
                self.xs: input_data,
                self.ys_orig: label_orig,
                self.lr: learning_rate,
                self.keep_prob: 1.0,
                self.is_training: False,
                self.penalty: L1_loss_penalty
            })

            [cross_entropy, L1_loss, accuracy, ys_orig] = self.sess.run([self.cross_entropy, self.l1_loss, self.accuracy,self.ys_orig],
                                                               feed_dict={
                                                                   self.xs: input_data,
                                                                   self.ys_orig: label_orig,
                                                                   self.lr: learning_rate,
                                                                   self.keep_prob: 1.0,
                                                                   self.is_training: False,
                                                                   self.penalty: L1_loss_penalty
                                                               })
            print(np.argmax(ys_orig),ys_orig.shape)

            if self.verbose:
                print("Epoch: {}: Cross_Entropy: {}, L1_loss: {}, Accuracy: {}".format(
                epoch, cross_entropy, L1_loss, accuracy))

            # print(self.AllGateVariables.keys())
            '''
                dict_keys(
                    [
                        'Conv2/composite_function/gate:0', 
                        'FC14/gate:0', 
                        'Conv5/composite_function/gate:0', 
                        'Conv13/composite_function/gate:0', 
                        'FC15/gate:0', 
                        'Conv9/composite_function/gate:0', 
                        'Conv3/composite_function/gate:0', 
                        'Conv7/composite_function/gate:0', 
                        'Conv10/composite_function/gate:0', 
                        'Conv6/composite_function/gate:0', 
                        'Conv4/composite_function/gate:0', 
                        'Conv12/composite_function/gate:0', 
                        'Conv1/composite_function/gate:0', 
                        'Conv11/composite_function/gate:0', 
                        'Conv8/composite_function/gate:0'
                    ]
                )

            '''
            newGate = []
            # Li Dongyue's Version

            # for gate in self.AllGateVariables.values():
            #     tmp = gate.eval(session=self.sess)
            #     tmp[tmp < threshold] = 0
            #     newGate.append(tmp)
            # if L1_loss == 'nan' or L1_loss > tmpLoss:
            #     continue
            # if accuracy > 0.99 and L1_loss != 'nan' and L1_loss < 1000:
            #     generateGate = np.array(newGate)
            #     tmpLoss = L1_loss

            # Cao Mengqi's version: Json
            for gate in self.AllGateVariables.keys():
                tmp = self.AllGateVariables[gate].eval(session=self.sess)
                tmp[tmp < threshold] = 0
                tmp = tmp.tolist()
                res = dict()
                res["layer_name"] = gate
                res["layer_lambda"] = tmp
                res["shape"] = len(tmp)
                newGate.append(res)
            if L1_loss == 'nan' or L1_loss > tmpLoss:
                continue

            ## Original version: only accept those that are still correct
            # if accuracy > 0.99 and L1_loss != 'nan' and L1_loss < 1000:
            ## In another network without correct labels
            if L1_loss != 'nan' and L1_loss < 1000:
                if self.verbose:
                    print("Assign gate")
                generateGate = newGate
                tmpLoss = L1_loss

        # now generatedGate is a list [(layer name, lambda), ...]
        return generateGate

    '''
    Close Session
    '''

    def close_sess(self):
        self.sess.close()

    '''
    Function that print out original VGG network weight
    '''
    # def print_weights_to_Json(self):
    #     import json
    #     from tensorflow.python import pywrap_tensorflow
    #     model_dir="vggNet/augmentation.ckpt-120" #checkpoint的文件位置
    #     # Read data from checkpoint file
    #     reader = pywrap_tensorflow.NewCheckpointReader(model_dir)
    #     var_to_shape_map = sorted(reader.get_variable_to_shape_map())
    #     # Print tensor name and values
    #     layer = dict()
    #     result = dict()
    #     for key in var_to_shape_map:
    #         layer[key] = dict()
    #         layer[key]["name"] = key
    #         layer[key]["shape"] = reader.get_tensor(key).shape
    #         result[key] = {"name": key, "shape": reader.get_tensor(key).shape, "vec": reader.get_tensor(key).tolist()}
    #     # with open("weights.json","w") as f:
    #     #     json.dump(result,f, sort_keys=True, indent=4, separators=(',', ':'))
    #     with open("layers.json","w") as f:
    #         json.dump(layer,f, sort_keys=True, indent=4, separators=(',', ':'))

    '''
    Helper Builder Functions: to build model more conveniently
    '''

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer(),
                               trainable=False)

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                               trainable=False)

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name=name, initializer=initial, trainable=False)

    def gate_variable(self, length, name='gate'):
        initial = tf.constant([1.0] * length)
        v = tf.get_variable(name=name, initializer=initial)
        self.AllGateVariables[v.name] = v
        v = tf.abs(v)
        v = v - tf.constant([0.01] * length)
        v = tf.nn.relu(v)
        self.AllGateVariableValues.append(v)
        return v

    def conv2d(self, input, in_features, out_features, kernel_size):
        W = self.weight_variable_msra([kernel_size, kernel_size, in_features, out_features], name='kernel')
        conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
        gate = self.gate_variable(out_features)
        conv = tf.multiply(conv, tf.abs(gate))
        if self.with_bias:
            return conv + self.bias_variable([out_features])
        return conv

    def batch_activ_conv(self, current, in_features, out_features, kernel_size, is_training, keep_prob):
        with tf.variable_scope("composite_function", reuse=tf.AUTO_REUSE):
            current = self.conv2d(current, in_features, out_features, kernel_size)
            if self.use_batch_norm:
                current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training,
                                                       updates_collections=None, trainable=False)
            # convValues.append(current)
            current = tf.nn.relu(current)
            # current = tf.nn.dropout(current, keep_prob)
        return current

    def batch_activ_fc(self, current, in_features, out_features, is_training):
        Wfc = self.weight_variable_xavier([in_features, out_features], name='W')
        bfc = self.bias_variable([out_features])
        current = tf.matmul(current, Wfc) + bfc
        gate = self.gate_variable(out_features)
        current = tf.multiply(current, tf.abs(gate))
        if self.use_batch_norm:
            current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None,
                                               trainable=False)
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
