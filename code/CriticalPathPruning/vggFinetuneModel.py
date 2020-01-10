import pickle
import random
# from decimal import *
import json
import keras
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
class FineTuneModel():
    '''
    Fine Tune Model: (Basically doing transfer learning)
        1. Last FC layer neurons number would be the target class number + 1
        2. Need to transform the original label to targeted class label
    '''
    def __init__(self, target_class_id=[0], target_cluster_id = [0], mode = "class",cluster_dir = None):
        '''
        Set hyperparameters
        '''
        self.learning_rate = 0.1
        self.epoch = 0
        self.prune_ratio = 0.95

        '''
        For one input image :
            1. Store all the gates infomation
            2. Store all the gates values 
        '''
        self.AllGateVariables = dict()
        self.AllGateVariableValues = list()

        self.mode = mode
        if mode == "class":
            self.target_id = target_class_id
            self.target_number = len(target_class_id) + 1
            self.cluster_dir = None
        else:
            self.target_id = target_cluster_id
            self.target_number = len(target_cluster_id) + 1
            self.cluster_dir = os.path.join(cluster_dir,"cluster%d.json")

        self.target_class_id = target_class_id # assign the trim class id


        self.graph = tf.Graph()
        self.build_model(self.graph)
        print("restored the pretrained model......")
        self.restore_model(self.graph)

    def from_checkpoint(self,filename):
        # If GPU is needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph = graph, config = config)
        # Else if CPU needed
        # self.sess = tf.Session(graph = graph)
        self.sess.run(self.init)
        with graph.as_default():
            saver = tf.train.Saver(savedVariable)
            saver.restore(self.sess, filename)
            print("Restored successfully!")


    '''
    Test Accuracy
    '''
    def test_accuracy(self, test_images, test_labels):

        accuracy,ys_pred_argmax = self.sess.run(
            [self.accuracy,self.ys_pred_argmax], feed_dict={
            self.xs: test_images,
            self.ys_orig: test_labels, 
            self.lr : 0.1,
            self.is_training: False,
            self.keep_prob: 1.0,
        }) 

        new_test_labels = []
        for item in test_labels:
            if item[0]>item[1]:
                new_test_labels.append(0)
            else:
                new_test_labels.append(1)
        new_test_labels = np.array(new_test_labels)

        p = ys_pred_argmax==0
        n = ys_pred_argmax==1

        p_label = new_test_labels[p]
        p_pred = ys_pred_argmax[p]
        n_label = new_test_labels[n]
        n_pred = ys_pred_argmax[n]

        tp = sum(p_label == p_pred)
        fp = sum(p) - tp
        tn = sum(n_label == n_pred)
        fn = sum(n) - tn

        print("Test Accuracy:" + str(accuracy))
        print("TP: %f, FP: %f, TN: %f, FN: %f"%(tp,fp,tn,fn))

        return (accuracy,tp,fp,tn,fn)
        # print("test:",ys_pred_argmax)
    '''
    Fine tune training
    '''
    def train_model(self, input_images, input_labels):
        # if self.epoch == 5: self.learning_rate /= 10
        if self.epoch == 1200: self.learning_rate /= 5
        if self.epoch == 1500: self.learning_rate /= 5
        _, loss,ys_pred_argmax  = self.sess.run([self.train_step, self.loss, self.ys_pred_argmax], feed_dict = {
                self.xs: input_images,
                self.ys_orig : input_labels, 
                self.lr : self.learning_rate, 
                self.keep_prob : 1.0, 
                self.is_training : True
            })
        self.epoch += 1
        # print("train:", ys_pred_argmax)
        print("Loss: ",np.array(loss))


    def save_model(self,classifier_id,output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = "classifier%d.ckpt"%classifier_id
        filename = os.path.join(output_dir,filename)
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess,filename)
        print('Wrote snapshot to: {:s}'.format(filename))


    '''
    Restore the original network weights
    '''
    def restore_model(self, graph):
        savedVariable = {}

        # If GPU is needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph = graph, config = config)
        # Else if CPU needed
        # self.sess = tf.Session(graph = graph)
        self.sess.run(self.init)

        with graph.as_default():
            for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                variable = i
                name = i.name
                if name == 'pl:0':
                    continue
                if name[:4] == "FC16":
                    continue
                if name in self.AllGateVariables:
                    continue 
                if len(name) >= 8 and name[-11:] == '/Momentum:0':
                    name_prefix = name[:-11]
                    name_prefix += ':0'
                    if name_prefix in self.AllGateVariables:
                        continue
                if "BatchNorm" in name and "data-d" in self.cluster_dir:
                    continue
                name = i.name[:-2]
                savedVariable[name] = variable
            saver = tf.train.Saver(savedVariable)
            # saver = tf.train.Saver(max_to_keep = None)
            if "data-d" in self.cluster_dir:
                saver.restore(self.sess, "vgg16-d/best.ckpt")
            else:
                saver.restore(self.sess, "vggNet/augmentation.ckpt-120")
            print("Restored successfully!")

    '''
    Find mask class unit
    '''
    def mask_class_unit(self, classid):
        self.test_counter = 0
        theshold = 10
        if self.mode == "class":
            json_path = "./ClassEncoding/class" + str(classid) + ".json"
        else:
            json_path = self.cluster_dir%classid
        with open(json_path, "r") as f:
            gatesValueDict = json.load(f)
            for idx in range(len(gatesValueDict)):
                layer = gatesValueDict[idx]
                name = layer["name"]
                vec = layer["shape"]
                # process name
                name = name.split('/')[0]
                # process vec
                for i in range(len(vec)):
                    if vec[i] < 10:
                        vec[i] = 0
                    else:
                        vec[i] = 1
                layer["name"] = name
                layer["shape"] = vec

            return gatesValueDict
    '''
    mask by value
    '''
    def mask_unit_by_value(self, classid):
        formulizedDict = {}
        if self.mode == "class":
            json_path = "./ClassEncoding/class" + str(classid) + ".json"
        else:
            json_path = self.cluster_dir%classid

        
        allGatesValue = []

        with open(json_path, "r") as f:
            gatesValueDict = json.load(f)
            for idx in range(len(gatesValueDict)):
                layer = gatesValueDict[idx]
                name = layer["name"]
                vec = layer["shape"]
                allGatesValue += vec
                
        allGatesValue.sort()
        allGatesValue = allGatesValue[:int(len(allGatesValue)*self.prune_ratio)]
        
        allGatesValue = set(allGatesValue)
        with open(json_path, "r") as f:
            gatesValueDict = json.load(f)
            for idx in range(len(gatesValueDict)):
                layer = gatesValueDict[idx]
                name = layer["name"]
                vec = layer["shape"]        
                # process name
                name = name.split('/')[0]
                # process vec
                for i in range(len(vec)):
                    if vec[i] in allGatesValue or vec[i]==0:
                        vec[i] = 0
                    else:
                        vec[i] = 1
                layer["name"] = name
                layer["shape"] = vec
            with open("./ClassEncoding/mask1.json", "w") as tmpFile:
                json.dump(gatesValueDict, tmpFile)
            return gatesValueDict

    '''
    Fine mask class multi, merge multi-class JSONs
    '''
    def mask_class_multi(self):
        theshold = 5
        self.test_counter = 0
        ''' init the dict with class0.json '''
        multiClassGates = self.mask_class_unit(self.target_id[0])
        for classid in self.target_id:
            if (classid == self.target_id[0]):
                continue
            ''' Merge JSONs continuously '''

            if self.mode == "class":
                json_path = "./ClassEncoding/class" + str(classid) + ".json"
            else:
                json_path = self.cluster_dir % classid

            with open(json_path, "r") as f:
                gatesValueDict = json.load(f)
                for idx in range(len(gatesValueDict)):
                    layer = gatesValueDict[idx]
                    name = layer["name"]
                    vec = layer["shape"]
                    # process name
                    name = name.split('/')[0]
                    # process vec
                    for i in range(len(vec)):
                        if vec[i] < theshold:
                            vec[i] = 0
                        else:
                            vec[i] = 1
                    gatesValueDict[idx]["name"] = name
                    gatesValueDict[idx]["shape"] = vec
                
                ''' Now we merge gatesValueDict and multiClassGates '''
                for idx1 in range(len(gatesValueDict)):
                    for idx2 in  range(len(multiClassGates)):
                        if (gatesValueDict[idx1]["name"] == multiClassGates[idx2]["name"]):
                            tomerge = gatesValueDict[idx1]["shape"]
                            for idx3 in range(len(tomerge)):
                                if (tomerge[idx3] == 1 and multiClassGates[idx2]["shape"][idx3] == 0):
                                    multiClassGates[idx2]["shape"][idx3] = 1
                                    self.test_counter += 1
                                else:
                                    pass
                        else:
                            pass
            print("Furthermore, class ", str(classid), " activate nums of neurons: ", str(self.test_counter))

        return multiClassGates

    def mask_class_multi_by_value(self):
        '''
        Calculate sum of multi-class scalars
        '''
        print("RUNNING mask_class_multi_by_value.py")
        # print("Pruning Ratio: ", self.prune_ratio)
        multiClassGates = list()
        for classid in self.target_id:
            '''
            Merge JSONs continuously
            '''
            if self.mode == "class":
                json_path = "./ClassEncoding/class" + str(classid) + ".json"
            else:
                json_path = self.cluster_dir % classid

            with open(json_path, "r") as f:
                gatesValueDict = json.load(f)
                for idx in range(len(gatesValueDict)):
                    layer = gatesValueDict[idx]
                    name = layer["name"]
                    vec = layer["shape"]
                    # process name
                    name = name.split('/')[0]
                    gatesValueDict[idx]["name"] = name
                    gatesValueDict[idx]["shape"] = vec
                if not multiClassGates:
                    '''
                    Initialize the multiClassGates
                    '''
                    multiClassGates = gatesValueDict
                else:
                    '''
                    Now we merge gatesValueDict and multiClassGates
                    '''
                    for idx1 in range(len(gatesValueDict)):
                        for idx2 in range(len(multiClassGates)):
                            if (gatesValueDict[idx1]["name"] == multiClassGates[idx2]["name"]):
                                tomerge = gatesValueDict[idx1]["shape"]
                                for idx3 in range(len(tomerge)):
                                    multiClassGates[idx2]["shape"][idx3] += tomerge[idx3]
                            else:
                                pass
        '''
        Sort & Mask for multi-class conditions
        '''
        allGatesValue = []
        for idx in range(len(multiClassGates)):
            layer = multiClassGates[idx]
            name = layer["name"]
            vec = layer["shape"]
            allGatesValue += vec
                
        allGatesValue.sort()
        allGatesValue = allGatesValue[:int(len(allGatesValue)*self.prune_ratio)]
        allGatesValue = set(allGatesValue)
        
        result = multiClassGates

        for idx in range(len(result)):
            layer = result[idx]
            name = layer["name"]
            vec = layer["shape"]        
            # process name
            name = name.split('/')[0]
            # process vec
            for i in range(len(vec)):
                if vec[i] in allGatesValue:
                    vec[i] = 0
                else:
                    vec[i] = 1

            layer["name"] = name
            layer["shape"] = vec
        return result

    '''
    Assign mask weights: original control gates would be 0 or 1
    '''
    def assign_weight(self):
        '''
        Encapsulate unit-class pruning and multi-class pruning print("PRUNE FOR CLASS", self.target_class_id)
        '''
        print("assign weights......")
        maskDict = []
        if (len(self.target_id) > 1):
            maskDict = self.mask_class_multi_by_value()
        else:
            maskDict = self.mask_unit_by_value(self.target_id[0])

        for tmpLayer in maskDict:
            if (tmpLayer["name"][0] == "C"): # if the layer is convolutional layer
                with self.graph.as_default():
                    layerNum = tmpLayer["name"].strip("Conv")
                    name = "Conv" + layerNum + "/composite_function/gate:0"
                    for var in tf.global_variables():
                        if var.name == name:
                            tmpWeights = np.array(tmpLayer["shape"])
                            assign = tf.assign(var, tmpWeights)

                            # for i in range(len(tmpWeights)):
                            #     if tmpWeights[i]<0.001:
                            #         assign[i].assign(tf.stop_gradient(assign[i]))

                            self.sess.run(assign)
    
        print("assign finished!")
        '''
        Save the model
        '''
        # with self.graph.as_default():
        #     saver = tf.train.Saver(max_to_keep = None)
        #     saver.save(self.sess, 'vggNet/test.ckpt')

    '''
    Build VGG Network with Control Gate Lambdas
    '''
    def build_model(self, graph, label_count = 100):
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
            self.ys_orig = tf.placeholder("float", shape=[None, self.target_number])
            self.lr = tf.placeholder("float", shape=[])
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder("bool", shape=[])
            # weight_decay = 5e-4
            weight_decay = 0

            '''
            VGG Network Model Construction with Control Gates 
            '''
            current = self.xs
            if "data-d" in self.cluster_dir:
                with tf.variable_scope("Resize", reuse=tf.AUTO_REUSE):
                    current = tf.image.resize_images(current, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
            with tf.variable_scope("Conv1", reuse = tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 3, 64, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv2", reuse = tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 64, 64, 3, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv3", reuse = tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 64, 128, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv4", reuse = tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 128, 128, 3, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv5", reuse = tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 128, 256, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv6", reuse = tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 256, 256, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv7", reuse = tf.AUTO_REUSE):
                if "data-d" in self.cluster_dir:
                    current = self.batch_activ_conv(current, 256, 256, 3, self.is_training, self.keep_prob)
                else:
                    current = self.batch_activ_conv(current, 256, 256, 1, self.is_training, self.keep_prob) ###
                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv8", reuse = tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 256, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv9", reuse = tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv10", reuse = tf.AUTO_REUSE):
                if "data-d" in self.cluster_dir:
                    current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
                else:
                    current = self.batch_activ_conv(current, 512, 512, 1, self.is_training, self.keep_prob)

                current = self.maxpool2d(current, k=2)
            with tf.variable_scope("Conv11", reuse = tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv12", reuse = tf.AUTO_REUSE):
                current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
            with tf.variable_scope("Conv13", reuse = tf.AUTO_REUSE):
                if "data-d" in self.cluster_dir:
                    current = self.batch_activ_conv(current, 512, 512, 3, self.is_training, self.keep_prob)
                else:
                    current = self.batch_activ_conv(current, 512, 512, 1, self.is_training, self.keep_prob)
                current = self.maxpool2d(current, k=2)
                if "data-d" not in self.cluster_dir:
                    current = tf.reshape(current, [ -1, 512 ])
                else:
                    current = tf.reshape(current, [-1, 25088])
            with tf.variable_scope("FC14", reuse = tf.AUTO_REUSE):
                if "data-d" not in self.cluster_dir:
                    current = self.batch_activ_fc(current, 512, 4096, self.is_training)
                else:
                    current = self.batch_activ_fc(current, 25088, 4096, self.is_training)
            with tf.variable_scope("FC15", reuse = tf.AUTO_REUSE):
                current = self.batch_activ_fc(current, 4096, 4096, self.is_training)
            with tf.variable_scope("FC16", reuse = tf.AUTO_REUSE):
                Wfc = self.weight_variable_xavier([ 4096, self.target_number ], name = 'W')
                bfc = self.bias_variable([ self.target_number ])
                self.ys_pred = tf.matmul(current, Wfc) + bfc

            self.ys_pred_softmax = tf.nn.softmax(self.ys_pred)
            '''
            Loss Function
            '''
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels = self.ys_orig, logits = self.ys_pred_softmax
            ))
            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            total_loss = l2_loss * weight_decay + cross_entropy
            self.loss = total_loss
            
            '''
            Optimizer
            '''
            self.train_step = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True).minimize(total_loss)
                        
            '''
            Check whether correct
            '''
            correct_prediction = tf.equal(tf.argmax(self.ys_orig, 1), tf.argmax(self.ys_pred, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
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
        return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.variance_scaling_initializer(), trainable = True)

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.xavier_initializer(), trainable = True)

    def bias_variable(self, shape, name = 'bias'):
        initial = tf.constant(0.0, shape = shape)
        return tf.get_variable(name = name, initializer = initial, trainable = True)

    def gate_variable(self, length, name = 'gate'):
        initial = tf.constant([1.0] * length)
        v = tf.get_variable(name = name, initializer = initial, trainable = False)
        self.AllGateVariables[v.name] = v
        self.AllGateVariableValues.append(v)
        return v

    def conv2d(self, input, in_features, out_features, kernel_size, with_bias=False):
        W = self.weight_variable_msra([ kernel_size, kernel_size, in_features, out_features ], name = 'kernel')
        conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
        gate = self.gate_variable(out_features)
        conv = tf.multiply(conv, tf.abs(gate))
        if with_bias:
            return conv + self.bias_variable([ out_features ])
        return conv

    def batch_activ_conv(self, current, in_features, out_features, kernel_size, is_training, keep_prob):
        with tf.variable_scope("composite_function", reuse = tf.AUTO_REUSE):
            current = self.conv2d(current, in_features, out_features, kernel_size)
            if "data-d" not in self.cluster_dir:
                current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=True)
            # convValues.append(current)
            current = tf.nn.relu(current)
            #current = tf.nn.dropout(current, keep_prob)
        return current

    def batch_activ_fc(self, current, in_features, out_features, is_training):
        Wfc = self.weight_variable_xavier([ in_features, out_features ], name = 'W')
        bfc = self.bias_variable([ out_features ])
        current = tf.matmul(current, Wfc) + bfc
        # gate = self.gate_variable(out_features)
        # current = tf.multiply(current, tf.abs(gate))
        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=True)
        current = tf.nn.relu(current)
        return current

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                            padding='VALID')