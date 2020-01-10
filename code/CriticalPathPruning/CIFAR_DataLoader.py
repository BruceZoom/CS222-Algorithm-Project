# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
import random
import _pickle as cPickle
import pickle

# =============== Define Data Loader ===============
''' Utility Functions '''
DATA_PATH = "cifar-100-python"

def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
        return dict

def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

class CifarLoader(object):
    def __init__(self, sourcefiles):
        self._source = sourcefiles
        self._i = 0

        self.images = None
        self.labels = None
        self.class_image = None
        self.class_label = None

    def load(self, load_cluster=False, cluster_dir = "../data"):
        if load_cluster:
            dir = os.path.join(cluster_dir,"clusters.pkl")
            with open(dir,"rb") as file:
                self.cluster = pickle.load(file,encoding="bytes")
            dir = os.path.join(cluster_dir,"class_clusters.pkl")
            with open(dir,"rb") as file:
                self.class_cluster = pickle.load(file,encoding="bytes")

            dir = os.path.join(cluster_dir,"clusters_test.pkl")
            with open(dir,"rb") as file:
                self.cluster_test = pickle.load(file,encoding="bytes")
            dir = os.path.join(cluster_dir,"class_clusters_test.pkl")
            with open(dir,"rb") as file:
                self.class_cluster_test = pickle.load(file,encoding="bytes")

        data = [unpickle(f) for f in self._source]
        images = np.vstack(d["data"] for d in data)

        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float)
        # self.labels = one_hot(np.hstack([d["fine_labels"] for d in data]), 100)
        self.labels = np.hstack([d["fine_labels"] for d in data])
        self.images = self.normalize_images(self.images)

        self.class_image = []
        self.class_label = []
        for id in range(100):
            self.class_label.append(list(self.labels[self.labels==id]))
            self.class_image.append(list(self.images[self.labels == id]))

        # self.class_label = np.array(self.class_label)
        # self.class_image = np.array(self.class_image)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i:self._i+batch_size], self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, one_hot(y, 100)

    def next_batch_balance(self,batch_size,target_class_id,cluster_id=[0],mode = "class", type = "train"):
        num = batch_size // (len(target_class_id)+1)
        tmp = []
        if type == "train":
            class_cluster = self.class_cluster
        else:
            class_cluster = self.class_cluster_test
        for class_id in target_class_id:
            # class_image = random.sample(self.class_image[class_id],num)
            # class_label = random.sample(self.class_label[class_id],num)
            for _ in range(num):
                i = random.randint(0,len(self.class_image[class_id])-1)
                # if type=="train" and class_cluster[class_id][i] not in cluster_id:
                #     i = random.randint(0, len(self.class_image[class_id]) - 1)
                tmp.append((self.class_image[class_id][i],self.class_label[class_id][i],class_cluster[class_id][i]))

        other_num = batch_size - num * len(target_class_id)
        other_id = list(set(range(5))-set(target_class_id))
        for j in range(other_num):
            id = random.choice(other_id)
            i = random.randint(0,len(self.class_image[id])-1)
            # image = random.choice(self.class_image[id])
            tmp.append((self.class_image[id][i],id,class_cluster[id][i]))
        random.shuffle(tmp)
        x = []
        y = []
        z = []
        for i in range(batch_size):
            x.append(tmp[i][0])
            if mode == "class":
                y.append(tmp[i][1])
                hot_num = 100
            else:
                y.append(tmp[i][2])
                hot_num = 200
        x = np.array(x)
        y = np.array(y)
        return x, one_hot(y,hot_num)

    def next_batch_balance_without_onehot(self,batch_size,target_class_id,cluster_id=[0],mode="class",type = "train"):
        num = batch_size // (len(target_class_id)+1)
        tmp = []
        if type == "train":
            class_cluster = self.class_cluster
        else:
            class_cluster = self.class_cluster_test
        for class_id in target_class_id:
            # class_image = random.sample(self.class_image[class_id],num)
            # class_label = random.sample(self.class_label[class_id],num)
            for _ in range(num):
                i = random.randint(0,len(self.class_image[class_id])-1)
                # if type=="train" and class_cluster[class_id][i] not in cluster_id:
                #     i = random.randint(0, len(self.class_image[class_id]) - 1)
                tmp.append((self.class_image[class_id][i],self.class_label[class_id][i],class_cluster[class_id][i]))

        other_num = batch_size - num * len(target_class_id)
        other_id = list(set(range(5))-set(target_class_id))
        for j in range(other_num):
            id = random.choice(other_id)

            i = random.randint(0,len(self.class_image[id])-1)
            # image = random.choice(self.class_image[id])
            tmp.append((self.class_image[id][i],id,class_cluster[id][i]))
        random.shuffle(tmp)
        x = []
        y = []
        for i in range(batch_size):
            x.append(tmp[i][0])
            if mode=="class":
                y.append(tmp[i][1])
            else:
                y.append(tmp[i][2])
        x = np.array(x)
        y = np.array(y)
        return x, y

    def next_batch_without_onehot(self, batch_size):
        x, y = self.images[self._i:self._i+batch_size], self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

    def generateSpecializedData(self, class_id, count = 500):
        train_index = []

        index = list(np.where(self.labels[:] == class_id)[0])[0:count]
        train_index += index

        sp_x = self.images[train_index]
        sp_y = self.labels[train_index]
        sp_y = one_hot(sp_y, 100)

        sp_y = sp_y.astype('float32')
        sp_x = sp_x.astype('float32')
        return sp_x, sp_y
    
    def generateAllData(self):
        return self.images, one_hot(self.labels, 100)

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


# ============ Data Manager: Wrap the Data Loader===============
class CifarDataManager(object):
    def __init__(self, load_cluster=True, cluster_dir = "../data"):
        '''
        CIFAR 10 Data Set 
        '''
        # self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1,6)]).load()
        # self.test = CifarLoader(["test_batch"]).load()

        '''
        CIFAR 100 Data Set 
        '''
        self.train = CifarLoader(["train"]).load(load_cluster, cluster_dir)
        self.test = CifarLoader(["test"]).load(load_cluster, cluster_dir)

def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
    for i in range(size)])
    plt.imshow(im)
    plt.show()

# d = CifarDataManager()
# print("Number of train images: {}".format(len(d.train.images)))
# print("Number of train labels: {}".format(len(d.train.labels)))
# print("Number of test images: {}".format(len(d.test.images)))
# print("Number of test images: {}".format(len(d.test.labels)))
# images = d.train.images
# print(images[0])
# display_cifar(images, 10)
# print(images.shape)

# meta_dict = unpickle("meta")
# print(meta_dict)
