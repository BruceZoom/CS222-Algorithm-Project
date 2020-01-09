from vgg16 import Model, NameMapping
import numpy as np
import argparse
import os
import sys
import tensorflow as tf
from CIFAR_DataLoader import CifarDataManager

data_loader = CifarDataManager()
model = Model()
checkpoint_path = "vgg-imagenet/vgg16-imagenet.ckpt"
name_mapping = NameMapping.imagenet_mapping_
output_size = 100

print("Restoring network...")
graph = tf.Graph()
model.build_model(graph, output_size)
model.restore_model(graph, checkpoint_path, name_mapping)

max_acc = 0

for i in range(1000):
    print(i)
    train_images, train_labels = data_loader.train.next_batch(200)
    model.train_model(train_images, train_labels)
    test_images, test_labels = data_loader.test.next_batch(200)
    cur_acc = model.test_accuracy(test_images, test_labels)
    if i % 100 == 0:
        model.save_ckpt()
        if cur_acc > max_acc:
            model.save_ckpt('best')
            max_acc = cur_acc


model.close_sess()
