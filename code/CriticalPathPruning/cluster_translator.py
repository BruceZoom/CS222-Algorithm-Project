import pickle
import json
from copy import deepcopy
from sklearn.cluster import KMeans, MiniBatchKMeans
# from utils import *
from CIFAR_DataLoader import CifarDataManager, display_cifar
# from subclass_encoder import SubclassEncoder
import sys
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Cluster predictor")
    parser.add_argument(
        "--encoder_save_path", help="The file to save the encoder.",
        default="../data", type=str)
    parser.add_argument(
        "--start_class", help="start class of predictor",
        default=0, type=int
    )
    parser.add_argument(
        "--end_class", help="end class of predictor",
        default=99, type=int
    )
    args = parser.parse_args()

    #
    # with open(args.encoder_save_path, "rb") as file:
    #     encoder = pickle.load(file)
    dir = os.path.join(args.encoder_save_path,"sub_encoding.txt")
    with open(dir,"r") as file:
        sub_encoding = file.readlines()
    classid_cluster = {}
    for l in sub_encoding:
        line = l.strip()
        line = line.split(" ")
        if len(line)<2:
            continue
        label = int(float(line[1]))

        cluster = int(float(line[0]))
        if label in classid_cluster:
            classid_cluster[label].append(cluster)
        else:
            classid_cluster[label] = [cluster]
    class_counter = {}
    class_clusters = {}
    for class_id in range(args.start_class,args.end_class+1):
        class_counter[class_id] = 0
        class_clusters[class_id] = []

    data_loader = CifarDataManager()

    batch_size = 100
    clusters = []
    for _ in range(500*100//batch_size):
        train_images, train_labels = data_loader.train.next_batch_without_onehot(batch_size)
        for i in range(batch_size):
            label = train_labels[i]
            if label<args.start_class or label>args.end_class:
                continue
            clusters.append(classid_cluster[label][class_counter[label]])
            class_clusters[label].append(classid_cluster[label][class_counter[label]])
            class_counter[label] += 1

    dir = os.path.join(args.encoder_save_path, "clusters.pkl")
    with open(dir,"wb") as file:
        pickle.dump(clusters,file)
    dir = os.path.join(args.encoder_save_path, "class_clusters.pkl")
    with open(dir,"wb") as file:
        pickle.dump(class_clusters,file)




