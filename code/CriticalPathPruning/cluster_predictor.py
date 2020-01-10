import pickle
import json
from copy import deepcopy
from sklearn.cluster import KMeans, MiniBatchKMeans
# from utils import *
from CIFAR_DataLoader import CifarDataManager, display_cifar
from subclass_encoder import SubclassEncoder
import sys
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Cluster predictor")
    parser.add_argument(
        "--start_class", help="start class of predictor",
        default=0, type=int
    )
    parser.add_argument(
        "--end_class", help="end class of predictor",
        default=99, type=int
    )
    parser.add_argument(
        "--encoder_save_path", help="The file to save the encoder.",
        default="../data", type=str)
    parser.add_argument(
        "--image_encoding_save_path", help="The file to save the image encodings",
        default="ImageEncoding_test", type=str)

    args = parser.parse_args()


    class_counter = {}
    class_clusters = {}
    for class_id in range(args.start_class,args.end_class+1):
        class_counter[class_id] = 0
        class_clusters[class_id] = []


    dir = os.path.join(args.encoder_save_path,"encoder.pkl")
    with open(dir, "rb") as file:
        encoder = pickle.load(file)
    kmeans = encoder.encoders[0]

    data_loader = CifarDataManager()

    batch_size = 100
    clusters = []
    for _ in range(100*100//batch_size):
        print(_)
        train_images, test_labels = data_loader.test.next_batch_without_onehot(batch_size)
        for i in range(batch_size):
            label = test_labels[i]
            if label<args.start_class or label>args.end_class:
                continue
            path = "class%d/class%d-pic%d.json"%(label,label,class_counter[label])
            path = os.path.join(args.image_encoding_save_path,path)
            with open(path, "r") as file:
                res = json.load(file)
            tmp = []
            res = sorted(res, key=lambda item: item["layer_name"])
            for layer in res:
                tmp += layer["layer_lambda"]
            cluster = kmeans.predict([tmp])[0]
            clusters.append(cluster)
            class_clusters[label].append(cluster)
            class_counter[label] += 1

    dir = os.path.join(args.encoder_save_path,"clusters_test.pkl")
    with open(dir,"wb") as file:
        pickle.dump(clusters,file)
    dir = os.path.join(args.encoder_save_path,"class_clusters_test.pkl")
    with open(dir,"wb") as file:
        pickle.dump(class_clusters,file)
    for key in class_clusters:
        print(key,class_clusters[key])