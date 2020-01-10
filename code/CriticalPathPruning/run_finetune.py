from vggFinetuneModel import FineTuneModel
from CIFAR_DataLoader import CifarDataManager
import numpy as np
import argparse
import json
import os

'''
For fine tune model, the data label should be different:
    if the fine tune trimmed model if for classify label:[0]
    the fine tune model data labels would be of 2
    Thus, in this case, we need to change the label correspongdingly then go to train 
'''
def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

def modify_label(labels, test_classes = [0]):
    test_classes.sort()
    tmp_labels = []
    for i in labels:
        if i in test_classes:
            tmp_labels.append(test_classes.index(i))
        else:
            tmp_labels.append(len(test_classes))
    return one_hot(tmp_labels, vals=len(test_classes)+1)


parser = argparse.ArgumentParser("run finetune")
parser.add_argument(
    "--mode", help="run finetune for 'class' or 'cluster' CDRP classifier",
    default="cluster",type=str)
parser.add_argument(
    "--save_dir", help="directory to save trained classifiers",
    default="classifiers",type=str)
parser.add_argument(
    "--cluster_dir", help="directory saving clusters",
    default="../data",type=str)
args = parser.parse_args()

mode = args.mode
assert mode in ["class","cluster"]

data_loader = CifarDataManager(cluster_dir=args.cluster_dir)

# cluster_ids = [3]
# class_ids = [0]
corre = [([3],[0]),([4],[3]),([1],[1]),([2],[2]),([6],[4]),([0],[4])]
# corre = [([2],[2,4,3]),([6],[0]),([1],[0]),([0],[3,4,1]),([4],[3,4]),([3],[1]),([5],[1])]
# corre = [([3],[0]),([4],[3])]
# test_images, test_labels = data_loader.test.next_batch_balance_without_onehot(200,class_ids)
for item in corre:
    cluster_ids = item[0]
    class_ids = item[1]
    print(cluster_ids,class_ids)
    if mode == "class":
        target_ids = class_ids
    else:
        target_ids = cluster_ids
    if mode == "class":
        test_images, test_labels = data_loader.test.next_batch_balance_without_onehot(200, class_ids, mode=mode)
    else:
        test_images, test_labels = data_loader.test.next_batch_balance_without_onehot(200,class_ids,mode = mode,type="test")
    print(test_labels)
    test_labels = modify_label(test_labels, test_classes = target_ids)

    model = FineTuneModel(target_class_id=class_ids,target_cluster_id = cluster_ids, mode = mode, cluster_dir = args.cluster_dir)
    model.assign_weight()
    model.test_accuracy(test_images, test_labels)

    for i in range(1500):
        print(i)
        train_images, train_labels = data_loader.train.next_batch_balance_without_onehot(200,class_ids,cluster_ids,mode = mode)
        # train_images, train_labels = data_loader.train.next_batch_without_onehot(100)

        train_labels = modify_label(train_labels, test_classes = target_ids)
        model.train_model(train_images, train_labels)
        # test_images, test_labels = data_loader.test.next_batch_balance_without_onehot(100, class_ids)
        # test_labels = modify_label(test_labels, test_classes=class_ids)
        model.test_accuracy(test_images, test_labels)

    model.assign_weight()
    res = model.test_accuracy(test_images, test_labels)
    res = [float(i) for i in res]
    print(res)
    if args.cluster_dir=="../data-d-":
        id = cluster_ids[0]+7
    else:
        id = cluster_ids[0]
    model.save_model(id,args.save_dir)
    dir = os.path.join(args.save_dir,"accuracy","classifier%d.json"%id)
    with open(dir,"w") as file:
        json.dump(res,file)