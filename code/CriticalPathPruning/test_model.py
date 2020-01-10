import json
import pickle
import os
from vggClassifierModel import ClassifierModel
from CIFAR_DataLoader import CifarDataManager
import numpy as np
import argparse


def find_label(encodings,code):
    codes = list(encodings.keys())
    for i in range(len(code)):
        new_codes = []
        for item in codes:
            if item[i]=="2" or item[i]==code[i]:
                new_codes.append(item)
        codes = new_codes
        if len(new_codes)==1:
            break
    print(codes[0],encodings[codes[0]])
    return list(encodings[codes[0]])[0]



if __name__=="__main__":
    parser = argparse.ArgumentParser("run finetune")
    parser.add_argument(
        "--model_dir", help="the directory saving the classifiers",
        default="classifiers",type=str)
    parser.add_argument(
        "--chosen_dir", help="the pickle file saving the chosen classifers and encodings",
        default="../chosen_classifier.pkl",type=str)
    args = parser.parse_args()

    data_loader = CifarDataManager()
    class_ids = [0,1,2,3,4]
    cluster_ids = list(range(0,20))
    test_images = []
    test_labels = []
    for id in class_ids:
        test_images += data_loader.test.class_image[id]
        test_labels += data_loader.test.class_label[id]
    num = len(test_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    # test_images = data_loader.test.next_batch_balance_without_onehot(num, class_ids, mode="class")

    with open(args.chosen_dir,"rb") as file:
        used, encodings = pickle.load(file,encoding="bytes")
    codes = [""]*num
    for item in used:
        if int(item)<7:
            vgg_type = "C"
        else:
            vgg_type = "D"
        model = ClassifierModel(vgg_type)
        filename = os.path.join(args.model_dir,"classifier%d.ckpt"%int(item))
        model.from_checkpoint(filename)
        classifier_res = model.predict(test_images, test_labels)
        for i in range(len(classifier_res)):
            codes[i] = codes[i] + str(classifier_res[i])

    correct = 0
    # print(codes)
    # print(test_labels)
    count_label = [0]*5
    for i in range(len(codes)):
        label = find_label(encodings,codes[i])
        # print(label,test_labels[i])
        # print(type(label),type(test_labels[i]))
        if label == test_labels[i]:
            correct += 1
            count_label[label] += 1
        elif label not in class_ids and test_labels[i] not in class_ids:
            correct += 1
            count_label[label] += 1

    print("accuracy:", correct/num, count_label)
