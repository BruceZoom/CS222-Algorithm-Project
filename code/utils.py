import os
import json
import numpy as np


def load_data(root="CriticalPathPruning/ImageEncoding", classes=None):
    if classes is None:
        classes = [dir for dir in os.listdir(root) if ".json" not in dir]
    data = []
    labels = []
    sample_names = []
    for class_id in classes:
        print("loading {}".format(class_id))
        path = os.path.join(root, class_id)
        for fname in os.listdir(path):
            with open(os.path.join(path, fname), "r") as f:
                res = json.load(f)
            tmp = []
            res.sort(key=lambda item: item["layer_name"])
            for layer in res:
                tmp += layer["layer_lambda"]
            data.append(tmp)
            labels.append(int(class_id[5:]))
            sample_names.append(fname)
    return np.array(data), np.array(labels), sample_names


def to_txt(path, data, labels):
    if not os.path.exists(path):
        os.mkdir(path)
    np.savetxt(os.path.join(path, "data.txt"), data)
    np.savetxt(os.path.join(path, "labels.txt"), labels)


if __name__ == '__main__':
    data, labels, sample_names = load_data(classes=['class0', 'class1', 'class2', 'class3', 'class4',])
    print(data.shape, labels.shape)
    print(np.histogram(labels, bins=15))
    to_txt("./data", data, labels)
