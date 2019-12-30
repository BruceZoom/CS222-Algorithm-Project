import os
import json
import numpy as np

# layer names are sorted
layer2idx = {
    'Conv1': (0, 64),
    'Conv10': (64, 576),
    'Conv11': (576, 1088),
    'Conv12': (1088, 1600),
    'Conv13': (1600, 2112),
    'Conv2': (2112, 2176),
    'Conv3': (2176, 2304),
    'Conv4': (2304, 2432),
    'Conv5': (2432, 2688),
    'Conv6': (2688, 2944),
    'Conv7': (2944, 3200),
    'Conv8': (3200, 3712),
    'Conv9': (3712, 4224),
    'FC14': (4224, 8320),
    'FC15': (8320, 12416)
}


# layer2idx = {
#     'Conv1': (0, 64),
#     'Conv10': (2176, 2688),
#     'Conv11': (2688, 3200),
#     'Conv12': (3200, 3712),
#     'Conv13': (3712, 4224),
#     'Conv2': (64, 128),
#     'Conv3': (128, 256),
#     'Conv4': (256, 384),
#     'Conv5': (384, 640),
#     'Conv6': (640, 896),
#     'Conv7': (896, 1152),
#     'Conv8': (1152, 1664),
#     'Conv9': (1664, 2176),
#     'FC14': (4224, 8320),
#     'FC15': (8320, 12416),
# }


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


def update_layer2idx():
    dist = 0
    for key, item in layer2idx.items():
        incr = item[1] - item[0]
        layer2idx[key] = (dist, dist + incr)
        dist += incr
    print(layer2idx)


def normalize(x, lim):
    return (x - x.min()) / (x.max() - x.min()) * (lim[1] - lim[0]) + lim[0]


if __name__ == '__main__':
    data, labels, sample_names = load_data(classes=['class0', 'class1', 'class2', 'class3', 'class4', ])
    print(data.shape, labels.shape)
    print(np.histogram(labels, bins=np.unique(labels).shape[0]))
    to_txt("./data", data, labels)

    # update_layer2idx()
    pass
