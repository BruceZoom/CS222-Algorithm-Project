import os
import json
import numpy as np
import argparse

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
            res = sorted(res, key=lambda item: item["layer_name"])
            for layer in res:
                tmp += layer["layer_lambda"]
            # assert len(tmp) == 12416, fname + " is bad shaped."
            sample_names.append(fname)
            if len(tmp) != 12416:
                print(fname + " is bad shaped.")
                data.append(np.array(data[-1][:]))
                labels.append(labels[-1])
                continue
            data.append(np.array(tmp))
            labels.append(int(class_id[5:]))
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


def idx2layer(data):
    res = []
    for layer in layer2idx:
        l = layer2idx[layer][0]
        r = layer2idx[layer][1]
        tmp = {}
        if layer[0]=="C":
            tmp["name"] = layer+"/"+"composite_function/gate:0"
        else:
            tmp["name"] = layer+"/gate:0"
        tmp["shape"] = list(data[l:r])
        res.append(tmp)
    return res



def normalize(x, lim):
    return (x - x.min()) / (x.max() - x.min()) * (lim[1] - lim[0]) + lim[0]


def bool_string(s):
    s = str.lower(s)
    assert s in ["true", "false"], "The bool value can only be 'True' or 'False'."
    return s == "true"


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Utils")
    # parser.add_argument("prog", help="One of the 'CvtTxt'.", type=str)
    parser.add_argument(
        "--encoding_dir", help="Encoding directory.",
        default="CriticalPathPruning/ImageEncoding", type=str)
    parser.add_argument(
        "--save_dir", help="The directory to save converted data.",
        default="data", type=str)
    parser.add_argument(
        "--conv_only", help="Only save the convolution layers.",
        default="False", type=bool_string)
    parser.add_argument(
        "--begin", help="The index of the class to begin. Inclusive.",
        default=0, type=int)
    parser.add_argument(
        "--end", help="The index of the class to end. Exclusive",
        default=100, type=int)
    args = parser.parse_args()

    data, labels, sample_names = load_data(root=args.encoding_dir,
                                           classes=['class{}'.format(i) for i in range(args.begin, args.end)])
    print(data.shape, labels.shape)
    print(np.histogram(labels, bins=np.unique(labels).shape[0]))
    if args.conv_only:
        to_txt(args.save_dir, data[:, :layer2idx['FC14'][0]], labels)
    else:
        to_txt(args.save_dir, data, labels)

    # update_layer2idx()
    pass
