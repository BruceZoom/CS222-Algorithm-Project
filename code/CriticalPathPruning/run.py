import numpy as np
import os
import json
from CIFAR_DataLoader import CifarDataManager, display_cifar
from vggNet import Model, NameMapping
import argparse
import warnings
warnings.filterwarnings("ignore")


'''
Configuration:
    1. learning rate 
    2. L1 loss penalty
    3. Lambda cotrol gate threshold
'''

layer_names = ['FC14/gate:0',
               'Conv7/composite_function/gate:0',
               'Conv12/composite_function/gate:0',
               'Conv2/composite_function/gate:0',
               'Conv4/composite_function/gate:0',
               'Conv1/composite_function/gate:0',
               'Conv6/composite_function/gate:0',
               'Conv10/composite_function/gate:0',
               'Conv9/composite_function/gate:0',
               'FC15/gate:0',
               'Conv13/composite_function/gate:0',
               'Conv5/composite_function/gate:0',
               'Conv3/composite_function/gate:0',
               'Conv11/composite_function/gate:0',
               'Conv8/composite_function/gate:0'
               ]


def calculate_total_by_threshold(classid, size=500, dataset='cifar'):
    name_shape_match = [
        {'name': 'FC14/gate:0', 'shape': 4096},
        {'name': 'Conv7/composite_function/gate:0', 'shape': 256},
        {'name': 'Conv12/composite_function/gate:0', 'shape': 512},
        {'name': 'Conv2/composite_function/gate:0', 'shape': 64},
        {'name': 'Conv4/composite_function/gate:0', 'shape': 128},
        {'name': 'Conv1/composite_function/gate:0', 'shape': 64},
        {'name': 'Conv6/composite_function/gate:0', 'shape': 256},
        {'name': 'Conv10/composite_function/gate:0', 'shape': 512},
        {'name': 'Conv9/composite_function/gate:0', 'shape': 512},
        {'name': 'FC15/gate:0', 'shape': 4096},
        {'name': 'Conv13/composite_function/gate:0', 'shape': 512},
        {'name': 'Conv5/composite_function/gate:0', 'shape': 256},
        {'name': 'Conv3/composite_function/gate:0', 'shape': 128},
        {'name': 'Conv11/composite_function/gate:0', 'shape': 512},
        {'name': 'Conv8/composite_function/gate:0', 'shape': 512}
    ]
    for i in range(len(name_shape_match)):
        name_shape_match[i]['shape'] = name_shape_match[i]['shape'] * [0]
    for i in range(size):
        jsonpath = "./ImageEncoding/class" + str(classid) + "-pic" + str(i) + ".json"
        with open(jsonpath, 'r') as f:
            dataset = json.load(f)
            for gate in range(len(dataset)):
                for index in range(len(name_shape_match)):
                    if name_shape_match[index]['name'] == dataset[gate]['layer_name']:
                        tmp = dataset[gate]['layer_lambda']
                        for conv in range(len(tmp)):
                            if tmp[conv] > 0.1:  # threshold 0.1
                                name_shape_match[index]['shape'][conv] += 1
                            else:
                                pass
                    else:
                        pass
    json_write_path = "./ClassEncoding/class" + str(classid) + ".json"
    with open(json_write_path, 'w') as g:
        json.dump(name_shape_match, g, sort_keys=True, indent=4, separators=(',', ':'))


def calculate_total_by_weights(classid, size=500, dataset='cifar'):
    name_shape_match = [
        {'name': 'FC14/gate:0', 'shape': 4096},
        {'name': 'Conv7/composite_function/gate:0', 'shape': 256},
        {'name': 'Conv12/composite_function/gate:0', 'shape': 512},
        {'name': 'Conv2/composite_function/gate:0', 'shape': 64},
        {'name': 'Conv4/composite_function/gate:0', 'shape': 128},
        {'name': 'Conv1/composite_function/gate:0', 'shape': 64},
        {'name': 'Conv6/composite_function/gate:0', 'shape': 256},
        {'name': 'Conv10/composite_function/gate:0', 'shape': 512},
        {'name': 'Conv9/composite_function/gate:0', 'shape': 512},
        {'name': 'FC15/gate:0', 'shape': 4096},
        {'name': 'Conv13/composite_function/gate:0', 'shape': 512},
        {'name': 'Conv5/composite_function/gate:0', 'shape': 256},
        {'name': 'Conv3/composite_function/gate:0', 'shape': 128},
        {'name': 'Conv11/composite_function/gate:0', 'shape': 512},
        {'name': 'Conv8/composite_function/gate:0', 'shape': 512}
    ]
    for i in range(len(name_shape_match)):
        name_shape_match[i]['shape'] = name_shape_match[i]['shape'] * [0]
    for i in range(size):
        if dataset == 'cifar':
            if not os.path.exists('ImageEncoding'):
                os.mkdir('ImageEncoding')
            jsonpath = "./ImageEncoding/class" + str(classid) + "-pic" + str(i) + ".json"
        else:
            if not os.path.exists(dataset + "-ImageEncoding"):
                os.mkdir(dataset + "-ImageEncoding")
            jsonpath = "./" + dataset + "-ImageEncoding/class" + str(classid) + "-pic" + str(i) + ".json"
        with open(jsonpath, 'r') as f:
            dataset = json.load(f)
            for gate in range(len(dataset)):
                for index in range(len(name_shape_match)):
                    if name_shape_match[index]['name'] == dataset[gate]['layer_name']:
                        tmp = dataset[gate]['layer_lambda']
                        for conv in range(len(tmp)):
                            name_shape_match[index]['shape'][conv] += tmp[conv]
                    else:
                        pass
    if dataset == 'cifar':
        if not os.path.exists('ClassEncoding'):
            os.mkdir('ClassEncoding')
        json_write_path = "./ClassEncoding/class" + str(classid) + ".json"
    else:
        if not os.path.exists(dataset + "-ClassEncoding"):
            os.mkdir(dataset + "-ClassEncoding")
        json_write_path = "./" + dataset + "-ClassEncoding/class" + str(classid) + ".json"
    with open(json_write_path, 'w') as g:
        json.dump(name_shape_match, g, sort_keys=True, indent=4, separators=(',', ':'))


if __name__ == '__main__':
    def bool_string(input_string):
        if input_string not in {"True", "False"}:
            raise ValueError("Please Enter a valid Ture/False choice")
        else:
            return (input_string == "True")

    parser = argparse.ArgumentParser(description="CDRP Decompose")
    parser.add_argument(
        "--dataset", help="Dataset to use. Either 'imagenet' or 'cifar'.",
        default='cifar', type=str)
    parser.add_argument(
        "--l1_loss_penalty", help="Penalty coefficient for L1 loss.",
        default=0.03, type=float)
    parser.add_argument(
        "--entropy_penalty", help="Penalty coefficient for L1 loss.",
        default=1, type=float)
    parser.add_argument(
        "--learning_rate", help="Learning rate of CDRP.",
        default=0.1, type=float)
    parser.add_argument(
        "--threshold", help="Lambda control gate threshold.",
        default=0.0, type=float)
    parser.add_argument(
        "--max_samples", help="Max number of samples per class.",
        default=500, type=int)
    parser.add_argument(
        "--begin_class", help="Class to begin.",
        default=0, type=int)
    parser.add_argument(
        "--end_class", help="Class to end.",
        default=99, type=int)
    parser.add_argument(
        "--verbose", help="Print messages.",
        default=True, type=bool_string)
    parser.add_argument(
        "--save_folder", help="folder to save results",
        default=None, type=str)
    parser.add_argument(
        "--mode", help="'train' or 'test' data",
        default="train", type=str)

    args = parser.parse_args()
    assert args.dataset in ['imagenet', 'cifar'], "--dataset can only be 'imagenet' or 'cifar'."

    d = CifarDataManager()
    model = Model(
        learning_rate=args.learning_rate,
        L1_loss_penalty=args.l1_loss_penalty,
        threshold=args.threshold,
        entropy_penalty=args.entropy_penalty
    )

    for i in range(args.begin_class, args.end_class+1):
        print("Current class: {}".format(i))
        print("Gnerating data...")
        if args.mode == "train":
            train_images, train_labels = d.train.generateSpecializedData(class_id=i, count=args.max_samples)
        else:
            train_images, train_labels = d.test.generateSpecializedData(class_id=i, count=args.max_samples)
        print("Encoding data...")
        if args.dataset == 'cifar':
            model.encode_class_data(i, train_images,save_folder = args.save_folder)
        elif args.dataset == 'imagenet':
            model.encode_class_data(i, train_images, "vgg-imagenet/vgg16-imagenet.ckpt", NameMapping.imagenet_mapping_,
                                    use_batch_norm=False, with_bias=True, vgg_type='D', output_size=1000,
                                    dataset_name='imagenet', verbose=args.verbose, save_folder = args.save_folder)
        print("Encoding class...")
        #calculate_total_by_weights(i, size=args.max_samples, dataset=args.dataset)
