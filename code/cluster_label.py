import json
import argparse
import os


if __name__=="__main__":
    parser = argparse.ArgumentParser("Cluster Label")
    parser.add_argument(
        "--encoder_save_path", help="The file to save the encoder.",
        default="data", type=str)
    parser.add_argument(
        "--start_class", help="start class of predictor",
        default=0, type=int
    )
    parser.add_argument(
        "--end_class", help="end class of predictor",
        default=99, type=int
    )
    args = parser.parse_args()

    dir = os.path.join(args.encoder_save_path,"sub_encoding.txt")
    with open(dir,"r") as file:
        sub_encoding = file.readlines()
    cluster_label = {}
    for l in sub_encoding:
        line = l.strip().split()
        cluster = int(float(line[0]))
        label = int(float(line[1]))
        if cluster not in cluster_label:
            cluster_label[cluster] = {label:1}
        else:
            if label in cluster_label[cluster]:
                cluster_label[cluster][label] += 1
            else:
                cluster_label[cluster][label] = 1
    keys = cluster_label
    clusters = {}
    for key in keys:
        clusters[key] = {}
        labels = sorted(cluster_label[key],key=cluster_label[key].__getitem__,reverse=True)
        for label in labels:
            clusters[key][label] =  cluster_label[key][label]
    cluster_label = clusters

    print(cluster_label)
    classifier_label = {}

    for key in cluster_label:
        cluster_keys = list(cluster_label[key].keys())
        cluster_size = sum(list(cluster_label[key].values()))
        if cluster_size<500*(args.end_class-args.start_class+1)/len(cluster_label)/5:
            continue
        tmp1 = []
        tmp2 = []
        for label in cluster_keys:
            if cluster_label[key][label]/cluster_size >0.05:
                tmp1.append(label)
                if cluster_label[key][label]<500*0.95:
                    tmp2.append(label)
        others = list(set(range(args.start_class,args.end_class+1))-set(tmp1))
        tmp2 += others
        classifier_label[key] = (tmp1,tmp2,cluster_size/(500*(args.end_class-args.start_class+1)))
        # cluster_label[key] = sorted(cluster_label[key],key=cluster_label[key].__getitem__,reverse=True)
        print(key,cluster_label[key],classifier_label[key])

    dir = os.path.join(args.encoder_save_path,"classifier.json")
    with open(dir,"w") as file:
        json.dump(classifier_label,file)
