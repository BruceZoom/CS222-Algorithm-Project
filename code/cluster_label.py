import json


if __name__=="__main__":
    with open("data/sub_encoding.txt","r") as file:
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


    classifier_label = {}

    for key in cluster_label:
        cluster_keys = list(cluster_label[key].keys())
        cluster_size = sum(list(cluster_label[key].values()))
        if cluster_size<500*100/len(cluster_label)/5:
            continue
        tmp1 = []
        tmp2 = []
        for label in cluster_keys:
            if cluster_label[key][label]/cluster_size >0.05:
                tmp1.append(label)
                if cluster_label[key][label]<500*0.95:
                    tmp2.append(label)
        others = list(set(range(100))-set(tmp1))
        tmp2 += others
        classifier_label[key] = (tmp1,tmp2,cluster_size/(500*100))
        # cluster_label[key] = sorted(cluster_label[key],key=cluster_label[key].__getitem__,reverse=True)
        print(key,cluster_label[key],classifier_label[key])
    with open("classifier.json","w") as file:
        json.dump(classifier_label,file)
