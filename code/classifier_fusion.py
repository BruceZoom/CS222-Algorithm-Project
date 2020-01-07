import json
import math
import argparse
from queue import PriorityQueue


class Element(object):
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description

    def __lt__(self, other):
        return self.priority < other.priority

    def get_value(self):
        return self.description


def check(used,encodings):
    for code in encodings:
        if len(encodings[code])>1:
            return False
    return True


def renew(classifier,used,encodings):
    pos = classifer_label[classifier][0]
    neg = classifer_label[classifier][1]
    new_used = used + [classifier]
    new_encodings = {}
    other_pos = set()
    for classifier_id in classifier_ids-set(used)-{classifier}:
        other_pos = other_pos | classifer_label[classifier_id][0]
    for key,label_set in encodings.items():
        if pos & label_set == set() or neg & label_set == set() or len(label_set)==1:
            code = key + "2"
            new_encodings[code] = label_set
            continue
        code1 = key + "0"
        new_encodings[code1] = label_set & pos
        code2 = key + "1"
        new_encodings[code2] = label_set & neg & other_pos
    return new_used,new_encodings


def calculate_entropy(classifier,used,encodings):
    new_used, new_encodings = renew(classifier,used,encodings)
    entropy = 0
    for key in new_encodings:
        p = 1
        for i in range(len(key)):
            classifier_id = new_used[i]
            if key[i] == "0":
                p = p * classifier_poss[classifier_id]
            if key[i] == "1":
                p = p * (1 - classifier_poss[classifier_id])
            if key[i] == "2":
                p = p * 1
            entropy += p * math.log(len(new_encodings[key]),2)
    return entropy


def heuristic_search(used,encodings):
    # print(used,encodings)
    if check(used,encodings):
        return used,encodings,True
    if len(used)==k:
        return used,encodings,False
    pq = PriorityQueue()
    for classifier in classifier_ids-set(used):
        entropy = calculate_entropy(classifier,used,encodings)
        pq.put(Element(entropy, classifier))
    while not pq.empty():
        classifier = pq.get()
        classifier = classifier.get_value()
        new_used, new_encodings = renew(classifier,used,encodings)
        res_used, res_encoding, flag = heuristic_search(new_used,new_encodings)
        if flag:
            return res_used, res_encoding, flag
    return used,encodings, False


parser = argparse.ArgumentParser("Classifier Fusion")
parser.add_argument(
    "--start_label",help="the start label to be classified",
    default=0,type=int
)
parser.add_argument(
    "--end_label",help="the end label to be classified",
    default=99,type=int
)
parser.add_argument(
    "--max_classifier",help="max number of classifier",
    default=200,type=int
)
args = parser.parse_args()
k = args.max_classifier
with open("classifier.json","r") as file:
    classifier = json.load(file)
classifier_ids = set(classifier.keys())
classifer_label = {}
classifier_poss = {}
for id in classifier:
    classifer_label[id] = (set(classifier[id][0]),set(classifier[id][1]))
    classifier_poss[id] = classifier[id][2]
encodings = {"2":set(range(args.start_label,args.end_label+1))}
chosen = heuristic_search([-1],encodings)
print(chosen)