from vggTrimmedModel import TrimmedModel
from CIFAR_DataLoader import CifarDataManager
import numpy as np
d = CifarDataManager()


model = TrimmedModel(target_class_id=[99],target_cluster_id = [0],mode="class",
                     multiPruning=True)

'''
Todo List: 
    1. Modify the accuracy in trimmed network (Done)
'''
for _ in range(50):
    test_images, test_labels = d.test.next_batch_balance(200,[99])

    model.test_accuracy_pretrim(test_images, test_labels)
    model.assign_weight()
    model.test_accuracy(test_images, test_labels)
    break




