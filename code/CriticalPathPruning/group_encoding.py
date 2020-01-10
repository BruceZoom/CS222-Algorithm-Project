import os
import shutil
import sys


"""
Regroup your image encoding folder. Completed classes only.
"""
def group(IE_dir, CE_dir):
    safe_classes = [fname.split(".")[0] for fname in os.listdir(CE_dir)]
    files = [(fname, fname.split("-")[0]) for fname in os.listdir(IE_dir) if ".json" in fname]
    for fname, class_num in files:
        if class_num not in safe_classes:
            continue
        class_dir = os.path.join(IE_dir, class_num)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        shutil.move(os.path.join(IE_dir, fname), os.path.join(class_dir, fname))


if __name__ == '__main__':
    dir = ['vgg16-d/ImageEncoding_test', 'ClassEncoding']
    if len(sys.argv) > 1: dir[0] = sys.argv[1]
    if len(sys.argv) > 2: dir[1] = sys.argv[2]
    group(*dir)
