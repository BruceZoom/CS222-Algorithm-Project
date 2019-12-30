import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from utils import *


def create_gif(image_list, gif_name, duration=1.0, extend=False):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    if extend:
        frames.append(imageio.imread(image_list[-1]))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)


def pca_isolation_gif(data_path="data/data.txt", label_path="data/labels.txt", data=None, labels=None,
               gif_name="data/pca.gif", duration=0.6, x_lim=(-5, 5), y_lim=(-5, 5),
               flips=None, rotates=None):
    if data is None:
        print("Importing data...")
        data = np.loadtxt(data_path)
    if labels is None:
        print("Importing labels...")
        labels = np.loadtxt(label_path)
    classes = np.unique(labels)

    if not os.path.exists(gif_name + "_"):
        os.mkdir(gif_name + "_")

    X = None
    for i in range(1, 14):
        print("Current plot: {}".format(i))
        plt.figure(figsize=(6, 5))
        layer = 'Conv{}'.format(i)
        idx = layer2idx[layer]
        if X is None:
            X = data[:, :idx[1]]
        else:
            X = np.concatenate([X, data[:, idx[0]:idx[1]]], axis=1)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        if flips is not None:
            if 1 in flips[i-1]:
                # horizontal flip
                X_pca[:, 0] = -X_pca[:, 0]
            if 2 in flips[i-1]:
                # vertical flip
                X_pca[:, 1] = -X_pca[:, 1]
            if 3 in flips[i-1]:
                # y = -x flip
                tmp = X_pca[:, 0].copy()
                X_pca[:, 0] = -X_pca[:, 1]
                X_pca[:, 1] = -tmp
            if 4 in flips[i-1]:
                # y = x flip
                tmp = X_pca[:, 0].copy()
                X_pca[:, 0] = X_pca[:, 1]
                X_pca[:, 1] = tmp
        if rotates is not None:
            tmp = X_pca.copy()
            X_pca[:, 0] = np.cos(rotates[i-1]) * X_pca[:, 0] + np.sin(rotates[i-1]) * X_pca[:, 1]
            X_pca[:, 1] = -np.sin(rotates[i-1]) * X_pca[:, 0] + np.cos(rotates[i-1]) * X_pca[:, 1]
        X_pca[:, 0] = normalize(X_pca[:, 0], x_lim)
        X_pca[:, 1] = normalize(X_pca[:, 1], y_lim)
        for label in classes:
            mask = (labels == label)
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        alpha=0.4, label="class " + str(int(label)))
        plt.title("Before {}".format(layer))
        plt.legend(loc=1)
        plt.savefig(os.path.join(gif_name + "_", '_pca_{}.png'.format(i)))

    create_gif([os.path.join(gif_name + "_", '_pca_{}.png'.format(i+1)) for i in range(13)],
               gif_name, duration, extend=True)


def run_pca_gif():
    flips = [
        [],  # 1
        [],  # 2
        [],  # 3
        [],  # 4
        [4],  # 5
        [4],  # 6
        [],  # 7
        [1],  # 8
        [1],  # 9
        [],  # 10
        [],  # 11
        [],  # 12
        [],  # 13
    ]
    rotates = np.array([
        0,  # 1
        0,  # 2
        0,  # 3
        0,  # 4
        -30,  # 5
        35,  # 6
        -45,  # 7
        -45,  # 8
        -42,  # 9
        6,  # 10
        8.5,  # 11
        6,  # 12
        0,  # 13
    ])
    rotates = rotates / 180 * np.pi
    pca_isolation_gif(flips=flips, rotates=rotates)


if __name__ == '__main__':
    run_pca_gif()
