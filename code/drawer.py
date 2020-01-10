import os
import imageio
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans

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


def plot_3d_cluster(data, labels, target, n_cluster):
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    X = data[labels == target, :]

    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_jobs=-1, verbose=1).fit(X)
    centers = pca.transform(kmeans.cluster_centers_)

    for label in np.unique(kmeans.labels_):
        mask = (kmeans.labels_ == label)
        plt.scatter(data_pca[mask, 0], data_pca[mask, 1], data_pca[mask, 2], label="cluster " + str(int(label)))
    plt.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='s', label='centers', c='b')


def plot_network_venn(data, subclasses, cluster_sets, ncol):
    count = lambda x: sum((data[:, 0] == x[0]) & (data[:, 1] == x[1]))
    per_col = np.ceil(subclasses.shape[0] / ncol)
    labels = subclasses[:, -1]
    subclasses = subclasses[:, :-1]
    stride = 15 * 0.5 / per_col
    markers = dict()
    cluster_markers = dict()

    G = nx.Graph()
    cl_pos = dict()
    cmap = ['b', 'g']
    for i, subclass in enumerate(subclasses):
        for j, cl in enumerate(subclass):
            plt.plot([1 + 1 * (i // per_col), j * 4], [stride * (i % per_col) - 0.5, cl * 1],
                     c=plt.cm.Blues(j * 0.4 + 0.4))
    for i, subclass in enumerate(subclasses):
        G.add_node(i, encoding=subclass, size=count(subclass) * 5 + 200, shape='skyblue',
                   pos=(1 + 1 * (i // per_col) - 0.2, stride * (i % per_col) - 0.5))
        markers[int(labels[i])], = plt.plot(1 + 1 * (i // per_col), stride * (i % per_col) - 0.5, marker='o', ms=5,
                                            c=plt.cm.winter(labels[i] / labels.max()))
        plt.plot(1 + 1 * (i // per_col), stride * (i % per_col) - 0.5, marker='o', ms=count(subclass) / 10 + 5,
                 c=plt.cm.winter(labels[i] / labels.max()))
    for i, clusters in enumerate(cluster_sets):
        for cl in clusters:
            G.add_node('{:.0f}-{:.0f}'.format(i, cl), encoding=None, size=500, shape='orangered', pos=(i * 4, cl * 1))
            cluster_markers[i], = plt.plot(i * 4, cl * 1, ms=5, marker='s', c=plt.cm.Blues(i * 0.3 + 0.3))
            plt.plot(i * 4, cl * 1, ms=25, marker='s', c=plt.cm.Blues(i * 0.3 + 0.3))

    nx.draw_networkx(G, node_size=0, pos=G.nodes(data='pos'))
    lines = list(markers.values()) + list(cluster_markers.values())
    labels = ['class {}'.format(label) for label in markers.keys()] + ['model {}'.format(label) for label in
                                                                       cluster_markers.keys()]
    plt.legend(lines, labels, loc=1, ncol=1, bbox_to_anchor=(0.9, 1))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()


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


def run_3d_cluster(data_path="data/data.txt", label_path="data/labels.txt"):
    print("Importing data...")
    data = np.loadtxt(data_path)
    print("Importing labels...")
    labels = np.loadtxt(label_path)
    plot_3d_cluster(data, labels, 4, 3)


def run_network_venn(encoding_path="data/encoding_merged.txt", threshold=6, ncol=3):
    data = np.loadtxt(encoding_path)
    subclasses, freq = np.unique(list(data[:, :]), axis=0, return_counts=True)
    tmp = subclasses[freq > threshold]
    tmp = np.array(sorted(list(tmp), key=lambda x: x[-1]))
    plot_network_venn(data, tmp, [np.unique(tmp[:, i]) for i in range(tmp.shape[1] - 1)], ncol)


if __name__ == '__main__':
    # run_pca_gif()
    run_network_venn()
    plt.show()
