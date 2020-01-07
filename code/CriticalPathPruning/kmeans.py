# coding: utf-8
import math
import random
import os
import json
from collections import defaultdict
import numpy as np


def distance(point1, point2):
    """
    计算两点之间的欧拉距离，支持多维
    """
    # distance = 0.0
    # for a, b in zip(point1, point2):
    #     distance += math.pow(a - b, 2)
    # return math.sqrt(distance)
    # return np.sqrt(np.sum((point1 - point2) ** 2))
    # return np.sqrt(np.sum((point1 - point2) ** 2))
    return np.linalg.norm(point1-point2,ord=1)


def get_closest_dist(point, centroids):
    min_dist = math.inf  # 初始设为无穷大
    for i, centroid in enumerate(centroids):
        dist = distance(centroid, point)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def kpp_centers(data_set: list, k: int) -> list:
    """
    从数据集中返回 k 个对象可作为质心
    """
    cluster_centers = []
    cluster_centers.append(random.choice(data_set))
    d = [0 for _ in range(len(data_set))]
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data_set):
            d[i] = get_closest_dist(point, cluster_centers) # 与最近一个聚类中心的距离
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d): # 轮盘法选出下一个聚类中心；
            total -= di
            if total > 0:
                continue
            cluster_centers.append(data_set[i])
            break
    return cluster_centers


def point_avg(points):
    '''
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2
    Returns a new points which is the center of all the points
    :param points:
    :return:
    '''
    dimensions = len(points[0])

    new_center = []
    points = np.array(points)
    # for dimension in range(dimensions):
    #     dim_sum = 0
    #     for p in points:
    #         dim_sum += p[dimension]
    #
    #     # average of each dimension
    #     new_center.append(dim_sum / float(len(points)))

    return np.mean(points,axis=0)[0]


def update_centers(date_set, assignments):
    '''
    Accepts a dataset and a list of assignments; the indexes of both lists correspond
    to each other.
    compute the center for each of the assigned groups.
    Reture 'k' centers where  is the number of unique assignments.
    :param date_set:
    :param assignments:
    :return:
    '''
    new_means = defaultdict(list)
    centers = []
    for assigment, point in zip(assignments, date_set):
        new_means[assigment].append(point)

    for points in new_means.values():
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
    '''
    Given a data set and a list  of points between other points,
    assign each point to an index that corresponds to the index
    of the center point on its proximity to that point.
    Return a an array of indexes of the centers that correspond to
    an index in the data set; that is, if there are N points in data set
    the list we return will have N elements. Also If there ara Y points in centers
    there will be Y unique possible values within the returned list.
    :param data_points:
    :param centers:
    :return:
    '''
    assigments = []
    for point in data_points:
        shortest = float('Inf')
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assigments.append(shortest_index)

    return assigments


def generate_k(data_set, k):
    '''
    Given data set , which is an array of arrays,
    find the minimum and maximum foe each coordinate, a range.
    Generate k random points between the ranges
    Return an array of the random points within the ranges
    :param data_set:
    :param k:
    :return:
    '''
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]

            rand_point.append(random.uniform(min_val, max_val))

        centers.append(rand_point)
    return centers


def k_means(dataset, k):
    k_points = kpp_centers(dataset, k)
    # k_points = generate_k(dataset,k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    times = 0
    while assignments != old_assignments:
        print(times)
        times += 1
        print('times is :', times)
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)

    return (assignments, dataset)


def load_data():
    root_path = "ImageEncoding"
    cps = []
    classids = {}
    count = 0
    for root, dirs, files in os.walk(root_path):
        for file in files:
            print(file,count)
            with open(os.path.join(root,file),"r") as f:
                res = json.load(f)
            cp = []
            res.sort(key=lambda item: item["layer_name"])
            for layer in res:
                cp += layer["layer_lambda"]
            cps.append(np.array(cp))
            id_str = file.split("-")[0]
            id_str = int(id_str[5:])
            classids[count] = id_str
            count += 1
            # if count>100:
            #     return cps, classids

    return cps, classids


if __name__ == "__main__":
    data, classids = load_data()
    k = 35
    class_res = [{} for i in range(k)]
    assignment = k_means(data, k)[0]
    for i in range(len(assignment)):
        classid = classids[i]
        if classid in class_res[assignment[i]]:
            class_res[assignment[i]][classid] += 1
        else:
            class_res[assignment[i]][classid] = 1
    print(class_res)