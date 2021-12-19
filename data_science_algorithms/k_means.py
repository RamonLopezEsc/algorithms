#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------LIBRERIAS-------------------------- #
# ------------------------------------------------------------- #
import numpy as np
# ------------------------------------------------------------- #
# -------------------------FUNCIONES--------------------------- #
def generate_random_points_2D(num_puntos, media, mat_cov):
    aux_arr = []
    for i in range(num_puntos):
        x, y = np.random.multivariate_normal(media, mat_cov, 1).T
        aux_arr.append(np.array([float(x), float(y)]))
    return aux_arr

def init_means():
    means = [ ]
    for i in range(num_clusters):
        temporal_mean = [0.0, 0.0]
        for j in range(num_data):
            temporal_data = all_data[np.random.randint(0, len_data)]
            temporal_mean[0] = temporal_data[0] + temporal_mean[0]
            temporal_mean[1] = temporal_data[1] + temporal_mean[1]
        temporal_mean[0] = temporal_mean[0]/num_data
        temporal_mean[1] = temporal_mean[1]/num_data
        means.append(temporal_mean)
    return means

def zero_means():
    means = [ ]
    for i in range(num_clusters):
        means.append([0.0, 0.0])
    return means

def euclidean_distance(elemento1, elemento2):
    distance = 0
    for i in range(len(elemento1)):
        distance = distance + (elemento1[i] - elemento2[i]) ** 2
    return distance ** 0.5

def k_means(actual_means):
    clusters = [ ]
    copy_means = actual_means[:]
    for i in range(num_clusters):
        clusters.append([])
    for i in range(len_data):
        aux_distance = [euclidean_distance(all_data[i], actual_means[0]), 0]
        for j in range(num_clusters):
            distance = euclidean_distance(all_data[i], actual_means[j])
            if distance < aux_distance[0]:
                aux_distance = [distance, j]
        clusters[aux_distance[1]].append(all_data[i])
    means = zero_means()
    for i in range(num_clusters):
        len_cluster = len(clusters[i])
        for j in range(len_cluster):
            for k in range(len(clusters[i][j])):
                means[i][k] = means[i][k] + (clusters[i][j][k]/len_cluster)
    return means, copy_means
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #
num_data = 100
num_clusters = 4
data_gauss_1 = generate_random_points_2D(num_data, [2, 2], [[0.5, 0], [0, 0.5]])
data_gauss_2 = generate_random_points_2D(num_data, [6, 6], [[0.5, 0], [0, 0.5]])
data_gauss_3 = generate_random_points_2D(num_data, [10, 2], [[0.5, 0], [0, 0.5]])

all_data = np.asarray(data_gauss_1 + data_gauss_2 + data_gauss_3)
len_data = len(all_data)
means = init_means()

for p in range(1000):
    means, copy = k_means(means)
    if copy == means:
        break

print means


