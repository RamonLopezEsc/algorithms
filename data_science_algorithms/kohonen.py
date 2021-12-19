# -*- coding: utf-8 -*-

import random
from math import inf
import pandas as pd

random.seed(12)


def min_euclidean_distance(centroids, inputcoords, ndim=2):
    dist_coords = []
    for centindex, centroid in enumerate(centroids):
        distance = 0
        for index in range(ndim):
            distance = distance + (centroid[index] - inputcoords[index]) ** 2
        dist_coords.append([distance ** 0.5, centindex])
    return min(dist_coords)


def k_means(centroids, dataframe):
    error = []
    mean_centroids = [[0 for _ in range(len(centroids[0]) + 1)] for _ in range(len(centroids))]
    for index, row in dataframe.iterrows():
        min_dist_centr = min_euclidean_distance(centroids, [row['xcoord'], row['ycoord']], len(centroids[0]))
        mean_centroids[min_dist_centr[1]][0] += row['xcoord']
        mean_centroids[min_dist_centr[1]][1] += row['ycoord']
        mean_centroids[min_dist_centr[1]][2] += 1

    for centroid in mean_centroids:
        for coord in range(len(centroids[0])):
            centroid[coord] = centroid[coord] / centroid[-1]

    for index, centroid in enumerate(centroids):
        aux_error = 0
        for coord in range(len(centroids[0])):
            aux_error += (centroid[coord] - mean_centroids[index][coord]) ** 2
            centroid[coord] = mean_centroids[index][coord]
        error.append(aux_error ** 0.5)

    return max(error)


def init_centroids(nclusters, dataframe):
    centroid_coords = []
    min_x, max_x = min(dataframe['xcoord']), max(dataframe['xcoord'])
    min_y, max_y = min(dataframe['ycoord']), max(dataframe['ycoord'])
    for centroid in range(nclusters):
        xcoord, ycoord = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
        centroid_coords.append([xcoord, ycoord])
    return centroid_coords


err = inf
n_iter = 0
n_clusters = 4
n_max_iter = 1000
err_treshold = 0.001

mibici_df = pd.read_csv('csv/coords_mi_bici.txt')
centroid_clusters = init_centroids(n_clusters, mibici_df)

print('- ==================================================== -')
print('Coordenadas iniciales: ')
print(centroid_clusters)
print('- ==================================================== -')

while n_iter < n_max_iter:
    n_iter += 1
    if err < err_treshold:
        break
    err = k_means(centroid_clusters, mibici_df)
    print(f'Iter: {n_iter}')
    print(f'Coordenadas: {centroid_clusters}')
    print(f'Error: {err}\n')

print('- ==================================================== -')
print('Coordenadas Finales: ')
print(centroid_clusters)

coords_ramon_casa = [664368.2929025213, 2286539.520576209]
min_dist = min_euclidean_distance(centroid_clusters, coords_ramon_casa, len(centroid_clusters[0]))

print('- ==================================================== -')
print('Coordenada Ejemplo - Dato de entrada (casa Ramon - UTM Zona 13 Norte): ')
print(f'xcoord = {coords_ramon_casa[0]}')
print(f'ycoord = {coords_ramon_casa[1]}')
print(f'Cluster mas cercano: {min_dist[1]}')
print(f'Distancia en metros: {min_dist[0]}')
print('- ==================================================== -')
