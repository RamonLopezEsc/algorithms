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

def gaussian_evaluation(data, covariance, mean):
    data_mean = data - mean
    eval_part_1 = 1 / ((((2 * np.pi) ** (len(data))) * np.linalg.det(covariance)) ** (0.5))
    eval_part_2 = np.dot(data_mean.T, np.linalg.inv(covariance))
    eval_part_3 = np.dot(eval_part_2, data_mean)
    value = eval_part_1 * np.exp(-0.5 * eval_part_3)
    return value

def init_values(means, covariance, mixing_param):
    for i in range(num_clusters):
        temporal_mean = [0.0, 0.0]
        for j in range(num_data):
            temporal_data = all_data[np.random.randint(0, len_data)]
            temporal_mean[0] = temporal_data[0] + temporal_mean[0]
            temporal_mean[1] = temporal_data[1] + temporal_mean[1]
        temporal_mean[0] = temporal_mean[0]/num_data
        temporal_mean[1] = temporal_mean[1]/num_data
        means.append(np.asarray(temporal_mean))
        covariance.append(aux_covariance)
        mixing_param.append(1.0/num_clusters)

def expectation(probab_array):
    for i in range(len_data):
        aux_probab_eval = [ ]
        aux_gaussian_eval = [ ]
        summation_evaluations = 0.0
        for j in range(num_clusters):
            evaluation = (gaussian_evaluation(all_data[i], covariance[j], means[j])) * mixing_param[j]
            aux_gaussian_eval.append(evaluation)
            summation_evaluations = summation_evaluations + evaluation
        for j in range(num_clusters):
            probab_evaluation = aux_gaussian_eval[j] / summation_evaluations
            aux_probab_eval.append(probab_evaluation)
        probab_array.append(aux_probab_eval)

def maximization():
    copy_mean = np.copy(means)
    copy_cov  = np.copy(covariance)
    for i in range(num_clusters):
        mixing_param[i] = 0
        means[i] = np.array([0.0, 0.0])
        covariance[i] = np.array([[0, 0], [0, 0]])

    for i in range(len_data):
        for j in range(num_clusters):
            mixing_param[j] = mixing_param[j] + (probab_array[i][j] / len_data)

    for i in range(len_data):
        for j in range(num_clusters):
            aux = (probab_array[i][j] * all_data[i])/(mixing_param[j] * len_data)
            means[j] = means[j] + aux

    for i in range(len_data):
        for j in range(num_clusters):
            aux = np.array([all_data[i] - means[j]])
            aux_t = np.transpose(aux)
            multiply = np.dot(aux_t, aux) * probab_array[i][j]
            multiply = multiply / (mixing_param[j] * len_data)
            covariance[j] = covariance[j] + multiply

    sum_mean_prev = 0.0
    sum_mean_act  = 0.0
    sum_cova_prev = 0.0
    sum_cova_act  = 0.0

    for j in range(num_clusters):
        sum_cova_prev = sum_cova_prev + np.linalg.det(copy_cov[j])
        sum_cova_act  = sum_cova_act  + np.linalg.det(copy_cov[j])
        for k in range(len(means[j])):
            sum_mean_prev = sum_mean_prev + means[j][k]
            sum_mean_act  = sum_mean_act + copy_mean[j][k]

    return [abs(sum_mean_prev - sum_mean_act), abs(sum_cova_prev - sum_cova_act)]
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #
num_data = 50
num_clusters = 3
precition = 0.000001
data_gauss_1 = generate_random_points_2D(num_data, [1, 1], [[1, 0.4], [0.4, 1]])
data_gauss_2 = generate_random_points_2D(num_data, [2, 2], [[1, -0.6], [-0.6, 1]])
data_gauss_3 = generate_random_points_2D(num_data, [3, 1], [[1, 0], [0, 1]])

means = [ ]
covariance = [ ]
mixing_param = [ ]
gaussian_evaluations = [ ]
probab_array = [ ]

all_data = np.asarray(data_gauss_1 + data_gauss_2 + data_gauss_3)
len_data = len(all_data)
aux_covariance = np.cov(all_data.T)

init_values(means, covariance, mixing_param)
means = np.asarray(means)

for p in range(1000):
    probab_array = [ ]
    expectation(probab_array)
    value = maximization()
    if value[0] <= precition and value[1] <= precition:
        break

print 'Medias\n', means, '\n'
print 'Covarianza\n', covariance, '\n'
print 'Prob. a priori\n', mixing_param
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #