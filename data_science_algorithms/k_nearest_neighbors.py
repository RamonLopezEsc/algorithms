#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------LIBRERIAS-------------------------- #
# ------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------------------- #
# ---------------------------CLASES---------------------------- #
class KNearest:
    def __init__(self, num_datos, knear):
        self.num_data = num_datos
        self.k = knear
        self.datos_clase_0 = self.generate_random_points_2D(0, self.num_data, [ 0, 0], [[1, 0], [0, 1]])
        self.datos_clase_1 = self.generate_random_points_2D(1, self.num_data, [ 5, 0], [[1, 0], [0, 1]])
        self.test_data =   self.generate_random_points_2D(0, self.num_data/2, [ 0, 0], [[1, 0], [0, 1]]) + \
                           self.generate_random_points_2D(1, self.num_data/2, [ 5, 0], [[1, 0], [0, 1]])
        self.resultado = [ ]
        self.error = [0, 0]

    def generate_random_points_2D(self, clase, num_puntos, media, mat_cov):
        aux_arr = []
        for i in range(num_puntos):
            x, y = np.random.multivariate_normal(media, mat_cov, 1).T
            aux_arr.append([clase, np.array([1.0, float(x), float(y)])])
        return aux_arr

    def generate_empty_k_array(self):
        k_arreglo = [ ]
        for i in range(self.k): k_arreglo.append([0, -1])
        return k_arreglo

    def distancia_euclideana(self, elemento1, elemento2):
        distancia = 0
        for i in range(len(elemento1[1])):
            distancia = distancia + (elemento1[1][i] - elemento2[1][i]) ** 2
        return distancia ** 0.5

    def KNearest(self):
        for i in range(len(self.test_data)):
            k_vecinos = [ ]
            for j in range(self.num_data):
                k_vecinos.append([0, self.distancia_euclideana(self.test_data[i], self.datos_clase_0[j])])
                k_vecinos.append([1, self.distancia_euclideana(self.test_data[i], self.datos_clase_1[j])])
            k_vecinos.sort(key = lambda x:x[1])
            k_vecinos = k_vecinos[0:self.k]
            contador = [0, 0]
            for j in k_vecinos:
                if j[0] == 0: contador[0] = contador[0] + 1
                else: contador[1] = contador[1] + 1
            if contador[0] > contador[1]:
                if self.test_data[i][0] == 1: self.error[1] += 1
                self.resultado.append([0, self.test_data[i][1]])
            else:
                if self.test_data[i][0] == 0: self.error[0] += 1
                self.resultado.append([1, self.test_data[i][1]])
        print 'Error de clasificación'
        print 'Errores en la clase 1: ', self.error[0]
        print 'Errores en la clase 2: ', self.error[1]

    def graficar(self):
        lista_test_class0_X = [ ]
        lista_test_class0_Y = [ ]
        lista_test_class1_X = [ ]
        lista_test_class1_Y = [ ]
        lista_resu_class0_X = [ ]
        lista_resu_class0_Y = [ ]
        lista_resu_class1_X = [ ]
        lista_resu_class1_Y = [ ]

        for i in range(len(self.test_data)):
            if self.test_data[i][0] == 0:
                lista_test_class0_X.append(self.test_data[i][1][1])
                lista_test_class0_Y.append(self.test_data[i][1][2])
            else:
                lista_test_class1_X.append(self.test_data[i][1][1])
                lista_test_class1_Y.append(self.test_data[i][1][2])
            if self.resultado[i][0] == 0:
                lista_resu_class0_X.append(self.resultado[i][1][1])
                lista_resu_class0_Y.append(self.resultado[i][1][2])
            else:
                lista_resu_class1_X.append(self.resultado[i][1][1])
                lista_resu_class1_Y.append(self.resultado[i][1][2])

        f, (ax1, ax2) = plt.subplots(2, sharex = True)
        ax1.plot(lista_test_class0_X, lista_test_class0_Y, 'co', lista_test_class1_X, lista_test_class1_Y, 'mo')
        ax2.plot(lista_resu_class0_X, lista_resu_class0_Y, 'co', lista_resu_class1_X, lista_resu_class1_Y, 'mo')
        plt.show()

# ------------------------------------------------------------- #
# ------------------------------------------------------------- #
k_nearest = KNearest(10, 5)
k_nearest.KNearest()
k_nearest.graficar()