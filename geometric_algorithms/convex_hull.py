#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------LIBRERIAS-------------------------- #
# ------------------------------------------------------------- #
import random as rand
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------------------- #
# -------------------------FUNCIONES--------------------------- #
def turn_right():
    array = [coord_points[0], coord_points[1]]
    for i in range(2, len(coord_points)):
        array.append(coord_points[i])
        while len(array) > 2 and np.linalg.det([array[-3], array[-2], array[-1]]) > 0:
            array.pop(-2)
    return array

def convex_hull():
    coord_points.sort()
    l_upper = turn_right()
    coord_points.reverse()
    l_lower = turn_right()
    l = l_upper + l_lower
    return l

def graph(convex_pol, coord_points):
    # Acomodando listas adecuadas para graicar en matplot
    x_points = [i[0] for i in coord_points]
    y_points = [i[1] for i in coord_points]
    x_polygon = [i[0] for i in convex_pol]
    y_polygon = [i[1] for i in convex_pol]
    # Definiendo límites extremos de la gráfica
    x_lim_der = max(x_points) + 5
    y_lim_sup = max(y_points) + 5
    x_lim_izq = min(x_points) - 5
    y_lim_inf = min(y_points) - 5
    # Asignación de los límites extremos
    plt.xlim(x_lim_izq, x_lim_der)
    plt.ylim(y_lim_inf, y_lim_sup)
    # Graficación
    plt.title('Problema: Convex Hull')
    plt.xlabel('Eje de las abscisas')
    plt.ylabel('Eje de las ordenadas')
    plt.plot(x_points, y_points, 'ko')
    plt.plot(x_polygon, y_polygon, 'r-', linewidth = 2.0)
    plt.show()
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #
# Generación de coordenadas de manera aleatoria (números reales)
num_points = 100
coord_points = []
for i in range(num_points): coord_points.append([rand.uniform(0, 50), rand.uniform(0, 100), 1.0])
# Creación y graficación del polígono convexo
convex_pol = convex_hull()
graph(convex_pol, coord_points)
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #