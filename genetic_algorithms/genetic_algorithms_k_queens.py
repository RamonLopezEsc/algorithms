#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------LIBRERIAS-------------------------- #
# ------------------------------------------------------------- #
import time
import numpy as np
from itertools import izip_longest, imap
import matplotlib.pyplot as plt
import random as rand
# ------------------------------------------------------------- #
# -------------------------FUNCIONES--------------------------- #
def initial():
    pobl_parents = []
    perm_array = range(num_queens)
    for i in range(tam_pobl):
        chromosome = list(np.random.permutation(perm_array))
        fitness = evalua(chromosome)
        pobl_parents.append([fitness, chromosome])
    prom_fit, desv_std = prom_fitness(pobl_parents)
    fit_prom.append(prom_fit)
    fit_desv_std.append(desv_std)
    return pobl_parents

def evalua(chromosome):
    global counter_eval
    counter_eval += 1
    aux_fitness = 0
    for i in range(len(chromosome)):
        for j in range(i + 1, num_queens):
            if abs(i - j) == abs(chromosome[i] - chromosome[j]):
                aux_fitness += 1
    return aux_fitness

def select_parents():
    best_parents = []
    for i in range(num_parents):
        aux_index = rand.randint(0, tam_pobl - 1)
        best_parents.append(pobl_parents[aux_index])
    best_parents.sort()
    return [best_parents[0], best_parents[1]]

def swap_recomb(cross_point, arr_x, arr_y):
    active_index = cross_point
    swap_index = cross_point + 1
    for i in range(num_queens):
        if swap_index == num_queens: break
        active_index += 1
        aux_bool = True
        if active_index == num_queens: active_index = 0
        for j in range(0, cross_point + 1):
            if arr_y[1][active_index] == arr_x[1][j]: aux_bool = False
        if aux_bool == True:
            arr_x[1][swap_index] = arr_y[1][active_index]
            swap_index += 1

def recomb():
    cross_point = rand.randint(1, num_queens - 1)
    offspring = []
    x = list(best_parents[0][1])
    x = [best_parents[0][0], x]
    y = list(best_parents[1][1])
    y = [best_parents[1][0], y]
    aux_x = list(x[1])
    aux_x = [x[0], aux_x]
    swap_recomb(cross_point, x, y)
    swap_recomb(cross_point, y, aux_x)
    offspring.append(x)
    offspring.append(y)
    return offspring

def mutation():
    for i in range(num_offspring):
        rand_prob = rand.random()
        var_bool = False
        if rand_prob < prob_muta:
            var_bool = True
        if var_bool == True:
            rand_index1 = rand.randint(0, num_queens - 1)
            rand_index2 = rand.randint(0, num_queens - 1)
            aux_var = offspring[i][1][rand_index1]
            offspring[i][1][rand_index1] = offspring[i][1][rand_index2]
            offspring[i][1][rand_index2] = aux_var

def selection():
    for i in range(num_offspring):
        fitness = evalua(offspring[i][1])
        offspring[i][0] = fitness
    pobl_child.append(offspring[0])
    pobl_child.append(offspring[1])
    if len(pobl_child) == tam_pobl:
        global counter_gener
        counter_gener += 1
        aux_solution = []
        aux_parents = pobl_parents + pobl_child
        aux_parents.sort()
        for i in range(tam_pobl):
            aux_solution.append(aux_parents[i])
        aux_parents = []
        prom_fit, desv_std = prom_fitness(pobl_parents)
        fit_prom.append(prom_fit)
        fit_desv_std.append(desv_std)
        return aux_solution, aux_parents
    else:
        return pobl_parents, pobl_child

def prom_fitness(array_pobl):
    desv_std = 0.0
    prom = 0.0
    for i in range(tam_pobl):
        prom = prom + array_pobl[i][0]
    prom = prom/tam_pobl
    for i in range(tam_pobl):
        desv_std = desv_std + ((array_pobl[i][0] - prom) ** 2)
    desv_std = (desv_std / (tam_pobl - 1)) ** 0.5
    return prom, desv_std

def prom_time(array_time):
    desv_std = 0.0
    prom = 0.0
    for i in range(len(array_time)):
        prom = prom + array_time[i]
    prom = prom/len(array_time)
    for i in range(len(array_time)):
        desv_std = desv_std + ((array_time[i] - prom) ** 2)
    desv_std = (desv_std / (len(array_time) - 1)) ** 0.5
    return prom, desv_std

def prom_gener(x):
    y = [i for i in x if i is not None]
    return sum(y, 0.0) / len(y)

def print_solution(counter):
    if counter_eval == num_eval:
        end_time = time.time() - start_time
        time_exec.append(end_time)
        print 'Intento #', counter + 1
        print 'Solución no encontrada'
        print "Tiempo de ejecución: ", end_time
        print
    else:
        solution = pobl_parents[0][1]
        end_time = time.time() - start_time
        time_exec.append(end_time)
        print 'Intento #', counter + 1
        print "¡Solución encontrada!"
        print "Número de generaciones: ", counter_gener
        print "Configuración del tablero: ", solution
        print_chessboard(solution)
        print "\n\nTiempo de ejecución: ", end_time
        print

def print_chessboard(solution):
    for i in range(num_queens):
        print
        for j in range(num_queens):
            if j == solution[i]:
                print "[x]",
            else:
                print "[ ]",

def graph_plots():
    x_axis = np.arange(0, len(mas_std), 1)
    x_lim_der = max(arr_gen) - 1
    y_lim_sup = max(mas_std) + 1
    x_lim_izq = 1
    y_lim_inf = 0

    plt_1 = plt.plot(x_axis, all_fitprom_1, 'k-')
    plt_2 = plt.plot(x_axis, mas_std, 'r--')
    plt_3 = plt.plot(x_axis, menos_std, 'b--')

    leg_1 = plt.legend(plt_1, ['Fitness promedio'], loc = 1)
    leg_2 = plt.legend(plt_2, ['$+ \sigma$'], loc = 3)
    leg_3 = plt.legend(plt_3, ['$- \sigma$'], loc = 4)

    plt.gca().add_artist(leg_1)
    plt.gca().add_artist(leg_2)
    plt.gca().add_artist(leg_3)

    plt.xlim(x_lim_izq, x_lim_der)
    plt.ylim(y_lim_inf, y_lim_sup)
    plt.title('Fitness promedio por generacion (128 reinas)')
    plt.xlabel('Generaciones')
    plt.ylabel('Fitness')

    plt.show()
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #
# Parámetros del algoritmo
num_eval = 10000
num_tries = 30
tam_pobl = 100
num_queens = 16
num_parents = 5
num_offspring = 2
prob_muta = 0.8
# Parámetros de inicio
arr_gen = []
time_exec = []
all_fitprom = []
all_fitstd = []
# Ciclo de intentos
for i in range(num_tries):
    # Contabilizador de tiempo
    start_time = time.time()
    # Parámetros de inicio
    pobl_child = []
    fit_prom = []
    fit_desv_std = []
    counter_eval = 0
    counter_gener = 1
    # Inicialización
    pobl_parents = initial()
    # Ciclo del algoritmo
    while counter_eval != num_eval:
        # Condición de paro si se encuentra una solución
        if pobl_parents[0][0] == 0:
            print_solution(i)
            break
        else:
            # Selección de padres
            best_parents = select_parents()
            # Recombinación
            offspring = recomb()
            # Mutación
            mutation()
            # Selección
            pobl_parents, pobl_child = selection()
    # Almacenamiento de promedios y desviaciones std
    all_fitprom.append(fit_prom)
    all_fitstd.append(fit_desv_std)
    # Impresión si es que no se encuentra una solución
    if counter_eval == num_eval: print_solution(i)
    arr_gen.append(counter_gener)

# Cálculos finales
mas_std = []
menos_std = []
time_prom, time_desv_std = prom_time(time_exec)
all_fitprom_1 = list(imap(prom_gener, izip_longest(*all_fitprom)))
all_fitstd_1 = list(imap(prom_gener, izip_longest(*all_fitstd )))

for i in range(len(all_fitprom_1)):
    mas_std.append(all_fitprom_1[i] + all_fitstd_1[i])
    menos_std.append(all_fitprom_1[i] - all_fitstd_1[i])

# Impresión de tiempos promedio
print 'Tiempo promedio:', time_prom
print 'Desviación estándar del tiempo:', time_desv_std
# Gráfica de resultados
graph_plots()
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #
