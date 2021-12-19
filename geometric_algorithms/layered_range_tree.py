#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------LIBRERIAS-------------------------- #
# ------------------------------------------------------------- #
import copy
import random as rand
import matplotlib.pyplot as plt
# ------------------------------------------------------------- #
# ---------------------------CLASES---------------------------- #
class Estructura2D:
    def __init__(self, valor):
        self.puntero_izq = [0, None]
        self.puntero_der = [0, None]
        self.valor = valor

    def __getitem__(self, key):
        return self.valor[key]

class Node:
    def __init__(self, val):
        self.arreglo_punteros = []
        self.izquierda = None
        self.derecha = None
        self.valor = val

class LayeredRangeTree:
    def __init__(self):
        self.raiz = None
        self.nodos_negrosx = []
        self.nodos_negros = []

    def tree_maximum(self, node):
        if node.derecha is not None:
            return self.tree_maximum(node.derecha)
        else:
            return node

    def build_from_list(self,list):
        sorted_list = sorted(list)
        self.raiz = self.divide_list(sorted_list)
        self.internalnode_2d_tree(self.raiz, self.raiz)
        self.build_pointers(self.raiz)

    def divide_list(self, sorted_list):
        if len(sorted_list) <= 1:
            new_node = Node(sorted_list[0])
            return new_node
        else:
            left = self.divide_list(sorted_list[0 : int(len(sorted_list) / 2)])
            right = self.divide_list(sorted_list[int(len(sorted_list) / 2) : len(sorted_list)])
            new_node = Node(self.tree_maximum(left).valor)
            new_node.izquierda = left
            new_node.derecha = right
            return new_node

    def internalnode_2d_tree(self, nodohijo, nodopadre):
        if nodohijo.izquierda != None:
            self.internalnode_2d_tree(nodohijo.izquierda, nodohijo)
            for i in nodohijo.izquierda.arreglo_punteros:
                nodohijo.arreglo_punteros.append(copy.copy(i))
            self.internalnode_2d_tree(nodohijo.derecha, nodohijo)
            for i in nodohijo.derecha.arreglo_punteros:
                nodohijo.arreglo_punteros.append(copy.copy(i))
            nodohijo.arreglo_punteros.sort(key = lambda x:x[1])
        else:
            estruc_puntero = Estructura2D(nodohijo.valor)
            nodohijo.arreglo_punteros.append(estruc_puntero)

    def rangeQuery_2D(self, intervalx, intervaly):
        split_node, puntero = self.find_split_node(intervalx[0], intervalx[1], intervaly[0])
        self.LeftSeachQuery(split_node, puntero, intervalx[0], intervalx[1], intervaly[1], intervaly[0])
        self.RightSearchQuery(split_node, puntero, intervalx[0], intervalx[1], intervaly[1], intervaly[0])

    def find_split_node(self, interval_1, interval_2, interval_3):
        aux = self.raiz
        while aux.izquierda != None:
            if interval_2 >= aux.valor[0] and interval_1 <= aux.valor[0]:
                break
            else:
                if interval_2 <= aux.valor[0]:
                    aux = aux.izquierda
                else:
                    aux = aux.derecha
        puntero = binary_search(interval_3, aux.arreglo_punteros)
        return aux, puntero

    def LeftSeachQuery(self, split_node, punt, interval_1, interval_2, interval_3, interval_4):
        if split_node.izquierda == None:
            if split_node.valor[0] >= interval_1 and split_node.valor[0] <= interval_2:
                if split_node.valor[1] <= interval_3:
                    self.nodos_negros.append(split_node.valor)
        else:
            left_search_node = split_node.izquierda
            puntero = split_node.arreglo_punteros[punt].puntero_izq[0]
            while left_search_node.izquierda != None:
                if puntero == None: break
                if interval_1 <= left_search_node.valor[0]:
                    nodo_report = left_search_node.derecha
                    if left_search_node.arreglo_punteros[puntero].puntero_der[0] != None:
                        puntero_report = left_search_node.arreglo_punteros[puntero].puntero_der[0]
                        for i in range(puntero_report, len(nodo_report.arreglo_punteros)):
                            if nodo_report.arreglo_punteros[i].valor[1] <= interval_3:
                                self.nodos_negros.append(nodo_report.arreglo_punteros[i].valor)
                    puntero = left_search_node.arreglo_punteros[puntero].puntero_izq[0]
                    left_search_node = left_search_node.izquierda
                else:
                    if puntero == None: break
                    puntero = left_search_node.arreglo_punteros[puntero].puntero_der[0]
                    left_search_node = left_search_node.derecha
            if left_search_node.valor[0] >= interval_1 and left_search_node.valor[0] <= interval_2:
                if left_search_node.valor[1] <= interval_3 and left_search_node.valor[1] >= interval_4:
                    self.nodos_negros.append(left_search_node.valor)

    def RightSearchQuery(self, split_node, punt, interval_1, interval_2, interval_3, interval_4):
        if split_node.izquierda == None:
            if split_node.valor[0] <= interval_1 and split_node.valor[0] >= interval_2:
                if split_node.valor[1] <= interval_3:
                    self.nodos_negros.append(split_node.valor)
        else:
            right_search_node = split_node.derecha
            puntero = split_node.arreglo_punteros[punt].puntero_der[0]
            while right_search_node.izquierda != None:
                if puntero == None: break
                if interval_2 >= right_search_node.valor[0]:
                    nodo_report = right_search_node.izquierda
                    if right_search_node.arreglo_punteros[puntero].puntero_izq[0] != None:
                        puntero_report = right_search_node.arreglo_punteros[puntero].puntero_izq[0]
                        for i in range(puntero_report, len(nodo_report.arreglo_punteros)):
                            if nodo_report.arreglo_punteros[i].valor[1] <= interval_3:
                                self.nodos_negros.append(nodo_report.arreglo_punteros[i].valor)
                    puntero = right_search_node.arreglo_punteros[puntero].puntero_der[0]
                    right_search_node = right_search_node.derecha
                    #if puntero == None: break
                else:
                    if puntero == None: break
                    puntero = right_search_node.arreglo_punteros[puntero].puntero_izq[0]
                    right_search_node = right_search_node.izquierda
            if right_search_node.valor[0] >= interval_1 and right_search_node.valor[0] <= interval_2:
                if right_search_node.valor[1] <= interval_3:
                    if right_search_node.valor[1] <= interval_3 and right_search_node.valor[1] >= interval_4:
                        self.nodos_negros.append(right_search_node.valor)

    def build_pointers(self, nodointerno):
        if nodointerno.izquierda != None:
            for i in nodointerno.arreglo_punteros:
                counter_left = -1
                counter_right = -1
                self.look_left_pointer(i, nodointerno, counter_left)
                self.look_right_pointer(i, nodointerno, counter_right)
            self.build_pointers(nodointerno.izquierda)
            self.build_pointers(nodointerno.derecha)

    def look_left_pointer(self, elem_array, nodo, counter):
        for j in nodo.izquierda.arreglo_punteros:
            counter += 1
            if elem_array.valor[1] == j.valor[1]:
                elem_array.puntero_izq = [counter, j]
                break
            elif j.valor[1] > elem_array.valor[1]:
                elem_array.puntero_izq = [counter, j]
                break
            else:
                elem_array.puntero_izq = [None, None]

    def look_right_pointer(self, elem_array, nodo, counter):
        for j in nodo.derecha.arreglo_punteros:
            counter += 1
            if elem_array.valor[1] == j.valor[1]:
                elem_array.puntero_der = [counter, j]
                break
            elif j.valor[1] > elem_array.valor[1]:
                elem_array.puntero_der = [counter, j]
                break
            else:
                elem_array.puntero_der = [None, None]

    def stringify(self):
        if self is None:
            return ''
        return '\n' + '\n'.join(self.build_str(self.raiz)[0])

    def build_str(self, node):
        if node is None: return [], 0, 0, 0
        line1 = []
        line2 = []
        new_root_width = gap_size = len(str(node.valor))
        l_box, l_box_width, l_root_start, l_root_end = self.build_str(node.izquierda)
        r_box, r_box_width, r_root_start, r_root_end = self.build_str(node.derecha)
        if l_box_width > 0:
            l_root = -int(-(l_root_start + l_root_end) / 2) + 1  # ceiling
            line1.append(' ' * (l_root + 1))
            line1.append('_' * (l_box_width - l_root))
            line2.append(' ' * l_root + '/')
            line2.append(' ' * (l_box_width - l_root))
            new_root_start = l_box_width + 1
            gap_size += 1
        else: new_root_start = 0
        line1.append(str(node.valor))
        line2.append(' ' * new_root_width)
        if r_box_width > 0:
            r_root = int((r_root_start + r_root_end) / 2)
            line1.append('_' * r_root)
            line1.append(' ' * (r_box_width - r_root + 1))
            line2.append(' ' * r_root + '\\')
            line2.append(' ' * (r_box_width - r_root))
            gap_size += 1
        new_root_end = new_root_start + new_root_width - 1
        gap = ' ' * gap_size
        new_box = [''.join(line1), ''.join(line2)]
        for i in range(max(len(l_box), len(r_box))):
            l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
            r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
            new_box.append(l_line + gap + r_line)
        return new_box, len(new_box[0]), new_root_start, new_root_end

    def PrintTree(self):
        print(self.stringify())
# ------------------------------------------------------------- #
# -------------------------FUNCIONES--------------------------- #
def binary_search(num_buscar, sorted_array):
    limite_inferior = 0
    limite_superior = len(sorted_array) - 1
    #print sorted_array[limite_superior]
    if sorted_array[limite_superior].valor[1] < num_buscar:
        return
    while True:
        if limite_superior < limite_inferior:
            return limite_inferior
        bisect = (limite_inferior + limite_superior) / 2
        if sorted_array[bisect].valor[1] < num_buscar:
            limite_inferior = bisect + 1
        elif sorted_array[bisect].valor[1] > num_buscar:
            limite_superior = bisect - 1
        elif sorted_array[bisect].valor[1] == num_buscar:
            return bisect

def Graph(lista, resultado, intervalox, intervaloy):
    lista_x = [i[0] for i in lista]
    lista_y = [i[1] for i in lista]
    resultado_x = [i[0] for i in resultado]
    resultado_y = [i[1] for i in resultado]
    intervalo_X = [intervalox[0], intervalox[0], intervalox[1], intervalox[1], intervalox[0]]
    intervalo_y = [intervaloy[0], intervaloy[1], intervaloy[1], intervaloy[0], intervaloy[0]]

    if max(intervalo_X) > max(lista_x): x_lim_der = max(intervalo_X) + 0.5
    else: x_lim_der = max(lista_x) + 0.5
    if min(intervalo_X) < min(lista_x): x_lim_izq = min(intervalo_X) - 0.5
    else: x_lim_izq = min(lista_x) - 0.5
    if max(intervalo_y) > max(lista_y): y_lim_sup = max(intervalo_y) + 0.5
    else: y_lim_sup = max(lista_y) + 0.5
    if min(intervalo_y) < min(lista_y): y_lim_inf = min(intervalo_y) - 0.5
    else: y_lim_inf = min(lista_y) - 0.5

    plt.xlim(x_lim_izq, x_lim_der)
    plt.ylim(y_lim_inf, y_lim_sup)

    plt_1 = plt.plot(lista_x, lista_y, 'ko')
    plt_2 = plt.plot(resultado_x, resultado_y, 'ro')
    plt_3 = plt.plot(intervalo_X, intervalo_y, 'r--')

    plt.title('Problema: Busqueda de Rango 2D')
    leg_1 = plt.legend(plt_1, ['Datos de entrada'], loc = 1)
    leg_2 = plt.legend(plt_2, ['Datos en el intervalo'], loc = 2)
    leg_3 = plt.legend(plt_3, ['Intervalo'], loc = 3)
    plt.gca().add_artist(leg_1)
    plt.gca().add_artist(leg_2)
    plt.gca().add_artist(leg_3)

    plt.show()
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #
# Intervalo a buscar
num_puntos = 35
intx_1 = rand.randint(0, 15)
intx_2 = rand.randint(0, 15)
if intx_1 > intx_2: intx_1, intx_2 = intx_2, intx_1
while intx_1 == intx_2:
    intx_1 = rand.randint(0, 15)
    intx_2 = rand.randint(0, 15)
inty_1 = rand.randint(0, 15)
inty_2 = rand.randint(0, 15)
if inty_1 > inty_2: inty_1, inty_2 = inty_2, inty_1
while inty_1 == inty_2:
    inty_1 = rand.randint(0, 15)
    inty_2 = rand.randint(0, 15)
intervalo_x = [intx_1, intx_2]
intervalo_y = [inty_1, inty_2]
# Datos de entrada y creación del árbol
arreglo = []
for i in range(num_puntos): arreglo.append([rand.uniform(0, 15), rand.uniform(0, 15)])
arbol = LayeredRangeTree()
arbol.build_from_list(arreglo)
# Query de los puntos a buscar
arbol.rangeQuery_2D(intervalo_x, intervalo_y)
# Impresión de gráfica y del árbol
print 'Puntos dento del intervalo: ', arbol.nodos_negros
arbol.PrintTree()
Graph(arreglo, arbol.nodos_negros, intervalo_x, intervalo_y)
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #