#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------LIBRERIAS-------------------------- #
# ------------------------------------------------------------- #
import random as rand
import matplotlib.pyplot as plt
# ------------------------------------------------------------- #
# ---------------------------CLASES---------------------------- #
class Node:
    def __init__(self, val):
        self.izquierda = None
        self.derecha = None
        self.valor = val

class Range_Tree:
    def __init__(self):
        self.raiz = None
        self.nodos_negros = []

    def tree_maximum(self, node):
        if node.derecha is not None:
            return self.tree_maximum(node.derecha)
        else:
            return node

    def build_from_list(self,list):
        sorted_list = sorted(list)
        self.raiz = self.divide_list(sorted_list)

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

    def rangeQuery_1D(self, interval_1, interval_2):
        split_node = self.find_split_node(interval_1, interval_2)
        self.LeftSeachQuery(split_node, interval_1, interval_2)
        self.RightSearchQuery(split_node, interval_1, interval_2)
        self.nodos_negros.sort()

    def find_split_node(self, interval_1, interval_2):
        aux = self.raiz
        while aux.izquierda != None:
            if interval_2 >= aux.valor or interval_1 > aux.valor:
                break
            else:
                if interval_2 <= aux.valor:
                    aux = aux.izquierda
                else:
                    aux = aux.derecha
        return aux

    def ReportSubtree(self, nodo):
        if nodo != None:
            self.ReportSubtree(nodo.izquierda)
            self.ReportSubtree(nodo.derecha)
            if nodo.izquierda == None and nodo.derecha == None:
                self.nodos_negros.append(nodo.valor)

    def LeftSeachQuery(self, split_node, interval_1, interval_2):
        if split_node.izquierda == None:
            if split_node.valor >= interval_1 and split_node.valor <= interval_2:
                self.nodos_negros.append(split_node.valor)
        else:
            left_search_node = split_node.izquierda
            while left_search_node.izquierda != None:
                if interval_1 <= left_search_node.valor:
                    self.ReportSubtree(left_search_node.derecha)
                    left_search_node = left_search_node.izquierda
                else:
                    left_search_node = left_search_node.derecha
            if left_search_node.valor >= interval_1 and left_search_node.valor <= interval_2:
                self.nodos_negros.append(left_search_node.valor)

    def RightSearchQuery(self, split_node, interval_1, interval_2):
        if split_node.derecha == None:
            if split_node.valor <= interval_1 and split_node.valor >= interval_2:
                self.nodos_negros.append(split_node.valor)
        else:
            right_search_node = split_node.derecha
            while right_search_node.izquierda != None:
                if interval_2 >= right_search_node.valor:
                    self.ReportSubtree(right_search_node.izquierda)
                    right_search_node = right_search_node.derecha
                else:
                    right_search_node = right_search_node.izquierda
            if right_search_node.valor >= interval_1 and right_search_node.valor <= interval_2:
                self.nodos_negros.append(right_search_node.valor)

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
def Graph(lista, resultado, intervalo1, intervalo2):
    y_lista = []
    y_resultado = []
    linea_interv_x1 = [intervalo_1, intervalo_1]
    linea_interv_x2 = [intervalo_2, intervalo_2]
    linea_interv_y = [0.1, -0.1]
    for i in range(len(lista)): y_lista.append(0)
    for i in range(len(resultado)): y_resultado.append(0)

    x_lim_der = max(lista) + 0.5
    x_lim_izq = min(lista) - 0.5
    y_lim_sup = 0.1
    y_lim_inf = -0.1
    plt.xlim(x_lim_izq, x_lim_der)
    plt.ylim(y_lim_inf, y_lim_sup)

    plt_1 = plt.plot(lista, y_lista, 'ko')
    plt_2 = plt.plot(resultado, y_resultado, 'ro')
    plt_3 = plt.plot(linea_interv_x1, linea_interv_y, 'r--')
    plt.plot(linea_interv_x2, linea_interv_y, 'r--')

    plt.title('Problema: Busqueda de Rango 1D')
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
num_puntos = 80
intervalo_1 = 1
intervalo_2 = 6
# Datos de entrada y creación del árbol
input  = []
for i in range(num_puntos): input.append(rand.uniform(0, 15))
arbol = Range_Tree()
arbol.build_from_list(input)
# Query de los puntos a buscar
arbol.rangeQuery_1D(intervalo_1, intervalo_2)
print 'Puntos dento del intervalo: ', arbol.nodos_negros
# Impresión de gráfica y del árbol
arbol.PrintTree()
Graph(input, arbol.nodos_negros, intervalo_1, intervalo_2)
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #