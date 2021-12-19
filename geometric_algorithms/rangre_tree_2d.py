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
        self.padre = None
        self.arbol = Range_Tree2D()
        self.auxarray = []
        self.valor = val

class Range_Tree2D:
    def __init__(self):
        self.raiz = None
        self.nodos_negros = []
        self.nodos_negrosx = []

    def tree_maximum(self, node):
        if node.derecha is not None:
            return self.tree_maximum(node.derecha)
        else:
            return node

    def build_from_list(self,list):
        sorted_list = sorted(list)
        self.raiz = self.divide_list(sorted_list)
        self.internalnode_2d_tree(self.raiz, self.raiz)
        self.build_2D_Trees(self.raiz)

    def internalnode_2d_tree(self, nodohijo, nodopadre):
        if nodohijo.izquierda != None:
            self.internalnode_2d_tree(nodohijo.izquierda, nodohijo)
            nodohijo.auxarray.extend(nodohijo.izquierda.auxarray)
            self.internalnode_2d_tree(nodohijo.derecha, nodohijo)
            nodohijo.auxarray.extend(nodohijo.derecha.auxarray)
        else:
            nodohijo.auxarray.append(nodohijo.valor)

    def build_2D_Trees(self, nodo):
        if nodo.izquierda != None:
            self.build_2D_Trees(nodo.izquierda)
            nodo.arbol.build_from_list2D(nodo)
            self.build_2D_Trees(nodo.derecha)
        else:
            nodo.arbol.build_from_list2D(nodo)

    def build_from_list2D(self, nodo):
        sorted_list = sorted(nodo.auxarray, key = lambda x:x[1])
        nodo.arbol.raiz = nodo.arbol.divide_list(sorted_list)
        nodo.auxarray = []

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

    def find_split_nodeX(self, interval_1, interval_2):
        aux = self.raiz
        while aux.izquierda != None:
            if interval_2 >= aux.valor[0] and interval_1 <= aux.valor[0]:
                break
            else:
                if interval_2 <= aux.valor[0]:
                    aux = aux.izquierda
                else:
                    aux = aux.derecha
        return aux

    def find_split_nodeY(self, interval_1, interval_2):
        aux = self.raiz
        while aux.izquierda != None:
            if interval_2 >= aux.valor[1] and interval_1 <= aux.valor[1]:
                break
            else:
                if interval_2 <= aux.valor[1]:
                    aux = aux.izquierda
                else:
                    aux = aux.derecha
        return aux

    def rangeQuery_2D(self, intervalx, intervaly):
        split_node = self.find_split_nodeX(intervalx[0], intervalx[1])
        self.LeftSeachQueryX(split_node, intervalx[0], intervalx[1])
        self.RightSearchQueryX(split_node, intervalx[0], intervalx[1])
        for i in self.nodos_negrosx:
            split_node = i.arbol.find_split_nodeY(intervaly[0], intervaly[1])
            i.arbol.LeftSeachQuery(split_node, intervaly[0], intervaly[1], self)
            i.arbol.RightSearchQuery(split_node, intervaly[0], intervaly[1], self)

    def ReportSubtree(self, nodo, maintree):
        if nodo != None:
            self.ReportSubtree(nodo.izquierda, maintree)
            self.ReportSubtree(nodo.derecha, maintree)
            if nodo.izquierda == None and nodo.derecha == None:
                maintree.nodos_negros.append(nodo.valor)

    def LeftSeachQuery(self, split_node, interval_1, interval_2, maintree):
        if split_node.izquierda == None:
            if split_node.valor[1] >= interval_1 and split_node.valor[1] <= interval_2:
                maintree.nodos_negros.append(split_node.valor)
        else:
            left_search_node = split_node.izquierda
            while left_search_node.izquierda != None:
                if interval_1 <= left_search_node.valor[1]:
                    self.ReportSubtree(left_search_node.derecha, maintree)
                    left_search_node = left_search_node.izquierda
                else:
                    left_search_node = left_search_node.derecha
            if left_search_node.valor[1] >= interval_1 and left_search_node.valor[1] <= interval_2:
                maintree.nodos_negros.append(left_search_node.valor)

    def RightSearchQuery(self, split_node, interval_1, interval_2, maintree):
        if split_node.izquierda == None:
            if split_node.valor[1] <= interval_1 and split_node.valor[1] >= interval_2:
                maintree.nodos_negros.append(split_node.valor)
        else:
            right_search_node = split_node.derecha
            while right_search_node.izquierda != None:
                if interval_2 >= right_search_node.valor[1]:
                    self.ReportSubtree(right_search_node.izquierda, maintree)
                    right_search_node = right_search_node.derecha
                else:
                    right_search_node = right_search_node.izquierda
            if right_search_node.valor[1] >= interval_1 and right_search_node.valor[1] <= interval_2:
                maintree.nodos_negros.append(right_search_node.valor)

    def LeftSeachQueryX(self, split_node, interval_1, interval_2):
        if split_node.izquierda == None:
            if split_node.valor[0] >= interval_1 and split_node.valor[0] <= interval_2:
                self.nodos_negrosx.append(split_node)
        else:
            left_search_node = split_node.izquierda
            while left_search_node.izquierda != None:
                if interval_1 <= left_search_node.valor[0]:
                    self.nodos_negrosx.append(left_search_node.derecha)
                    left_search_node = left_search_node.izquierda
                else:
                    left_search_node = left_search_node.derecha
            if left_search_node.valor[0] >= interval_1 and left_search_node.valor[0] <= interval_2:
                self.nodos_negrosx.append(left_search_node)

    def RightSearchQueryX(self, split_node, interval_1, interval_2):
        if split_node.izquierda == None:
            if split_node.valor[0] <= interval_1 and split_node.valor[0] >= interval_2:
                self.nodos_negrosx.append(split_node)
        else:
            right_search_node = split_node.derecha
            while right_search_node.izquierda != None:
                if interval_2 >= right_search_node.valor[0]:
                    self.nodos_negrosx.append(right_search_node.izquierda)
                    right_search_node = right_search_node.derecha
                else:
                    right_search_node = right_search_node.izquierda
            if right_search_node.valor[0] >= interval_1 and right_search_node.valor[0] <= interval_2:
                self.nodos_negrosx.append(right_search_node)

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
num_puntos = 10
intervalo_x = [3, 12]
intervalo_y = [5, 10]
# Datos de entrada y creación del árbol
input = []
for i in range(num_puntos): input.append([rand.uniform(0, 15), rand.uniform(0, 15)])
arbol = Range_Tree2D()
arbol.build_from_list(input)
# Query de los puntos a buscar
arbol.rangeQuery_2D(intervalo_x, intervalo_y)
print 'Puntos dento del intervalo: ', arbol.nodos_negros
# Impresión de gráfica y del árbol
arbol.PrintTree()
Graph(input, arbol.nodos_negros, intervalo_x, intervalo_y)
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #