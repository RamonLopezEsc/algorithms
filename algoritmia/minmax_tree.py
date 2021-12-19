# -*- coding: utf-8 -*-

from math import inf

class Nodo(object):
    def __init__(self, inputval=0):
        self.izq  = None
        self.der  = None
        self.data = inputval

    # Funcion extraida de la siguiente fuente:
    # https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python/34013268#34013268
    def print_as_tree(self):
        lines, *_ = self._aux_print_as_tree()
        for line in lines:
            print(line)

    def _aux_print_as_tree(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.der is None and self.izq is None:
            line = '%s' % self.data
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only izq child.
        if self.der is None:
            lines, n, p, x = self.izq._aux_print_as_tree()
            s = '%s' % self.data
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only der child.
        if self.izq is None:
            lines, n, p, x = self.der._aux_print_as_tree()
            s = '%s' % self.data
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        izq, n, p, x = self.izq._aux_print_as_tree()
        der, m, q, y = self.der._aux_print_as_tree()
        s = '%s' % self.data
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            izq += [n * ' '] * (q - p)
        elif q < p:
            der += [m * ' '] * (p - q)
        zipped_lines = zip(izq, der)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    def minimax(self, type, prof, alfa=-inf, beta=inf):
        prof -= 1
        if (self.izq is None and self.der is None) or prof == 0:
            return self, 'None', self.data, self.data
        if self.izq is not None:
            returned_izq_node, direccion, ret_alfa, ret_beta = self.izq.minimax(not type, prof, alfa, beta)
            if type:
                if alfa < ret_alfa:
                    alfa = ret_alfa
            else:
                if beta > ret_beta:
                    beta = ret_beta
        else:
            returned_izq_node = None
        if alfa < beta:
            if self.der is not None:
                returned_der_node, direccion, ret_alfa, ret_beta = self.der.minimax(not type, prof, alfa, beta)
                if type:
                    if alfa < ret_alfa:
                        alfa = ret_alfa
                else:
                    if beta > ret_beta:
                        beta = ret_beta
            else:
                returned_der_node = None
        else:
            print('Poda Aplicada! Nodo Actual: ', self.data)
            if self.der is not None:
                print('Nodo NO vistado: ', self.der.data)
            else:
                print('No habia rubramas que podar :(')
            returned_der_node = None
        if returned_izq_node is not None and returned_der_node is not None:
            if type:
                if returned_izq_node.data > returned_der_node.data:
                    return returned_izq_node, 'Izquierda', alfa, alfa
                else:
                    return returned_der_node, 'Derecha', alfa, alfa
            if returned_izq_node.data < returned_der_node.data:
                return returned_izq_node, 'Izquierda', beta, beta
            else:
                return returned_der_node, 'Derecha', beta, beta
        if returned_izq_node is not None and returned_der_node is None:
            if type:
                return returned_izq_node, 'Izquierda', alfa, alfa
            return returned_izq_node, 'Izquierda', beta, beta
        if returned_izq_node is None and returned_der_node is not None:
            if type:
                return returned_der_node, 'Derecha', alfa, alfa
            return returned_der_node, 'Derecha', beta, beta

# Arbol ejemplo para los primeros dos ejercicios
raiz = Nodo(0)
raiz.izq = Nodo(4)
raiz.der = Nodo(9)
raiz.izq.izq = Nodo(5)
raiz.izq.der = Nodo(2)
raiz.der.izq = Nodo(1)
raiz.der.der = Nodo(-3)
raiz.izq.izq.izq = Nodo(3)
raiz.izq.izq.der = Nodo(5)
raiz.izq.der.izq = Nodo(6)
raiz.izq.der.der = Nodo(9)
raiz.der.izq.izq = Nodo(1)
raiz.der.izq.der = Nodo(2)
raiz.der.der.izq = Nodo(0)
raiz.der.der.der = Nodo(-1)

# Impresion de resultados
print('---------------------------------------------')
print('Impresion del arbol:')
raiz.print_as_tree()
print()

# Ejemplos para verificar funcionalidad
print('---------------------------------------------')
returned_node, direction, alfa, beta = raiz.minimax(True, 4)
print('---------------------------------------------')
print('Profundidad 4, Raiz Max')
print('Valor del nodo seleccionado: ', returned_node.data)
print('Direccion: ', direction)
print('---------------------------------------------')
