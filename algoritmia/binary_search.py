#!/usr/bin/env python
# -*- coding: utf-8 -*-

def binary_search(num_buscar, sorted_array):
    limite_inferior = 0
    limite_superior = len(arreglo) - 1
    while True:
        if limite_superior < limite_inferior:
            return None
        bisect = (limite_inferior + limite_superior) / 2
        if sorted_array[bisect] < num_buscar:
            limite_inferior = bisect + 1
        elif sorted_array[bisect] > num_buscar:
            limite_superior = bisect - 1
        elif sorted_array[bisect] == num_buscar:
            return bisect

num_buscar = 3
arreglo = [0, 1, 2, 3, 4, 5, 6, 7]
result = binary_search(num_buscar, arreglo)

print result