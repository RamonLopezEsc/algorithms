#!/usr/bin/env python
# -*- coding: utf-8 -*-

def parent(index_query):
    index_parent = int(index_query / 2)
    return index_parent

def left_child(index_query):
    index_lchild = (2 * index_query) + 1
    return index_lchild

def right_child(index_query):
    index_rchild = (2 * index_query) + 2
    return index_rchild

def max_heapify(array, i):
    aux_1 = i
    largest = -1
    while largest != i:
        i = aux_1
        left = left_child(i)
        right = right_child(i)

        if left <= len(array) and array[left] > array[i]:
            largest = left
        elif right <= len(array) and array[right] > array[largest]:
            largest = right
        else:
            largest = i

        aux_2 = array[i]
        array[i] = array[largest]
        array[largest] = aux_2

        aux_1 = largest

array = [16, 4, 10, 14, 7, 9, 3, 2, 8, 1]
max_heapify(array, 1)
print array



