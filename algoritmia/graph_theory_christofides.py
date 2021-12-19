#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------LIBRERÍAS-------------------------- #
# ------------------------------------------------------------- #
import math, random, sys, time
import networkx as nx
# ------------------------------------------------------------- #
# ---------------------------CLASES---------------------------- #
class Graph:
    def __init__(self):
        self.MST_path = [ ]
        self.vertices = { }

    def add_edge(self, frm, to, weight = 1.0):
        if self.vertices.has_key(frm) == False:
            print 'Vértice de partida %s no está en el grafo'
            quit()
        if self.vertices.has_key(to) == False:
            print 'Vértice de meta %s no está en el grafo' % (to)
            quit()
        self.vertices[frm].add_adjacent(self.vertices[to], weight)
        self.vertices[to].add_adjacent(self.vertices[frm], weight)

    def add_vertex(self, id):
        aux_vertex = Vertex(id)
        self.vertices[id] = aux_vertex

    def build_min_heap(self, queue):
        for i in range(int(math.floor(len(queue)/2)), -1, -1):
            self.min_heapify(queue, i)

    def heap_extract_min(self, queue):
        heap_size = len(queue)
        if heap_size < 1:
            print('Error: Heap underflow!')
            quit()
        min = queue[0]
        queue[0] = queue[heap_size - 1]
        queue.pop()
        self.min_heapify(queue, 0)
        return min

    def min_heapify(self, queue, index):
        aux_one = index
        queue_size = len(queue)
        while True:
            left_child  = (2 * aux_one) + 1
            right_child = (2 * aux_one) + 2
            if left_child < queue_size and self.vertices[queue[left_child]].key < self.vertices[queue[aux_one]].key:
                smallest = left_child
            else:
                smallest = aux_one
            if right_child < queue_size and self.vertices[queue[right_child]].key < self.vertices[queue[smallest]].key:
                smallest = right_child
            if smallest == aux_one:
                return
            aux_two = queue[aux_one]
            queue[aux_one] = queue[smallest]
            queue[smallest] = aux_two
            aux_one = smallest

    def mst_Prim(self, root):
        mst_prim = MST(self)
        queue = [ ]
        for vertex in self.vertices:
            queue.append(vertex)
        self.vertices[root].key = 0
        self.build_min_heap(queue)
        while len(queue) != 0:
            u_vertex = self.heap_extract_min(queue)
            if self.vertices[u_vertex].parent != None:
                self.MST_path.append([graph.vertices[u_vertex].parent, graph.vertices[u_vertex].id])
                if graph.vertices[u_vertex].parent not in mst_prim.vertices:
                    mst_prim.add_vertex(graph.vertices[u_vertex].parent)
                if graph.vertices[u_vertex].id not in mst_prim.vertices:
                    mst_prim.add_vertex(graph.vertices[u_vertex].id)
                mst_prim.add_edge(graph.vertices[u_vertex].parent, graph.vertices[u_vertex].id,
                             self.vertices[u_vertex].adjacent[self.vertices[self.vertices[u_vertex].parent]])
                mst_prim.number_edges += 1
            for adjacent in self.vertices[u_vertex].adjacent:
                if adjacent.id in queue and self.vertices[u_vertex].adjacent[adjacent] < adjacent.key:
                    adjacent.parent = u_vertex
                    adjacent.key = self.vertices[u_vertex].adjacent[adjacent]
            self.build_min_heap(queue)
        return mst_prim

    def print_adjacents(self, id):
        for vertex in self.vertices[id].adjacent:
            print 'Arista %s --> %s: ' %(id, vertex.id), self.vertices[id].adjacent[vertex]

    def print_edges(self):
        for vertex in self.vertices:
            self.print_adjacents(vertex)

    def print_vertices(self):
        for vertex in self.vertices:
            print 'Vértice: ', vertex

class MST:
    def __init__(self, graph):
        self.original_graph = graph
        self.TSP_tour_cost = 0.0
        self.euler_path = [ ]
        self.hamilton_path = [ ]
        self.odd_vertices = [ ]
        self.number_edges = 0
        self.vertices = { }

    def add_edge(self, frm, to, weight = 1.0):
        if self.vertices[to] in self.vertices[frm].adjacent:
            to_prime = '\'' + to
            frm_prime = '\'' + frm
            self.add_vertex(to_prime)
            self.add_vertex(frm_prime)
            self.vertices[frm_prime].add_adjacent(self.vertices[to_prime], weight)
            self.vertices[to_prime].add_adjacent(self.vertices[frm_prime], weight)
        else:
            self.vertices[frm].add_adjacent(self.vertices[to], weight)
            self.vertices[to].add_adjacent(self.vertices[frm], weight)

    def add_vertex(self, id):
        aux_vertex = Vertex(id)
        self.vertices[id] = aux_vertex

    def euler_tour(self):
        stack_lenght = 0
        stack = [ ]
        actual_vertex = random.choice(self.vertices.keys())
        while actual_vertex[0] == '\'':
            actual_vertex = random.choice(self.vertices.keys())
        while self.vertices[actual_vertex].adjacent != {} or len(stack) != 0:
            if self.vertices[actual_vertex].adjacent == {} and len(stack) == 0:
                break
            if self.vertices[actual_vertex].adjacent == {}:
                self.euler_path.append(actual_vertex)
                actual_vertex = stack.pop(stack_lenght - 1)
                stack_lenght -= 1
            else:
                stack.append(actual_vertex)
                stack_lenght += 1
                random_neighbor = self.vertices[actual_vertex].adjacent.keys()
                random_neighbor = random.choice(random_neighbor)
                if '\'' + actual_vertex in self.vertices and '\'' + random_neighbor.id in self.vertices:
                    self.vertices.pop('\'' + actual_vertex)
                    self.vertices.pop('\'' + random_neighbor.id)
                else:
                    self.vertices[actual_vertex].adjacent.pop(random_neighbor)
                    self.vertices[random_neighbor.id].adjacent.pop(self.vertices[actual_vertex])
                actual_vertex = random_neighbor.id
        self.euler_path.append(self.euler_path[0])

    def compute_odd_vertices(self):
        for vertex in self.vertices:
            if len(self.vertices[vertex].adjacent) % 2 != 0:
                self.odd_vertices.append(self.vertices[vertex].id)

    def compute_TSP_cost(self):
        for i in range(len(self.hamilton_path) - 1):
            vertex_one = self.hamilton_path[i]
            vertex_two = self.hamilton_path[i + 1]
            aux_cost = self.original_graph.vertices[vertex_one].adjacent[self.original_graph.vertices[vertex_two]]
            self.TSP_tour_cost += aux_cost

    def hamilton_tour(self):
        aux_set = [ ]
        for i in self.euler_path:
            if i not in aux_set:
                aux_set.append(i)
        self.hamilton_path = aux_set
        self.hamilton_path.append(self.hamilton_path[0])

    def minimum_cost_pmatching(self):
        aux_graph = nx.Graph()
        max_value = -1.0
        for i in range(len(self.odd_vertices)):
            i_vertex = self.vertices[self.odd_vertices[i]].id
            for j in range(len(self.odd_vertices)):
                if i != j:
                    j_vertex = self.original_graph.vertices[self.odd_vertices[j]]
                    j_value  = self.original_graph.vertices[i_vertex].adjacent[j_vertex]
                    aux_graph.add_weighted_edges_from([(i_vertex, j_vertex.id, j_value)])
                    if j_value > max_value:
                        max_value = j_value
        for i in aux_graph.edges():
            aux_graph[i[0]][i[1]]['weight'] = max_value - aux_graph[i[0]][i[1]]['weight']
        max_weight_pmatch = nx.max_weight_matching(aux_graph)
        for i in max_weight_pmatch.keys():
            try:
                self.add_edge(i, max_weight_pmatch[i], aux_graph[i][max_weight_pmatch[i]]['weight'] + max_value)
                del max_weight_pmatch[max_weight_pmatch[i]]
            except:
                pass

    def print_adjacents(self, id):
        for vertex in self.vertices[id].adjacent:
            print 'Arista %s --> %s: ' %(id, vertex.id), self.vertices[id].adjacent[vertex]

    def print_edges(self):
        for vertex in self.vertices:
            self.print_adjacents(vertex)

    def print_results(self):
        print '#----------------------------------------------#'
        print 'Tour de Euler calculado: '
        print self.euler_path
        print '#----------------------------------------------#'
        print 'Tour de Hamilton calculado: '
        print self.hamilton_path
        print 'Costo del tour: '
        print self.TSP_tour_cost
        print '#----------------------------------------------#'
        print 'Tiempo de ejecución: '
        print time.time()-start
        print '#----------------------------------------------#'

    def print_vertices(self):
        for vertex in self.vertices:
            print 'Vértice: ', vertex

class Vertex:
    def __init__(self, id):
        self.id = id
        self.key = sys.maxint
        self.parent = None
        self.adjacent = { }

    def add_adjacent(self, vertex, weight):
        self.adjacent[vertex] = weight
# ------------------------------------------------------------- #
# -------------------------FUNCIONES--------------------------- #
def read_matrix_TSP(filename):
    file = open(filename, "r").read().splitlines()
    graph = Graph()
    num_vertices = int(file[0])
    vertices_labels = file[1].split()
    for i in vertices_labels:
        graph.add_vertex(i)
    file.pop(0), file.pop(0)
    for i in range(len(file)):
        data = file[i].split()
        for j in range(num_vertices):
            if i != j:
                graph.add_edge(vertices_labels[i], vertices_labels[j], float(data[j]))
    return graph

def christofides(graph):
    mst = graph.mst_Prim('A')
    mst.compute_odd_vertices()
    mst.minimum_cost_pmatching()
    mst.euler_tour()
    mst.hamilton_tour()
    mst.compute_TSP_cost()
    mst.print_results()
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #
start = time.time()
graph = read_matrix_TSP('TSP_Graph.txt')
christofides(graph)
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #

