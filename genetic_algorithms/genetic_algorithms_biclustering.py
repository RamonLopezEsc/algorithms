#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------LIBRERÍAS-------------------------- #
# ------------------------------------------------------------- #
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------------------- #
random.seed()
# -------------------------PARÁMETROS-------------------------- #
# --------------------------GLOBALES--------------------------- #
filename = 'Yeast_Expression.txt'
T = 20
nr = 3
delta = 300
max_iter = 400
size_pop = 100
prob_parents = 0.9
prob_crossover = 1.0
num_max_yprime = int(0.5 * T)
prob_mutation_genes  = 1.0
prob_mutation_condi  = 0.2
# --------------------------SELECCIÓN-------------------------- #
# --------------------------OPERADORES------------------------- #
repair_new_child   = True
repair_by_mutation = False

update_by_max_num = True
update_by_random  = False
p_parameter = None
# ------------------------------------------------------------- #
# ---------------------------CLASES---------------------------- #
class pool_chromosomes:
    def __init__(self, delta):
        self.delta = delta
        self.population = [ ]
        self.paretofrontier = [ ]
        self.zvalues = [4, self.delta + 1]
        self.averages = [0.0, 0.0, 0.0] # Size, MSR, CI
        self.bestchromosome = None
    # -----------------------FUNCIONES------------------------- #
    # ---------------------INCIALIZACIÓN----------------------- #
    def init_population(self, sizepop, sizematrix):
        n_rows = sizematrix[0] - 1
        n_cols = sizematrix[1] - 1
        angle_step = (np.pi / 2) / (size_pop + 1)
        angle = angle_step
        for i in range(sizepop):
            aux_chromosome = chromosome()
            self.init_parameters(aux_chromosome, i, n_rows, n_cols, angle)
            while aux_chromosome.fvalues[1] > self.delta:
                aux_chromosome = chromosome()
                self.init_parameters(aux_chromosome, i, n_rows, n_cols, angle)
            if aux_chromosome.fvalues[1] <= self.zvalues[1]:
                self.zvalues[1] = aux_chromosome.fvalues[1]
            self.population.append(aux_chromosome)
            angle = angle + angle_step
        self.find_neighbors(sizepop)
    # --------------------------------------------------------- #
    def init_parameters(self, chromosome, id, nrows, ncols, angle):
        chromosome.id = id
        chromosome.init_genes(nrows)
        chromosome.init_conditions(ncols)
        chromosome.init_weights(angle)
        chromosome.size_function()
        chromosome.homogeneity_function()
    # --------------------------------------------------------- #
    def find_neighbors(self, sizepop):
        real_size = sizepop - 1
        for i in range(T):
            self.population[0].neighbors.append(self.population[i + 1])
            self.population[real_size].neighbors.append(self.population[real_size - i - 1])
        for i in range(1, real_size):
            if (T / 2) > i:
                for j in range(i):
                    self.population[i].neighbors.append(self.population[j])
                for j in range(i + 1, T + 1):
                    self.population[i].neighbors.append(self.population[j])
            elif ((T / 2) + i) > real_size:
                for j in range(real_size, i, -1):
                    self.population[i].neighbors.append(self.population[j])
                for j in range(i - 1, (i - 1) - (T - len(self.population[i].neighbors)), -1):
                    self.population[i].neighbors.append(self.population[j])
            else:
                ceil  = math.ceil(float(T) / 2)
                floor = math.floor(float(T) / 2)
                if i + ceil > real_size:
                    for j in range(i + 1, int(i + floor) + 1):
                        self.population[i].neighbors.append(self.population[j])
                    for j in range(i - 1, int((i - ceil) - 1), -1):
                        self.population[i].neighbors.append(self.population[j])
                else:
                    for j in range(i + 1, int(i + ceil) + 1):
                        self.population[i].neighbors.append(self.population[j])
                    for j in range(i - 1, int((i - floor) - 1), -1):
                        self.population[i].neighbors.append(self.population[j])
    # -----------------------FUNCIONES------------------------- #
    # -----------------------GENÉTICAS------------------------- #
    def reproduction(self, idcrhomosome):
        child_1, child_2 = self.crossover(idcrhomosome)
        if child_1 and child_2 != None:
            child_1.mutation()
            child_2.mutation()
            child_1.size_function()
            child_2.size_function()
            child_1.homogeneity_function()
            child_2.homogeneity_function()
            if child_1.fvalues[0]  >  child_2.fvalues[0] and child_1.fvalues[1] < child_2.fvalues[1]:
                return child_1
            elif child_2.fvalues[0] > child_1.fvalues[0] and child_2.fvalues[1] < child_1.fvalues[1]:
                return child_2
            else:
                if random.randint(0, 1) == 0:
                    return child_1
                else:
                    return child_2
        else:
            return None
    # --------------------------------------------------------- #
    def crossover(self, idcrhomosome):
        chromo = self.population[idcrhomosome]
        parent_1, parent_2 = self.select_parents(chromo)
        if random.uniform(0, 1) <= prob_crossover:
            child_1, child_2 = self.offspring(parent_1, parent_2)
            return child_1, child_2
        else:
            return None, None
    # --------------------------------------------------------- #
    def select_parents(self, chromosome):
        global p_parameter
        parent_1 = random.randint(0, T - 1)
        parent_2 = random.randint(0, T - 1)
        while parent_1 == parent_2:
            parent_2 = random.randint(0, T - 1)
        if random.uniform(0, 1) <= prob_parents:
            p_parameter = False
            parent_1 = chromosome.neighbors[parent_1]
            parent_2 = chromosome.neighbors[parent_2]
            return parent_1, parent_2
        else:
            p_parameter = True
            random_chromosome_1 = random.randint(0, size_pop - 1)
            random_chromosome_2 = random.randint(0, size_pop - 1)
            parent_1 = self.population[random_chromosome_1].neighbors[parent_1]
            parent_2 = self.population[random_chromosome_2].neighbors[parent_2]
            return parent_1, parent_2
    # --------------------------------------------------------- #
    def offspring(self, parent1, parent2):
        child_1 = chromosome()
        child_2 = chromosome()
        self.cross_genes(parent1, parent2, child_1, child_2)
        self.cross_conditions(parent1, parent2, child_1, child_2)
        return child_1, child_2
    # --------------------------------------------------------- #
    def cross_genes(self, parent1, parent2, child1, child2):
        crossover_point_genes = parent1.genes[random.randint(0, len(parent1.genes) - 1)]
        for i in range(len(parent1.genes)):
            if parent1.genes[i] <= crossover_point_genes:
                child1.genes.append(parent1.genes[i])
            if parent1.genes[i] >= crossover_point_genes:
                child2.genes.append(parent1.genes[i])
        for i in range(len(parent2.genes)):
            if parent2.genes[i] > crossover_point_genes:
                child1.genes.append(parent2.genes[i])
            if parent2.genes[i] < crossover_point_genes:
                child2.genes.append(parent2.genes[i])
    # --------------------------------------------------------- #
    def cross_conditions(self, parent1, parent2, child1, child2):
        crossover_point_condi = parent1.coditions[random.randint(0, len(parent1.coditions) - 1)]
        for i in range(len(parent1.coditions)):
            if parent1.coditions[i] <= crossover_point_condi:
                child1.coditions.append(parent1.coditions[i])
            if parent1.coditions[i] >= crossover_point_condi:
                child2.coditions.append(parent1.coditions[i])
        for i in range(len(parent2.coditions)):
            if parent2.coditions[i] > crossover_point_condi:
                child1.coditions.append(parent2.coditions[i])
            if parent2.coditions[i] < crossover_point_condi:
                child2.coditions.append(parent2.coditions[i])
    # -----------------------FUNCIONES------------------------- #
    # -----------------------GENERALES------------------------- #
    def update_zvalues(self, yprime):
        if  self.zvalues[0] < yprime.fvalues[0]:
            self.zvalues[0] = yprime.fvalues[0]
        if  self.zvalues[1] > yprime.fvalues[1]:
            self.zvalues[1] = yprime.fvalues[1]
    # --------------------------------------------------------- #
    def update_pareto(self, yprime):
        index_remove = [ ] # Índices dominados por y prima
        aux_pareto = [ ]   # Índices que pertenecen a la frontera
        bool_yprime = True # Booleano para verificar si y prima está en la frontera
        if yprime.size[0] > 1 and yprime.size[1] > 1:
            for num, i in enumerate(self.paretofrontier):
                if (yprime.fvalues[0] >= i.fvalues[0] and yprime.fvalues[1] <  i.fvalues[1]) or\
                   (yprime.fvalues[0] >  i.fvalues[0] and yprime.fvalues[1] <= i.fvalues[1]) :
                    index_remove.append(num) # Almacenar índices dominados por y prima
            for num, i in enumerate(self.paretofrontier):
                if num not in index_remove:
                    aux_pareto.append(i) # Almacenar índices no dominados por y prima
            for num, i in enumerate(aux_pareto):
                if (i.fvalues[0] >= yprime.fvalues[0] and i.fvalues[1] <  yprime.fvalues[1]) or\
                   (i.fvalues[0] >  yprime.fvalues[0] and i.fvalues[1] <= yprime.fvalues[1]) :
                    bool_yprime = False
                    break # Si y prima está dominado por un x, no se almacena en la frontera
            if bool_yprime == True:
                aux_pareto.append(y_prime)
            self.paretofrontier = aux_pareto
    # --------------------------------------------------------- #
    def graph_pareto(self, iter, run):
        x_axis = [ ]
        y_axis = [ ]
        plot_margin = 3.0
        for i in self.paretofrontier:
            x_axis.append(i.fvalues[0])
            y_axis.append(i.fvalues[1])

        plt.plot(x_axis, y_axis, 'ko')
        plt.plot(self.zvalues[0], self.zvalues[1], 'ro')

        plt.tight_layout()
        x0, x1, y0, y1 = plt.axis()
        plt.axis((x0 - plot_margin,
                  x1 + plot_margin,
                  y0 - plot_margin,
                  y1 + plot_margin))

        if iter == max_iter - 1:
            x_lim_izq = 10000
            x_lim_der = 16000
            y_lim_inf = 240
            y_lim_sup = 300
            # plt.xlim(x_lim_izq, x_lim_der)
            # plt.ylim(y_lim_inf, y_lim_sup)
            plt.plot(x_axis, y_axis, 'ko')
            plt.plot(self.zvalues[0], self.zvalues[1], 'ro')
            plt.savefig('Resultados\Corrida_%s.png'%(str(run)))
        else:

            #pass
            plt.pause(0.1)
            plt.draw()
            plt.cla()
    # --------------------------------------------------------- #
    def pareto_statistics(self):
        aux = [0.0, 0.0, 0.0] # Size, MSR, CI
        aux_frontier = [ ]
        min_ci = [self.paretofrontier[0], self.paretofrontier[0].fvalues[1] /  self.paretofrontier[0].fvalues[0]]
        for i in self.paretofrontier:
            if i.fvalues[0] >= 10000 and i.fvalues[1] >= 240:
                aux_frontier.append(i)
        self.paretofrontier = aux_frontier
        for i in self.paretofrontier:
            aux[0] = aux[0] + i.fvalues[0]
            aux[1] = aux[1] + i.fvalues[1]
            aux[2] = aux[2] + (i.fvalues[1] / i.fvalues[0])
            if i.fvalues[1] / i.fvalues[0] <= min_ci[1]:
                min_ci[0] = i
                min_ci[1] = i.fvalues[1] / i.fvalues[0]
        aux[0] = aux[0] / len(self.paretofrontier)
        aux[1] = aux[1] / len(self.paretofrontier)
        aux[2] = aux[2] / len(self.paretofrontier)
        self.averages = aux
        self.bestchromosome = min_ci
# ------------------------------------------------------------- #
class chromosome:
    def __init__(self):
        self.id = None
        self.size = [ ]
        self.genes = [ ]
        self.coditions = [ ]
        self.fvalues = [4, 0.0]
        self.weights = [ ]
        self.neighbors = [ ]
    # --------------------------------------------------------- #
    def init_genes(self, nrows):
        self.genes.append(random.randint(0, nrows))
        index_gene = random.randint(0, nrows)
        while index_gene in self.genes:
            index_gene = random.randint(0, nrows)
        self.genes.append(index_gene)
    # --------------------------------------------------------- #
    def init_conditions(self, ncols):
        self.coditions.append(random.randint(0, ncols))
        index_condition = random.randint(0, ncols)
        while index_condition in self.coditions:
            index_condition = random.randint(0, ncols)
        self.coditions.append(index_condition)
    # --------------------------------------------------------- #
    def init_weights(self, angle):
        comp_x = np.cos(angle)
        comp_y = (1 - (comp_x ** 2)) ** 0.5
        self.weights.append(comp_x)
        self.weights.append(comp_y)
    # --------------------------------------------------------- #
    def init_zero_values(self, dimarray):
        output_array = [ ]
        for i in range(dimarray):
            output_array.append(0.0)
        return output_array
    # --------------------------------------------------------- #
    def mutation(self):
        if random.uniform(0, 1) <= prob_mutation_genes:
            gene_index = random.randint(0, size_matrix[0] - 1)
            if gene_index in self.genes:
                if len(self.genes) == 1:
                    aux_gene_index = random.randint(0, size_matrix[0] - 1)
                    while gene_index == aux_gene_index:
                        aux_gene_index = random.randint(0, size_matrix[0] - 1)
                    self.genes.remove(gene_index)
                    self.genes.append(aux_gene_index)
                else:
                    self.genes.remove(gene_index)
            else:
                self.genes.append(gene_index)
        if random.uniform(0, 1) <= prob_mutation_condi:
            condition_index = random.randint(0, size_matrix[1] - 1)
            if condition_index in self.coditions:
                if len(self.coditions) == 1:
                    aux_cond_index = random.randint(0, size_matrix[1] - 1)
                    while condition_index == aux_cond_index:
                        aux_cond_index = random.randint(0, size_matrix[1] - 1)
                    self.coditions.remove(condition_index)
                    self.coditions.append(aux_cond_index)
                else:
                    self.coditions.remove(condition_index)
            else:
                self.coditions.append(condition_index)
    # --------------------------------------------------------- #
    def repair(self, poolpopulation, idcrhomosome):
        if repair_new_child == True:
            while self.fvalues[1] > delta:
                repair_child = self.repair_new_child(poolpopulation, idcrhomosome)
                self.size = repair_child.size
                self.genes = repair_child.genes
                self.coditions = repair_child.coditions
                self.fvalues = repair_child.fvalues
        elif repair_by_mutation == True:
            while self.fvalues[1] > delta:
                self.repair_by_mutation()
    # --------------------------------------------------------- #
    def repair_new_child(self, poolpopulation, idcrhomosome):
        chromo = poolpopulation.population[idcrhomosome]
        parent_1, parent_2 = poolpopulation.select_parents(chromo)
        child_1, child_2 = poolpopulation.offspring(parent_1, parent_2)
        child_1.mutation()
        child_1.size_function()
        child_1.homogeneity_function()
        child_2.mutation()
        child_2.size_function()
        child_2.homogeneity_function()
        if random.randint(0, 1) == 0:
            return child_1
        else:
            return child_2
    # --------------------------------------------------------- #
    def repair_by_mutation(self):
        if self.size[0] == 0:
            self.genes.append(random.randint(0, size_matrix[0] - 1))
        else:
            aux = random.randint(0, self.size[0] - 1)
            self.genes.pop(aux)
        if len(self.coditions) == 0:
            self.coditions.append(random.randint(0, size_matrix[1] - 1))
        else:
            self.coditions.pop(random.randint(0, self.size[1] - 1))
        self.size[0] = len(self.genes)
        self.size[1] = len(self.coditions)
        size = self.size[0] * self.size[1]
        self.fvalues[0] = size
        self.homogeneity_function()
    # -----------------------FUNCIONES------------------------- #
    # -----------------------OBJETIVO-------------------------- #
    def size_function(self):
        self.size.append(len(self.genes))
        self.size.append(len(self.coditions))
        size = self.size[0] * self.size[1]
        self.fvalues[0] = size
    # --------------------------------------------------------- #
    def homogeneity_function(self):
        # TODO No estoy seguro que se ocupe ordenar estas variables
        self.genes.sort()
        self.coditions.sort()
        e_ic, e_gj, e_gc = self.compute_parameters()
        self.fvalues[1] = 0.0
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                e_ij = data[self.genes[i]][self.coditions[j]]
                self.fvalues[1] = self.fvalues[1] + ((e_ij - e_ic[i] - e_gj[j] + e_gc) ** 2)
        self.fvalues[1] = self.fvalues[1] / self.fvalues[0]
    # --------------------------------------------------------- #
    def compute_parameters(self):
        e_ic = self.init_zero_values(self.size[0])
        e_gj = self.init_zero_values(self.size[1])
        e_gc = 0.0
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                e_gj[j] = e_gj[j] + float(data[self.genes[i]][self.coditions[j]])
                e_gc = e_gc + float(data[self.genes[i]][self.coditions[j]])
        for i in range(self.size[1]):
            for j in range(self.size[0]):
                e_ic[j] = e_ic[j] + float(data[self.genes[j]][self.coditions[i]])
        for i in range(self.size[1]):
            e_gj[i] = e_gj[i] / self.size[0]
        for i in range(self.size[0]):
            e_ic[i] = e_ic[i] / self.size[1]
        e_gc = e_gc / self.fvalues[0]
        return e_ic, e_gj, e_gc
    # -----------------------FUNCIONES------------------------- #
    # -----------------------GENERALES------------------------- #
    def update_neighborhood(self, yprime, zvalues):
        if update_by_max_num == True:
            self.update_by_max_num(yprime, zvalues)
        elif update_by_random == True:
            self.update_by_random(yprime, zvalues)
    # -----------------------FUNCIONES------------------------- #
    def update_by_max_num(self, yprime, zvalues):
        counter = 0
        for i in range(T):
            yprime.weights = self.neighbors[i].weights
            yprime_tchebycheff = yprime.tchebycheff(zvalues)
            neighbor_tchebycheff = self.neighbors[i].tchebycheff(zvalues)
            if yprime_tchebycheff <= neighbor_tchebycheff:
                if yprime.size[0] > 1 and yprime.size[1] > 1:
                    counter += 1
                    if counter <= num_max_yprime:
                        self.neighbors[i].size = yprime.size
                        self.neighbors[i].genes = yprime.genes
                        self.neighbors[i].coditions = yprime.coditions
                        self.neighbors[i].fvalues = yprime.fvalues
    # --------------------------------------------------------- #
    def update_by_random(self, yprime, zvalues):
        p_set = [ ]
        counter = 0
        if p_parameter == False:
            self.p_neighborhood(counter, p_set, yprime, zvalues)
        else:
            self.p_neighborhood(counter, p_set, yprime, zvalues)
    # --------------------------------------------------------- #
    def p_neighborhood(self, counter, p_set, yprime, zvalues):
        for i in range(T - 1):
            if counter == nr:
                break
            random_index = random.randint(0, T - 1)
            while random_index in p_set:
                random_index = random.randint(0, T - 1)
            p_set.append(random_index)
            yprime.weights = self.neighbors[random_index].weights
            yprime_tchebycheff = yprime.tchebycheff(zvalues)
            neighbor_tchebycheff = self.neighbors[random_index].tchebycheff(zvalues)
            if yprime_tchebycheff <= neighbor_tchebycheff:
                self.neighbors[i].size = yprime.size
                self.neighbors[i].genes = yprime.genes
                self.neighbors[i].coditions = yprime.coditions
                self.neighbors[i].fvalues = yprime.fvalues
    # --------------------------------------------------------- #
    def p_population(self, counter, p_set, yprime, zvalues):
        for i in range(size_pop - 1):
            if counter == nr:
                break
            random_index = random.randint(0, size_pop - 1)
            while random_index in p_set:
                random_index = random.randint(0, size_pop - 1)
            p_set.append(random_index)
            yprime.weights = population.population[random_index].weights
            yprime_tchebycheff = yprime.tchebycheff(zvalues)
            neighbor_tchebycheff = population.population[random_index].tchebycheff(zvalues)
            if yprime_tchebycheff <= neighbor_tchebycheff:
                population.population[random_index].size = yprime.size
                population.population[random_index].genes = yprime.genes
                population.population[random_index].coditions = yprime.coditions
                population.population[random_index].fvalues = yprime.fvalues
    # --------------------------------------------------------- #
    def tchebycheff(self, zvalues):
        num_obj_functions = len(self.fvalues)
        set_tchebycheff = [ ]
        for i in range(num_obj_functions):
            aux = self.weights[i] * abs(self.fvalues[i] - zvalues[i])
            set_tchebycheff.append(aux)
        return max(set_tchebycheff)
# -------------------------FUNCIONES--------------------------- #
# --------------------------BÁSICAS---------------------------- #
def read_file_human(filename):
    data = [ ]
    file = open(filename, "r").read().splitlines()
    for i in range(len(file)):
        data_aux = [ ]
        aux_var = file[i].split()
        for j in range(len(aux_var)):
            try:
                aux_var_float = int(aux_var[j])
                if aux_var_float == 999:
                    if len(data_aux) == 0:
                        aux_var_float = int(sum(data[i - 1]) / len(data[i - 1]))
                    else:
                        aux_var_float = int(sum(data_aux) / len(data_aux))
                data_aux.append(aux_var_float)
            except:
                if aux_var[j][0] == '-':
                    aux_var_float = [float(x) * -1 for x in aux_var[j].split('-') if x != '']
                    for k in range(len(aux_var_float)):
                        if aux_var_float[k] == 999.0:
                            if len(data_aux) == 0:
                                aux_var_float[k] = int(sum(data[i - 1]) / len(data[i - 1]))
                            else:
                                aux_var_float[k] = int(sum(data_aux) / len(data_aux))
                        data_aux.append(aux_var_float[k])
                else:
                    aux_var_float = [float(x) * -1 for x in aux_var[j].split('-') if x != '']
                    aux_var_float[0] = aux_var_float[0] * -1
                    for k in range(len(aux_var_float)):
                        if aux_var_float[k] == 999.0:
                            if len(data_aux) == 0:
                                aux_var_float[k] = int(sum(data[i - 1]) / len(data[i - 1]))
                            else:
                                aux_var_float[k] = int(sum(data_aux) / len(data_aux))
                        data_aux.append(aux_var_float[k])
        data.append(data_aux)
    return data
# ------------------------------------------------------------- #
def read_file_yeast(filename):
    data = [ ]
    fichero = open(filename, "r").read().splitlines()
    for i in range(len(fichero)):
        data.append(fichero[i].split())
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
            if data[i][j] == -1.0:
                data[i][j] = float(random.randint(0, 800))
    return data
# ------------------------------------------------------------- #
def size_data_matrix(data):
    size_matrix = [ ]
    size_matrix.append(len(data))
    size_matrix.append(len(data[0]))
    return size_matrix
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #

# data = read_file_human(filename)
data = read_file_yeast(filename)

for iter in range(8):
    print 'Corrida número ', iter
    start_time = time.time()
    size_matrix = size_data_matrix(data)
    population = pool_chromosomes(delta)
    population.init_population(size_pop, size_matrix)
    for i in range(max_iter):
        print 'Gen ', i
        for j in range(size_pop):
            y_prime = population.reproduction(j)
            if y_prime != None:
                y_prime.repair(population, j)
                population.update_zvalues(y_prime)
                population.population[j].update_neighborhood(y_prime, population.zvalues)
                population.update_pareto(y_prime)
        population.graph_pareto(i, iter)
    population.pareto_statistics()
    summary_text = open("Resultados\Corrida_%s.txt"%(str(iter)), 'w')
    summary_text.write('Estadísticas!\n')
    summary_text.write('Tamaño de bicluster promedio: %s\n'%(str(population.averages[0])))
    summary_text.write('MSR promedio: %s\n'%(str(population.averages[1])))
    summary_text.write('CI promedio: %s\n'%(str(population.averages[2])))
    summary_text.write('Mejor individuo!\n')
    summary_text.write('Valores de funciones objetivo: %s\n'%(str(population.bestchromosome[0].fvalues)))
    summary_text.write('Tamaño de la matriz: %s\n'%(str(population.bestchromosome[0].size)))
    summary_text.write('CI: %s\n'%(str(population.bestchromosome[1])))
    end_time = time.time() - start_time
    summary_text.write('Zvalues: %s'%(str(population.zvalues)))
    summary_text.write('Tiempo: %s'%(str(end_time)))
    summary_text.close()
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #