# -*- coding: utf-8 -*-

from math import ceil
from numpy import array as nparr
from copy import deepcopy
from sys import maxsize
from math import inf
from numpy.random import randint as nprandint, seed

seed(26031991)

class Nodo(object):
    def __init__(self, tablero, posMax, posMin, playertype, score, booldummy=False):
        self.izq  = None
        self.der  = None
        self.arr  = None
        self.aba  = None
        self.tablero = deepcopy(tablero)
        self.pos_max = posMax
        self.pos_min = posMin
        if booldummy:
            self.data = score
        else:
            if playertype:
                self.data = self.tablero[self.pos_max[0]][self.pos_max[1]] + score
                self.tablero[self.pos_max[0]][self.pos_max[1]] = None
            else:
                self.data = score - self.tablero[self.pos_min[0]][self.pos_min[1]]
                self.tablero[self.pos_min[0]][self.pos_min[1]] = None

    def aux_alfabeta(self, alfa, beta, retalfa, retbeta, playertype):
        if playertype:
            if alfa < retalfa:
                alfa = retalfa
        else:
            if beta > retbeta:
                beta = retbeta
        return alfa, beta

    def aux_check_node(self, alfa, beta, node, playertype, prof):
        if alfa < beta:
            if node is not None:
                returned_node, direccion, ret_alfa, ret_beta = node.minimax(not playertype, prof, alfa, beta)
                alfa, beta = self.aux_alfabeta(alfa, beta, ret_alfa, ret_beta, playertype)
            else:
                returned_node = self.return_dummy_node(playertype)
        else:
            # print('Poda Aplicada! Nodo Actual: ', self.data)
            returned_node = self.return_dummy_node(playertype)
        return returned_node, alfa, beta

    def return_dummy_node(self, playertype):
        if playertype:
            return Nodo(self.tablero, self.pos_max, self.pos_min, playertype, -maxsize, True)
        return Nodo(self.tablero, self.pos_max, self.pos_min, playertype, maxsize, True)

    def select_min_max_node(self, nodeizq, nodeder, nodearr, nodeaba, playertype, alfa, beta):
        if playertype:
            dummy_max = max(nodeizq.data, nodeder.data, nodearr.data, nodeaba.data)
            if dummy_max == nodeizq.data:
                return nodeizq, 'Izquierda', alfa, alfa
            elif dummy_max == nodeder.data:
                return nodeder, 'Derecha', alfa, alfa
            elif dummy_max == nodearr.data:
                return nodearr, 'Arriba', alfa, alfa
            return nodeaba, 'Abajo', alfa, alfa
        dummy_min = min(nodeizq.data, nodeder.data, nodearr.data, nodeaba.data)
        if dummy_min == nodeizq.data:
            return nodeizq, 'Izquierda', beta, beta
        elif dummy_min == nodeder.data:
            return nodeder, 'Derecha', beta, beta
        elif dummy_min == nodearr.data:
            return nodearr, 'Arriba', beta, beta
        return nodeaba, 'Abajo', beta, beta

    def minimax(self, playertype, prof, alfa=-inf, beta=inf):
        prof -= 1
        if (self.izq is None and self.der is None and self.arr is None and self.aba is None) or prof == 0:
            return self, 'None', self.data, self.data
        if self.izq is not None:
            returned_izq_node, direccion, ret_alfa, ret_beta = self.izq.minimax(not playertype, prof, alfa, beta)
            alfa, beta = self.aux_alfabeta(alfa, beta, ret_alfa, ret_beta, playertype)
        else:
            returned_izq_node = self.return_dummy_node(playertype)
        returned_der_node, alfa, beta = self.aux_check_node(alfa, beta, self.der, playertype, prof)
        returned_arr_node, alfa, beta = self.aux_check_node(alfa, beta, self.arr, playertype, prof)
        returned_aba_node, alfa, beta = self.aux_check_node(alfa, beta, self.aba, playertype, prof)
        return self.select_min_max_node(returned_izq_node, returned_der_node, returned_arr_node,
                                        returned_aba_node, playertype, alfa, beta)

    def ampl_print(self):
        queue = Cola()
        queue.enqueue(self)
        while not queue.empty():
            actual_node = queue.dequeue()
            print(actual_node.data)
            if actual_node.izq is not None:
                queue.enqueue(actual_node.izq)
            if actual_node.der is not None:
                queue.enqueue(actual_node.der)
            if actual_node.arr is not None:
                queue.enqueue(actual_node.arr)
            if actual_node.aba is not None:
                queue.enqueue(actual_node.aba)

class Cola(object):
    def __init__(self, capacity=100):
        self.data = []
        self.capacity = capacity
        self._nelements = 0

    def empty(self):
        return False if self._nelements != 0 else True

    def enqueue(self, element):
        if self._nelements != self.capacity:
            self._nelements += 1
            self.data.insert(0, element)

    def dequeue(self):
        self._nelements -= 1
        return self.data.pop()

    def peek(self):
        return self.data[0]

    def print(self):
        ret_str = ''
        for i in range(self._nelements - 1, -1, -1):
            ret_str += '%s <-- ' % self.data[i]
        print(ret_str[:-5])

def generar_arbol(inputList, playertype):
    aux_list = [ ]
    # Si el jugador es MAX, trae los valores MIN
    if playertype:
        for inode in inputList:
            # NODO ARRIBA MIN
            if inode.tablero[inode.pos_min[0] - 1][inode.pos_min[1]] != None:
                inode.arr = Nodo(inode.tablero, inode.pos_max, [inode.pos_min[0] - 1, inode.pos_min[1]], not playertype, inode.data)
                aux_list.append(inode.arr)
            # NODO ABAJO MIN
            if inode.tablero[inode.pos_min[0] + 1][inode.pos_min[1]] != None:
                inode.aba = Nodo(inode.tablero, inode.pos_max, [inode.pos_min[0] + 1, inode.pos_min[1]], not playertype, inode.data)
                aux_list.append(inode.aba)
            # NODO DERECHA MIN
            if inode.tablero[inode.pos_min[0]][inode.pos_min[1] + 1] != None:
                inode.der = Nodo(inode.tablero, inode.pos_max, [inode.pos_min[0], inode.pos_min[1] + 1], not playertype, inode.data)
                aux_list.append(inode.der)
            # NODO IZQUIERDA MIN
            if inode.tablero[inode.pos_min[0]][inode.pos_min[1] - 1] != None:
                inode.izq = Nodo(inode.tablero, inode.pos_max, [inode.pos_min[0], inode.pos_min[1] - 1], not playertype, inode.data)
                aux_list.append(inode.izq)
    # Si el jugador es MIN, trae los valores MAX
    else:
        for inode in inputList:
            # NODO ARRIBA MAX
            if inode.tablero[inode.pos_max[0] - 1][inode.pos_max[1]] != None:
                inode.arr = Nodo(inode.tablero, [inode.pos_max[0] - 1, inode.pos_max[1]], inode.pos_min, not playertype, inode.data)
                aux_list.append(inode.arr)
            # NODO ABAJO MAX
            if inode.tablero[inode.pos_max[0] + 1][inode.pos_max[1]] != None:
                inode.aba = Nodo(inode.tablero, [inode.pos_max[0] + 1, inode.pos_max[1]], inode.pos_min, not playertype, inode.data)
                #pprint(inode.aba.tablero)
                aux_list.append(inode.aba)
            # NODO DERECHA MAX
            if inode.tablero[inode.pos_max[0]][inode.pos_max[1] + 1] != None:
                inode.der = Nodo(inode.tablero, [inode.pos_max[0], inode.pos_max[1] + 1], inode.pos_min, not playertype, inode.data)
                aux_list.append(inode.der)
            # NODO IZQUIERDA MAX
            if inode.tablero[inode.pos_max[0]][inode.pos_max[1] - 1] != None:
                inode.izq = Nodo(inode.tablero, [inode.pos_max[0], inode.pos_max[1] - 1], inode.pos_min, not playertype, inode.data)
                aux_list.append(inode.izq)
    return aux_list

def init_tablero(nsize):
    tablero = [ ]
    tablero.append([None for _ in range(nsize + 2)])
    for _ in range(nsize):
        aux_tablero = [None]
        for _ in range(nsize):
            aux_tablero.append(nprandint(1, (2 * nsize) + 1))
        aux_tablero.append(None)
        tablero.append(aux_tablero)
    tablero.append([None for _ in range(nsize + 2)])
    return tablero

def print_table_no_none(tablero, init=False):
    aux_tabl = [ ]
    for i in range(1, game_size + 1):
        aux_list = [ ]
        for j in range(1, game_size + 1):
            if tablero[i][j] is None:
                aux_list.append('-')
            elif [i, j] == position_max:
                if init == True:
                    aux_list.append(tablero[i][j])
                else:
                    aux_list.append('c')
            elif [i, j] == position_min:
                if init == True:
                    aux_list.append(tablero[i][j])
                else:
                    aux_list.append('x')
            else:
                aux_list.append(tablero[i][j])
        aux_tabl.append(aux_list)
    print(nparr(aux_tabl))

print('-----------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------')
print('Bienvenido! Elige el tamanio de juego y nivel de dicultad!')

game_size = 0
bool_size = False
while not bool_size:
    game_size = input('Indique el tamanio del tablero: ')
    if not game_size.isdigit():
        print('Ingrese un valor entero!')
    else:
        bool_size = True
        game_size = int(game_size)

print('-----------------------------------------------------------------------------------------------------------')

game_level = 0
bool_leve = False
while not bool_leve:
    game_level = input('Indique la dificultad del huego: [D] Dificil, [M] Medio, [F] Facil: ').lower()
    if game_level not in ('d', 'm', 'f'):
        print('Seleccione el nivel de dificultad de una manera correcta!')
    else:
        bool_leve = True
        if game_level == 'd':
            game_level = (game_size * 2) - 2
        elif game_level == 'm':
            game_level = game_size
        else:
            game_level = 2
print('-----------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------')
print(game_size, game_level)
end_game = False
tablero = init_tablero(game_size)
score_global = -tablero[game_size][game_size]
position_max = [1, 1]
position_min = [game_size, game_size]
num_turnos = 0

print('-----------------------------------------------------------------------------------------------------------')
print('Empieza el Juego!')
print('Score: ', tablero[position_max[0]][position_max[1]] - tablero[position_min[0]][position_min[1]])
print('Tablero Inicial: ')
print_table_no_none(tablero, True)
print('-----------------------------------------------------------------------------------------------------------')

while not end_game:
    num_turnos += 1
    aux_prof = 1
    bool_player = False
    nodo_raiz = Nodo(tablero, position_max, position_min, True, score_global)
    nodo_raiz.tablero[position_min[0]][position_min[1]] = None
    score_global = tablero[position_max[0]][position_max[1]] - tablero[position_min[0]][position_min[1]]

    print('-----------------------------------------------------------------------------------------------------------')
    print('Turno #%s' % num_turnos)
    print('Score: ', score_global)
    print('Tablero Actual: ')
    print_table_no_none(tablero)
    print('-----------------------------------------------------------------------------------------------------------')

    tablero[position_min[0]][position_min[1]] = None
    tablero[position_max[0]][position_max[1]] = None

    aux_list  = [nodo_raiz]
    while aux_prof < game_level:
        aux_prof += 1
        aux_list = generar_arbol(aux_list, bool_player)
        bool_player = not bool_player

    print('hola')

    minmax_results = [ ]
    if nodo_raiz.izq is not None:
        ret_node, dir, alpha_val, beta_val = nodo_raiz.izq.minimax(True, game_level - 1)
        minmax_results.append([ret_node.data, 'Izquierda', [position_max[0], position_max[1] - 1]])
    if nodo_raiz.der is not None:
        ret_node, dir, alpha_val, beta_val = nodo_raiz.der.minimax(True, game_level - 1)
        minmax_results.append([ret_node.data, 'Derecha', [position_max[0], position_max[1] + 1]])
    if nodo_raiz.arr is not None:
        ret_node, dir, alpha_val, beta_val = nodo_raiz.arr.minimax(True, game_level - 1)
        minmax_results.append([ret_node.data, 'Arriba', [position_max[0] - 1, position_max[1]]])
    if nodo_raiz.aba is not None:
        ret_node, dir, alpha_val, beta_val = nodo_raiz.aba.minimax(True, game_level - 1)
        minmax_results.append([ret_node.data, 'Abajo', [position_max[0] + 1, position_max[1]]])

    if not minmax_results:
        print('No existen movimientos posibles para el jugador max! Fin del juego!')
        end_game = True
        break#i, a, i, i,

    minmax_results.sort()
    print('El jugador max hara el siguiente movimiento: %s!' % minmax_results[-1][1])
    print('-----------------------------------------------------------------------------------------------------------')
    position_max = minmax_results[-1][2]

    # Turno del jugador #2
    dict_player_two = {'i': [position_min[0], position_min[1] - 1], 'd': [position_min[0], position_min[1] + 1],
                       'a': [position_min[0] - 1, position_min[1]], 'b': [position_min[0] + 1, position_min[1]]}

    if tablero[dict_player_two['i'][0]][dict_player_two['i'][1]] is None and \
       tablero[dict_player_two['d'][0]][dict_player_two['d'][1]] is None and \
       tablero[dict_player_two['a'][0]][dict_player_two['a'][1]] is None and \
       tablero[dict_player_two['b'][0]][dict_player_two['b'][1]] is None:
        print('No existen movimientos posibles para el jugador min! Fin del juego!')
        end_game = True
        break
    print('Tu turno!')
    mov_jugador = ''
    bool_mov_jugador = False
    while not bool_mov_jugador:
        mov_jugador = input('Indique su movimiento! [A] Arriba, [B] Abajo, [I] Izquierda, [D] Derecha: ').lower()
        if mov_jugador not in ('a', 'b', 'i', 'd'):
            print('Seleccione el movimiento de manera correcta!')
        else:
            if tablero[dict_player_two[mov_jugador][0]][dict_player_two[mov_jugador][1]] is None:
                print('Movimiento no permitido!')
            elif dict_player_two[mov_jugador] == position_max:
                print('Movimiento no permitido! El jugador max acaba de ubicarse en esa posicion!')
            else:
                bool_mov_jugador = True

    if mov_jugador == 'a':
        print('Usted ha hecho el siguiente movimiento: Arriba!')
        position_min = dict_player_two['a']
    if mov_jugador == 'b':
        print('Usted ha hecho el siguiente movimiento: Abajo!')
        position_min = dict_player_two['b']
    if mov_jugador == 'i':
        print('Usted ha hecho el siguiente movimiento: Izquierda!')
        position_min = dict_player_two['i']
    if mov_jugador == 'd':
        print('Usted ha hecho el siguiente movimiento: Derecha!')
        position_min = dict_player_two['d']
    print('-----------------------------------------------------------------------------------------------------------')

print('-----------------------------------------------------------------------------------------------------------')
if score_global < 0:
    print('Felicidades! Usted ha ganado el juego! Wuuu!')
elif score_global > 0:
    print('La computadora ha ganado! Urra para los arboles MINIMAX! Wuuu!')
else:
    print('Juego empatado!')
print('-----------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------')
