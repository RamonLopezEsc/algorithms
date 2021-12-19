# -*- coding: utf-8 -*-

import sympy as sp
from math import inf


def gradient_descent(eqstr: str, arrval: list, n_val: float, symstr: str, 
                     typeopt: str, order_accuracy: int = 3, round_val: int = 4) -> tuple:
    if typeopt == 'max':
        sign = 1
    elif typeopt == 'min':
        sign = -1
    else:
        raise ValueError('Not a valid optimization problem')
    
    tuple_sym = sp.symbols(symstr)
    if type(tuple_sym) != tuple:
        tuple_sym = (tuple_sym, )
    
    eq_vector = sp.Matrix([sp.sympify(eqstr)])
    jacobian_vector = eq_vector.jacobian(sp.Matrix(tuple_sym))
    
    num_var = len(tuple_sym)
    array_val, n_iter = arrval, 0
    err = [inf for _ in range(num_var)]
    dict_values = {tuple_sym[i]: array_val[i] for i in range(num_var)}
    f_x_subsval = [jacobian_vector[i].subs(dict_values) for i in range(num_var)]
    
    while max(err) > (1 * 10 ** -(order_accuracy - 1)):
        n_iter += 1
        array_val_temp = [round(array_val[i] + (sign * n_val * f_x_subsval[i]), round_val) 
                          for i in range(num_var)]
        dict_values = {tuple_sym[i]: array_val_temp[i] for i in range(num_var)}
        f_x_subsval = [jacobian_vector[i].subs(dict_values) for i in range(num_var)]
        err = [abs(array_val_temp[i] - array_val[i]) for i in range(num_var)]
        array_val = array_val_temp[:]
        print(array_val)
    
    f_x_subsval = [sp.sympify(eqstr).subs(dict_values) for _ in range(num_var)]
    return n_iter, array_val, f_x_subsval


str_expr = '(x ** 4) + (y ** 4) + (z ** 4) + (x * y * z)'
print(gradient_descent(str_expr, [1, 1, 1], 0.07, 'x,y,z', 'min', order_accuracy=3))
