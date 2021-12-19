# -*- coding: utf-8 -*-

import sympy as sp


def newton_raphson_multvar(eqstr, strsym, arrval, accuracy=0.0001):
    # Pre process the data
    arr_values, num_iter = arrval, 0
    tuple_sym = sp.symbols(strsym)
    equations = [sp.sympify(i.strip()) for i in eqstr.split(',')]
    # General data structures
    sympy_matrix = sp.Matrix(equations)
    inv_jacobian = sympy_matrix.jacobian(sp.Matrix(tuple_sym)).inv()
    # Initialization...
    arr_for_sust = [(tuple_sym[i], arr_values[i]) for i in range(len(tuple_sym))]
    f_xi = [i.subs(arr_for_sust).evalf() for i in equations]
    err = max(list(map(lambda x: abs(x), f_xi)))
    # Post process...
    while err > accuracy:
        num_iter += 1
        arr_for_sust = [(tuple_sym[i], arr_values[i]) for i in range(len(tuple_sym))]
        sust_jacobian = inv_jacobian.subs(arr_for_sust)
        f_xi = [i.subs(arr_for_sust).evalf() for i in equations]
        arr_values = list(sp.Matrix(arr_values) - sust_jacobian * sp.Matrix(f_xi))
        err = max(list(map(lambda x: abs(x), f_xi)))
    return list(map(lambda x: round(x, 5), arr_values)), num_iter


# Main...
str_expr = "x + y - z + 2, x ** 2 + y, z - y ** 2 - 1"
print(newton_raphson_multvar(str_expr, 'x,y,z', [1, 0, 1]))
