# -*- coding: utf-8 -*-

import sympy as sp
from math import inf


def aprox_by_taylor(xsympy, sympyeq, numderiv, avalue, xvalue=None, interval=(-inf, inf)):
    sum_taylor = (sp.diff(sympyeq, xsympy, 0).subs(xsympy, avalue) 
                  / sp.factorial(0)) * ((xsympy - avalue).subs(xsympy, xvalue) ** 0)
    
    for deriv in range(1, numderiv + 1):
        sum_taylor += (sp.diff(sympyeq, xsympy, deriv).subs(xsympy, avalue) 
                       / sp.factorial(deriv)) * ((xsympy - avalue).subs(xsympy, xvalue) ** deriv)
        
        if xvalue is not None:
            next_deriv = sp.diff(sympyeq, xsympy, numderiv + 1)
            
            if sp.is_decreasing(next_deriv.as_independent(x)[1], 
                                sp.Interval.open(interval[0], interval[1])):
                err_bound = abs((next_deriv.subs(xsympy, interval[0]) / 
                                 sp.factorial(numderiv + 1)) * ((xvalue - avalue) ** (numderiv + 1)))
            else:
                err_bound = abs((next_deriv.subs(xsympy, interval[1]) / 
                                 sp.factorial(numderiv + 1)) * ((xvalue - avalue) ** (numderiv + 1)))
                error = abs(sympyeq.subs(xsympy, xvalue) - sum_taylor)
        else:
            error = sp.nan
            err_bound = sp.nan
    
    # Print results...
    print('-------------------------------')
    print('Aprox Value: {}'.format(sum_taylor.evalf()))
    print('Err Bound: {}'.format(err_bound))
    print('Error: {}'.format(error))
    return sum_taylor


x = sp.symbols('x')
f_x = sp.sin(1 / x)
list_deriv = [2, 5, 8, 14, 20]

name_colors = ['red', 'blue', 'green', 'black', 'purple']
plot = sp.plot(show=False)

for counter, element in enumerate(list_deriv):
    plot.extend(sp.plot(aprox_by_taylor(x, f_x, numderiv=element, avalue=1), 
                        (x, 0, 1), show=False))
    plot[counter].line_color = name_colors[counter]
plot.show()
