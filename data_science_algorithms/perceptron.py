import numpy as np
from math import exp, inf
from random import uniform, seed

seed(112358)


def backward(desired_y, y_vector, w_o_matrix):
    out_err = (np.transpose(desired_y) - y_vector) * y_vector * (1 - y_vector)
    backpropagation = np.transpose(np.matmul(np.transpose(out_err), w_o_matrix))
    hid_err = y_hidden * (1 - y_hidden) * backpropagation
    return hid_err, out_err


def forward(xvector, w_h_matrix, w_o_matrix):
    net_h = np.matmul(w_h_matrix, np.transpose(xvector))
    y_h = sigmoid_function(net_h)
    net_o = np.matmul(w_o_matrix, y_h)
    return sigmoid_function(net_o), y_h


def sigmoid_function(xvector, aval=1):
    return_vector = [[1.0 / (1 + exp(-aval * value))] for value in xvector]
    return np.array(return_vector)


def update_err(outerr):
    return np.mean(outerr ** 2)


# Caso de uso para una matriz X y Y...
y_matrix = np.array([[0, 0], [1, 1], [1, 1],
                     [0, 1], [1, 0], [1, 0]])
x_matrix = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0],
                     [0, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0]])

alpha = 0.1
act_err = inf
num_patterns = len(x_matrix)
num_inputs = len(x_matrix[0])
num_outputs = len(y_matrix[0])
num_hidden_layers = num_inputs * num_outputs

w_h = np.array([[uniform(0, 1) for _ in range(num_inputs)] for _ in range(num_hidden_layers)])
w_o = np.array([[uniform(0, 1) for _ in range(num_hidden_layers)] for _ in range(num_outputs)])

while act_err > 0.000001:
    for element in range(num_patterns):
        y, y_hidden = forward(np.array([x_matrix[element]]), w_h, w_o)
        hidden_err, output_err = backward(np.array([y_matrix[element]]), y, w_o)
        w_o += alpha * np.matmul(output_err, np.transpose(y_hidden))
        w_h += alpha * np.matmul(hidden_err, np.array([x_matrix[element]]))
        act_err = update_err(output_err)

print('-----------------------------------------')
print('Hidden weights')
print(w_h)
print('-----------------------------------------')
print('Output weights')
print(w_o)
print('-----------------------------------------')
print('Resultados:\n')

for element in x_matrix:
    y_result, _ = forward(element, w_h, w_o)
    print(y_result, '\n')
