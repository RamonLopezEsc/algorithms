#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ====================================================================== #
# Centro de Investigación Científica y de Educación Superior de Ensenada #
#               DEPARTAMENTO DE CIENCIAS DE LA COMPUTACIÓN               #
#                  Procesamiento de Imágenes Digitales                   #
#                                                                        #
# Generado por:                                                          #
# Ramón López Escudero                                                   #
#                                                                        #
# Fecha de Creación:                                                     #
# 14 - Noviembre - 2016                                                  #
#                                                                        #
# Descripción:                                                           #
# Código para ejecutar un 'Gaussian Highpass Filter'                     #
#                                                                        #
# Fuente:                                                                #
# Chityala, R. & Pudipeddi, S. (2014). Image Processing and Acquisition  #
# using Python. CRC Press: Estados Unidos de América                     #
#                                                                        #
# ====================================================================== #


import scipy.misc
import numpy as np, math
import scipy.fftpack as fftim
from scipy.misc.pilutil import Image

# Abriendo la imagen y conviertiéndola en escala de grises
a = Image.open(r'C:\Users\RamonLopez\Desktop\Einstein.jpg').convert('L')
# Conversión de 'a' en un arreglo de NumPy
b = np.asarray(a)
# Transformada de Fourier
c = fftim.fft2(b)
# Shifting de la imagen transformada en el dominio de las frecuencias
d = fftim.fftshift(c)
# Inicializando variables para la función de convolución
M = d.shape[0]
N = d.shape[1]

# Se define H y los valores en H son inicializados en 1
H = np.ones((M, N))
center_1 = M / 2
center_2 = N / 2

# Radio de corte
d_0 = 30.0
# Variable t_1
t_1 = d_0 * 2

# Definiendo la función de convolución para el filtro
for i in range(1, M):
    for j in range(1, N):
        r_1 = (i - center_1) ** 2 + (j - center_2) ** 2
        # Se calcula la distancia euclideana desde el origen
        r = math.sqrt(r_1)
        # Usando el radio de corte para elminiar altas frecuencias
        if 0 < r < d_0:
            H[i, j] = 1 - math.exp(-r ** 2 / t_1 ** 2)

# Conviertiendo H a imagen
H = scipy.misc.toimage(H)
# Ejecutando la convolución
conv = d * H
# Calculando la magnitud de la función inversas de Fourier
e = abs(fftim.ifft2(conv))
# Se convierte 'e' desde un arreglo NumPy a una imagen
f = scipy.misc.toimage(e)

# Display del filtro en el dominio de Fourier
H.show()
# Display de la imagen filtrada
f.show()

# Se guarda el filtro en una carpeta
#f.save(r'C:\Users\RamonLopez\Desktop\filtro.jpg')
# Se guarda la imagen en una carpeta
#f.save(r'C:\Users\RamonLopez\Desktop\grass3.jpg')