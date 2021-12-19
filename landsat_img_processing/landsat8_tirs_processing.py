#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================================================== #
# =============================================================================================================== #

# Importar Librerías
import time
import os
import numpy as np
import ogr
from osgeo import gdal

# =============================================================================================================== #
# =============================================================================================================== #

time_ini = time.time()

# =============================================================================================================== #
# =============================================================================================================== #

def lect_metadata(img_metadata):
    # Lectura de los Metadatos de la Imagen de Entrada para Extraer el Mes y Constantes de Corrección
    fo = open(img_metadata)

    for i in range(158):
        fo.readline()

    # Almacen de la Variable 'ml' que Servirá Como Parámetro de Corrección de la Imagen de Temperatura
    m_l = fo.readline()
    int_m_l = float(m_l[28] + m_l[29] + m_l[30] + m_l[31] + m_l[32])
    dec_m_l = float(m_l[35] + m_l[36] + m_l[37])

    m_l = int_m_l * (10 ** dec_m_l)

    for i in range (10):
        fo.readline()

    # Almacen de la Variable 'al' que Servirá Como Parámetro de Corrección de la Imagen de Temperatura
    a_l = fo.readline()
    a_l = float(a_l[28] + a_l[29] + a_l[30] + a_l[31] + a_l[32] + a_l[33])

    for i in range(21):
        fo.readline()

    # Almacen de la Variable 'k1' que Servirá Como Parámetro de Corrección de la Imagen de Temperatura
    k_1 = fo.readline()
    k_1 = float(k_1[26] + k_1[27] + k_1[28] + k_1[29] + k_1[30] + k_1[31] + k_1[32] + k_1[33])

    fo.readline()

    # Almacen de la Variable 'k2' que Servirá Como Parámetro de Corrección de la Imagen de Temperatura
    k_2 = fo.readline()
    k_2 = float(k_2[26] + k_2[27] + k_2[28] + k_2[29] + k_2[30] + k_2[31] + k_2[32] + k_2[33] + k_2[34])

    fo.close()

    return m_l, a_l, k_1, k_2

# =============================================================================================================== #
# =============================================================================================================== #

# Imagen de entrada
imput_band_path = r'D:\Img_Prueba\LC80280452015272LGN00_B10.tif'
input_band_gdal = gdal.Open(imput_band_path)

# Extracción de parámetros básicos de la imagen
proj = input_band_gdal.GetProjection()
n_col = input_band_gdal.RasterXSize
n_row = input_band_gdal.RasterYSize
n_band = input_band_gdal.RasterCount

print "----------------------------------"
print 'Proyeccion: ', proj
print 'Numero de columnas: ', n_col
print 'Numero de filas: ', n_row
print 'Número de bandas: ', n_band
print "----------------------------------"

# Conversión de la imagen en un objeto
geotransform_band = input_band_gdal.GetGeoTransform()

# Extracción de parámetros geográficos
coord_x = geotransform_band[0]
spa_res_x = geotransform_band[1]
rotation_1 = geotransform_band[2]
coord_y = geotransform_band[3]
rotation_2 = geotransform_band[4]
spa_res_y = geotransform_band[5]

print 'Coordenada izquierda superior en el eje X: ', coord_x
print 'Resolucion espacial en el eje X: ', spa_res_x
print 'Rotacion de la imagen (1): ', rotation_1
print 'Coordenada izquierda superior en el eje Y: ', coord_y
print 'Rotacion de la imagen (2): ', rotation_2
print 'Resolucion espacial en el eje Y: ', spa_res_y
print "----------------------------------"

# Conversión del objeto (imagen) en un arreglo de NumPy
in_usage_band = input_band_gdal.GetRasterBand(1)
data = in_usage_band.ReadAsArray().astype(np.float)

# Coordenadas del punto analizado
point_coord_x = 241286.0
point_coord_y = 2400113.0

# Posición en la matriz de los pixeles del punto analizado
pix_index_in_x = int((point_coord_x - coord_x) / spa_res_x)
pix_index_in_y = int((point_coord_y - coord_y) / spa_res_y)

# Extracción del valor del punto
point_value = data[pix_index_in_x, pix_index_in_y]

print 'Valor del punto analizado: ', point_value
print "----------------------------------"

# =============================================================================================================== #
# =============================================================================================================== #

# Recorte de un Raster a partir de una mascara en shapefile

# Ruta del fichero shape
mask_shape = r'D:\Shape_Prueba\Guanajuato.shp'

# Especificación de la extensión a leer
driver = ogr.GetDriverByName('ESRI Shapefile')

# Lectura del fichero shape --> 0 significa lectura, 1 escritura
dataSource = driver.Open(mask_shape, 0)

# Datos generales del ficher shape
active_layer = dataSource.GetLayer()
featureCount = active_layer.GetFeatureCount()
print "Número de registros en el fichero abierto: %d" % (featureCount)
print "----------------------------------"

# Lectura e impresión de las coordenadas de los polígonos contenidos en el fichero shape
print "Coordenadas de los vértices de los polígonos\n"
for i in range(featureCount):
    active_feature = active_layer.GetFeature(i)
    geometry = active_feature.GetGeometryRef()
    ring = geometry.GetGeometryRef(0) # <-- Leyendo la perimetral "más" externa
    num_vertex =  ring.GetPointCount()
    print "POLÍGONO #%i" % (i + 1)
    for j in range(num_vertex):
        print ring.GetPoint(j)
print "----------------------------------"

# Almacenamiento de coordenadas de los polígonos
list_pol_coord = []

for i in range(featureCount):
    active_feature = active_layer.GetFeature(i)
    geometry = active_feature.GetGeometryRef()
    ring = geometry.GetGeometryRef(0)
    num_vertex =  ring.GetPointCount()
    for j in range(num_vertex):
        list_pol_coord.append(ring.GetPoint(j))

# Ordenamiento y almacenamiento de coordenadas extremas
list_pol_coord.sort()
rx = str(int(list_pol_coord[len(list_pol_coord)-1][0]))
lx = str(int(list_pol_coord[0][0]))

list_pol_coord.sort(key = lambda x: x[1]) # <-- lambda = Función "anónima"; key = Parámetro de ordenamiento
ly = str(int(list_pol_coord[len(list_pol_coord)-1][1]))
ry = str(int(list_pol_coord[0][1]))

# Recorte de la imagen a partir de las coordenadas extremas de la máscara
clip_raster = r"D:\Resultados_Prueba\Clip_Img.tif"
poly_shape = r'D:\Shape_Prueba\Shape_Prueba.shp'
cmd = "gdal_translate -projwin %s %s %s %s %s %s" % (lx, ly, rx, ry, imput_band_path, clip_raster)
os.popen(cmd)

input_band_gdal = None

# =============================================================================================================== #
# =============================================================================================================== #

# Corrección radiométrica de la imagen de temperatura

# Asignacion de la ruta del fichero de metadatos
input_metadata = r'D:\Img_Prueba\LC80280452015272LGN00_MTL.txt'

# Lectura y almacenamiento de coeficientes de correccion
m_l, a_l, k_1, k_2 = lect_metadata(input_metadata)

# Lectura de la imagen recortada
clip_band_gdal = gdal.Open(clip_raster)

# Extracción de parámetros básicos de la imagen recortada
geotransform_band = clip_band_gdal.GetGeoTransform()
n_col = clip_band_gdal.RasterXSize
n_row = clip_band_gdal.RasterYSize

# Conversión de la imagen recortada en un objeto
in_usage_band = clip_band_gdal.GetRasterBand(1)

# Conversión del objeto (imagen recortada) en un arreglo de NumPy
data = in_usage_band.ReadAsArray().astype(np.float)

# Operaciones para realizar las correcciones radiometricas
for i in range(n_row):
    for j in range(n_col):
        # Calculo de Radiancia
        data[i][j] = (m_l * data[i][j]) + a_l
        # Cálculo de Temperatura a Partir de la Radiancia (K)
        data[i][j] = (k_2) / (np.log((k_1 / data[i][j]) + 1))
        # Cálculo de Temperatura a Partir de la Radiancia (C)
        data[i][j] = data[i][j] - 273.15

# Ruta de salida de la imagen corregida
output = r'D:\Resultados_Prueba\Img_Correg.tif'

# Parámetros para la creacion de la imagen corregida
fmt = 'GTiff' # <-- Formato GeoTIFF
driver = gdal.GetDriverByName(fmt)
dst_dataset = driver.Create(output, n_col, n_row, 1, gdal.GDT_Float32)
dst_dataset.SetGeoTransform(geotransform_band)
dst_dataset.SetProjection(proj)
dst_dataset.GetRasterBand(1).WriteArray(data)
dst_dataset = None

# =============================================================================================================== #
# =============================================================================================================== #

time_fin = time.time()
time_algor = time_fin - time_ini

print "Tiempo de ejecucion: ", time_algor
print "----------------------------------"

# =============================================================================================================== #
# =============================================================================================================== #
