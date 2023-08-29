#Módulo 2
#Yaritzi Itzayana Nicio Nicolás A01745837


#Immportamos librerías
import numpy as np 
import pandas as pd 


#Definimos la función que calcula la distancia euclideana 
def distanciaEuclideana(x1, x2):
   a= np.array(x1)
   b= np.array(x2)
   matriz = a-b
   #La raíz cuadrada del producto punto de la transpuesta de la matriz y la matriz
   dist = np.sqrt(np.dot(matriz.T, matriz))
   return dist


distancias= [ ]
for i in range(len(X)):
    distancia= distanciaEuclideana(test, X.iloc[i])
    distancias.append(distancia)

distancias= np.array(distancias).sort_values()

indices= distancias.index[K]

y[indices]