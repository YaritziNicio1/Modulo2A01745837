#Módulo 2
#Yaritzi Itzayana Nicio Nicolás A01745837
#Modelo a implementar KNN

#Importamos librerías
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#Se lee el dataset de Iris
df=pd.read_csv("Iris.csv")
#Quitamos la columna del Id ya que no nos ayudará en el modelo 
df.drop("Id", axis=1, inplace=True)


#Se cambian los valores string por valores numéricos para que pueda funcionar el modelo
#Se utiliza un diccionario donde cada una de las variables se cambia por un valor entre 1 y 3
df['Species']=df['Species'].replace({"Iris-setosa":1,"Iris-versicolor":2,"Iris-virginica":3})
X=df.drop("Species", axis = 1) #Variables de entrada
y=df['Species'] #Variable de salida

#Dividir los datos para test y train utilizando Train test split, siendo un tamaño del test del 20% y un train del 80% del total dde los datos
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=32)


#Definimos la función que calcula la distancia euclideana entre dos puntos
    #Distancia euclideana entre dos puntos en el espacio 
def distanciaEuclideana(x1, x2):
   #La raíz cuadrada de la suma de la resta de dos puntos al cuadrdo
   dist = np.sqrt(np.sum((x1 - x2) ** 2))
   #Regresa el valor calculado anteriormente
   return dist


#Definimos la función que nos dará la predicción deseada
    #Se utilizan los valores del data set de train y se coloca un valor a k 
def prediccion(X_train, y_train, X_test, k=2):
    #Se crea una lista donde se almacenarán los valores predecidos del modelo 
    y_pred=[]
    #Iniciamos un ciclo for que itera el cálculo de la distancia euclideana
    for x in range(len(X_test)):
        x_test = X_test.iloc[x] 
        #Se crea la lista donde se guardaran las distancias 
        distances=[]
        for y in range(len(X_train)):
            x_train = X_train.iloc[y]
            #Guarda las distancias calculadas de x_train y x_test en la lista de distances 
            distances.append(distanciaEuclideana(x_train,x_test))
        #Se obtienen los índices de los puntos más cercanos
        k_indices = np.argsort(distances)[:k]
        #Se obtienen las etiquetas de los k puntos más cercanos 
        k_nearest_labels = y_train.iloc[list(k_indices)]
        #Obtenemos la etiqueta más común calculada  entre los k puntos más cercanos 
        most_common = np.bincount(k_nearest_labels).argmax()
        #se guardan en la lista de predicciones
        y_pred.append(most_common)
    #Regresa el valor de y_pred    
    return np.array(y_pred) 


#Definimos la función que nos calculará el accuracy del modelo 
def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    #Calculo de las predicciones correctas entre el total de predicciones
    accuracy = correct_predictions / total_predictions
    #Regresa el calculo anterior 
    return accuracy

#Se manda a llamar a la función  que predice con los datos de X_train, y_train y X_test con 3 vecinos más cercanos (el valor de k)
predictions = prediccion(X_train, y_train, X_test, k=3) #El valor de k está definido por las variables que 
#Se manda a llamar a la función correspondiente para el cálculo de accuracy 
acc = accuracy(y_test, predictions)
#Imprime ambos llamados anteriormente mencionados 
print("Predicción:", predictions[0])
print(predictions)
print("Precisión:", acc)
#se hace una copia de los valores que se tienen en el data set de X_test
df_pred = X_test.copy()
#Hacemos que el valor de la columna Species del data set anterior sea predictions en lugar del test 
df_pred["Species"]=predictions
#Graficamos los valores predecidos anteriormente con los valores de entrada 
plt.scatter(x=df_pred["SepalLengthCm"],y=df_pred["SepalWidthCm"], c= df_pred["Species"])
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.show()