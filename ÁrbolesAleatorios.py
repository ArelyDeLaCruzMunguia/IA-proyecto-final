import streamlit as st              # Para usar streamlit
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from PIL import Image
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import time
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def arboles(): 
    head = '<h1 style="font-family:Fantasy; color: #1a6563; font-size: 60px;text-align: center;">Árboles de Decisión</h1>'
    st.markdown(head, unsafe_allow_html=True)
    parrafo1='<p style=" color:black; font-size: 16px; text-align:justify;">Este es uno de los Algoritmos más utilizado en el Machine Learning y esta dentro del Aprendizaje Supervisado. Permite resolver problemas de Regresión y Clasificación, además es un algoritmo que admite tanto valores númericos como nominales. Su objetivo es construir una estructura jerárquica eficiente y escalable que divide a los datos en función de determinadas condiciones, por lo que utiliza la estrategia <strong>divide y vencerás.</strong></p>'
    st.markdown(parrafo1,unsafe_allow_html=True)
    parrafo2='<p style="color:black; font-size: 16px; text-align:justify;">Se conforma por los siguientes elementos <ul><li>Nodo raíz: Representa a todos los datos</li></ul></p>'
    st.markdown(parrafo2,unsafe_allow_html=True)
    st.header("Datos")
    st.markdown("Aqui se realiza la carga de tu archivo con extension .cvs")
    archivo_arboles= st.file_uploader("Archivo",key=5,type="csv") 
    if archivo_arboles is not None:
        datos=pd.read_csv(archivo_arboles)
        st.subheader("Tabla de datos")
        st.write(datos)
        st.header("Mapa de calor")
        plt.figure(figsize=(14,7))
        MatrizInf = np.triu(datos.corr())
        sns.heatmap(datos.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
        st.pyplot()
        variables_borrar_ar=st.multiselect('¿Qué variables deseas eliminar?',datos.columns.values)
        for borrar in variables_borrar_ar:
            datos=datos.drop(columns=[borrar])
        st.header('Datos sin variables eliminadas')
        st.write(datos)
        st.header('Elige tu variable clase')
        var_clase=st.selectbox('¿Qué variable desea seleccionar?',datos.columns.values)
        datos_sin_clase=datos.drop(columns=[var_clase])
        bandera_ar=np.size(datos_sin_clase)
        if bandera_ar:
            #Variables predictoras
            X = np.array(datos_sin_clase[datos_sin_clase.columns.values])
            #Variable clase
            Y = np.array(datos[[var_clase]])
            X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0,shuffle = True)
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0,shuffle = True)
            #Se entrena el modelo a partir de los datos de entrada
            clas_option1=st.selectbox("Árbol que deseas obtener:",('Pronóstico','Clasificación'))
            if clas_option1=="Pronóstico":
                clas_option=st.selectbox("Modo de entrenar al modelo",('Modo 1 con random_state=0','Modo 2 con parámetros de número hojas, muestras y niveles del árbol'))
                if clas_option=="Modo 1 con random_state=0":
                    PronosticoAD= DecisionTreeRegressor(random_state=0)
                    PronosticoAD.fit(X_train, Y_train)
                    #Se genera el pronóstico
                    Y_Pronostico = PronosticoAD.predict(X_test)
                    st.subheader("Score")
                    st.write(r2_score(Y_test, Y_Pronostico)*100)
                    print('Criterio: \n', PronosticoAD.criterion)
                    st.write('Importancia variables: \n', PronosticoAD.feature_importances_)
                    st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
                    st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
                    st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
                    ImportanciaMod2 = pd.DataFrame({'Variable': list(datos_sin_clase[datos_sin_clase.columns.values]),'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.header("Importancia de cada variable")
                    st.write(ImportanciaMod2)
                    pred=pd.DataFrame(columns=datos_sin_clase.columns.values)
                    for i in datos_sin_clase.columns.values:
                        st.write("Valor para la columna ",i)
                        val=st.number_input('Ingrese valor',value=0.1, step=0.1,key=i)
                        pred.at[0,i]=val
                            
                    #st.write(pred)
                    prediccion=PronosticoAD.predict(pred)
                    st.write("Predicción del modelo")
                    st.write(prediccion)
                if clas_option=="Modo 2 con parámetros de número hojas, muestras y niveles del árbol":
                    st.subheader("Valor de la máxima profundidad del árbol")
                    maxdepth=st.number_input('Valor de max_depth',value=10, step=1)
                    st.subheader("Valor de la cantidad mínima de datos para que un nodo de decisión se pueda dividir.")
                    split=st.number_input('Valor de split',value=4, step=1)
                    st.subheader("Valor de la cantidad mínima de datos que debe tener un nodo hoja.")
                    leaf=st.number_input('Valor de leaf',value=2, step=1)
                    st.subheader("Valor de random_state")
                    random=st.number_input('Valor de random_state',value=0, step=1)
                    PronosticoAD = DecisionTreeRegressor(max_depth=maxdepth, min_samples_split=split, min_samples_leaf=leaf,random_state=random)
                    PronosticoAD.fit(X_train, Y_train)
                    #Se genera el pronóstico
                    Y_Pronostico = PronosticoAD.predict(X_test)
                    st.subheader("Score")
                    st.write(r2_score(Y_test, Y_Pronostico))
                    print('Criterio: \n', PronosticoAD.criterion)
                    st.write('Importancia variables: \n', PronosticoAD.feature_importances_)
                    st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
                    st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
                    st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
                    ImportanciaMod2 = pd.DataFrame({'Variable': list(datos_sin_clase[datos_sin_clase.columns.values]),'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.header("Importancia de cada variable")
                    st.write(ImportanciaMod2)
                    pred=pd.DataFrame(columns=datos_sin_clase.columns.values)
                    for i in datos_sin_clase.columns.values:
                        st.write("Valor para la columna ",i)
                        val=st.number_input('Ingrese valor',value=0.1, step=0.1,key=i)
                        pred.at[0,i]=val
                            
                    #st.write(pred)
                    prediccion=PronosticoBA.predict(pred)
                    st.write("Predicción del modelo")
                    st.write(prediccion)

            if clas_option1=='Clasificación':
                clas_option=st.selectbox("Modo de entrenar al modelo",('Modo 1 con random_state de 0','Modo 2 con parámetros de número de árboles, hojas, muestras y niveles del árbol'))
                if clas_option=="Modo 1 con random_state de 0":
                    ClasificacionAD = DecisionTreeClassifier(random_state=0)
                    ClasificacionAD.fit(X_train, Y_train)
                    #Clasificación final 
                    Y_ClasificacionAD = ClasificacionAD.predict(X_validation)
                    ValoresMod2 = pd.DataFrame(Y_validation, Y_ClasificacionAD)
                    st.header("Score del modelo")
                    st.write("¿Qué tan fiable es el modelo? En porcentaje.")
                    st.text(accuracy_score(Y_validation, Y_ClasificacionAD)*100)
                    #Matriz de clasificación
                    ModeloClasificacion2 = ClasificacionAD.predict(X_validation)
                    Matriz_Clasificacion2 = pd.crosstab(Y_validation.ravel(),ModeloClasificacion2,rownames=['Reales'],colnames=['Clasificación'])
                    st.header("Matriz de clasificación")
                    st.write(Matriz_Clasificacion2)
                    #st.header("Reporte de la clasificación")
                    #Reporte de la clasificación
                    #st.write('Criterio: \n', ClasificacionBA.criterion)
                    #st.write("Exactitud:", accuracy_score(Y_validation, Y_ClasificacionBA))
                    #st.write(classification_report(Y_validation, Y_ClasificacionBA))
                    ImportanciaMod2 = pd.DataFrame({'Variable': list(datos_sin_clase[datos_sin_clase.columns.values]),'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.header("Importancia de cada variable")
                    st.write(ImportanciaMod2)
                    pred=pd.DataFrame(columns=datos_sin_clase.columns.values)
                    for i in datos_sin_clase.columns.values:
                        st.write("Valor para la columna ",i)
                        val=st.number_input('Ingrese valor',value=0.1, step=0.1,key=i)
                        pred.at[0,i]=val
                            
                    #st.write(pred)
                    prediccion=ClasificacionAD.predict(pred)
                    st.write("Predicción del modelo")
                    st.write(prediccion)
                if clas_option=="Modo 2 con parámetros de número de árboles, hojas, muestras y niveles del árbol":
                    st.subheader("Valor de la profundidad máxima del árbol")
                    maxdepth=st.number_input('Valor de max_depth',value=10, step=1)
                    st.subheader("Valor de la cantidad mínima de datos para que un nodo de decisión se pueda dividir.")
                    split=st.number_input('Valor de split',value=4, step=1)
                    st.subheader("Valor de la cantidad mínima de datos que debe tener un nodo hoja.")
                    leaf=st.number_input('Valor de leaf',value=2, step=1)
                    st.subheader("Valor de random_state")
                    random=st.number_input('Valor de random_state',value=0, step=1)
                    ClasificacionAD = RandomForestClassifier(max_depth=maxdepth, min_samples_split=split, min_samples_leaf=leaf,random_state=random)
                    ClasificacionAD.fit(X_train, Y_train)
                    #Clasificación final 
                    Y_ClasificacionAD = ClasificacionAD.predict(X_validation)
                    ValoresMod2 = pd.DataFrame(Y_validation, Y_ClasificacionAD)
                    st.header("Score del modelo")
                    st.write("¿Qué tan fiable es el modelo? En porcentaje")
                    st.text(accuracy_score(Y_validation, Y_ClasificacionAD)*100)
                    #Matriz de clasificación
                    ModeloClasificacion2 = ClasificacionBA.predict(X_validation)
                    Matriz_Clasificacion2 = pd.crosstab(Y_validation.ravel(), ModeloClasificacion2, rownames=['Reales'], colnames=['Clasificación']) 
                    st.header("Matriz de clasificación")
                    st.write(Matriz_Clasificacion2)
                    #st.header("Reporte de la clasificación")
                    #Reporte de la clasificación
                    #st.write('Criterio: \n', ClasificacionAD.criterion)
                    #st.write("Exactitud:", accuracy_score(Y_validation, Y_ClasificacionAD))
                    #st.write(classification_report(Y_validation, Y_ClasificacionAD))
                    ImportanciaMod2 = pd.DataFrame({'Variable': list(datos_sin_clase[datos_sin_clase.columns.values]),'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.header("Importancia de cada variable")
                    st.write(ImportanciaMod2)
                    pred=pd.DataFrame(columns=datos_sin_clase.columns.values)
                    for i in datos_sin_clase.columns.values:
                        st.write("Valor para la variable ",i)
                        val=st.number_input('Ingrese valor',value=0.1, step=0.1,key=i)
                        pred.at[0,i]=val
                            
                    #st.write(pred)
                    prediccion=ClasificacionAD.predict(pred)
                    st.write("Predicción del modelo")
                    st.write(prediccion)
