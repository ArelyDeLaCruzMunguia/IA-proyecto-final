import streamlit as st              # Para usar streamlit
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from PIL import Image
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def bosques():
    head = '<h1 style="font-family:Fantasy; color: #1a6563; font-size: 60px;text-align: center;">Bosques Aleatorios</h1>'
    st.markdown(head, unsafe_allow_html=True)
 
    texto1='<p style=" color:black; font-size: 16px; text-align:justify;">Este algoritmo pertenece al Aprendizaje Supervisado de Machine Learning, es capaz de resolver problemas de Clasificación (Regresión) y Prónostico. Este Algoritmo surgió a partir de un problema presentado en el algoritmo de Árboles de Decisión, ya que se presenta el problema de sobreajuste lo que quiere decir que aunque aprende muy bien de los datos, la generalización no llega a ser tan buena, por lo que el combinar varios árboles puede resultar mucho mejor y hace una mejor generalización, además trabaja con datos númericos y discretos.</p>'
    st.markdown(texto1,unsafe_allow_html=True)
    texto2='<p style=" color:black; font-size: 16px; text-align:justify;">Su principal objetivo es construir un conjunto de Árboles de Decisión combinados, estos árboles ven distintas porciones de los datos, es decir, ningún árbol ve todos los datos de entrenamiento, sino cada uno se entrena con distintas muestras para un mismo problema. De esta forma al combinar los resultados, los errores se compensan con otros y se tiene un mejor pronóstico o clasificación que generaliza de mejor manera al problema.</p>'
    st.markdown(texto2,unsafe_allow_html=True)
    columna1,columna2=st.columns(2)
    with columna1:
        tit3= '<h1 style="font-family:Fantasy; color:black; font-size: 35px;text-align: center;">Pronóstico</h1>'
        st.markdown(tit3,unsafe_allow_html=True)
        texto3='<p style=" color:black; font-size: 16px; text-align:justify;"></p>'
        st.markdown(texto3,unsafe_allow_html=True)
 
    st.header("Datos")
    st.markdown("Aqui se realiza la carga de tu archivo con extension .cvs")
    archivo_bosque_1=st.file_uploader("Sube tu archivo",type="csv",key=4)
    if archivo_bosque_1 is not None:
        #datos=archivo.getvalue()
        datos=pd.read_csv(archivo_bosque_1)
        st.header("Tabla de Datos")
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
            clas_option1=st.selectbox("Bosque que deseas obtener:",('Pronóstico','Clasificación'))
            if clas_option1=="Pronóstico":
                st.header("Pronóstico")
                clas_option=st.selectbox("Modo de entrenar al modelo",('Modo 1 sin número de árboles','Modo 2 con parámetros de número de árboles, hojas, muestras y niveles del árbol'))
                if clas_option=="Modo 1 sin número de árboles":
                    st.subheader("Valor de la cantidad de árboles en el bosque aleatorio.")
                    st.subheader("Valor de la máxima profundidad del árbol")
                    maxdepth=st.number_input('Valor de max_depth',value=10, step=1)
                    st.subheader("Valor de la cantidad mínima de datos para que un nodo de decisión se pueda dividir.")
                    split=st.number_input('Valor de split',value=4, step=1)
                    st.subheader("Valor de la cantidad mínima de datos que debe tener un nodo hoja.")
                    leaf=st.number_input('Valor de leaf',value=2, step=1)
                    st.subheader("Valor de random_state")
                    random=st.number_input('Valor de random_state',value=0, step=1)
                    PronosticoBA = RandomForestRegressor(max_depth=maxdepth, min_samples_split=split, min_samples_leaf=leaf,random_state=random)
                    PronosticoBA.fit(X_train, Y_train)
                    #Se genera el pronóstico
                    Y_Pronostico = PronosticoBA.predict(X_test)
                    st.subheader("Score en porcentaje")
                    st.write(r2_score(Y_test, Y_Pronostico)*100)
                    print('Criterio: \n', PronosticoBA.criterion)
                    st.write('Importancia variables: \n', PronosticoBA.feature_importances_)
                    st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
                    st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
                    st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
                    ImportanciaMod2 = pd.DataFrame({'Variable': list(datos_sin_clase[datos_sin_clase.columns.values]),'Importancia': PronosticoBA.feature_importances_}).sort_values('Importancia', ascending=False)
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
                if clas_option=="Modo 2 con parámetros de número de árboles, hojas, muestras y niveles del árbol":
                    st.subheader("Valor de la cantidad de árboles en el bosque aleatorio.")
                    estimators=st.number_input('Valor de n_estimators',value=105, step=1)
                    st.subheader("Valor de la máxima profundidad del árbol")
                    maxdepth=st.number_input('Valor de max_depth',value=10, step=1)
                    st.subheader("Valor de la cantidad mínima de datos para que un nodo de decisión se pueda dividir.")
                    split=st.number_input('Valor de split',value=4, step=1)
                    st.subheader("Valor de la cantidad mínima de datos que debe tener un nodo hoja.")
                    leaf=st.number_input('Valor de leaf',value=2, step=1)
                    st.subheader("Valor de random_state")
                    random=st.number_input('Valor de random_state',value=0, step=1)
                    PronosticoBA = RandomForestRegressor(n_estimators=estimators,max_depth=maxdepth, min_samples_split=split, min_samples_leaf=leaf,random_state=random)
                    PronosticoBA.fit(X_train, Y_train)
                    #Se genera el pronóstico
                    Y_Pronostico = PronosticoBA.predict(X_test)
                    st.subheader("Score en porcentaje")
                    st.write(r2_score(Y_test, Y_Pronostico)*100)
                    print('Criterio: \n', PronosticoBA.criterion)
                    st.write('Importancia variables: \n', PronosticoBA.feature_importances_)
                    st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
                    st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
                    st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
                    ImportanciaMod2 = pd.DataFrame({'Variable': list(datos_sin_clase[datos_sin_clase.columns.values]),'Importancia': PronosticoBA.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.header("Importancia de cada variable")
                    st.write(ImportanciaMod2)
                    pred=pd.DataFrame(columns=datos_sin_clase.columns.values)
                    for i in datos_sin_clase.columns.values:
                        st.write("Valor para la variable ",i)
                        val=st.number_input('Ingrese valor',value=0.1, step=0.1,key=i)
                        pred.at[0,i]=val
                            
                    #st.write(pred)
                    prediccion=PronosticoBA.predict(pred)
                    st.write("Predicción del modelo")
                    st.write(prediccion)

            if clas_option1=='Clasificación':
                st.header("Clasificación")
                clas_option=st.selectbox("Modo de entrenar al modelo",('Modo 1 con random_state de 0','Modo 2 con parámetros de número de árboles, hojas, muestras y niveles del árbol'))
                if clas_option=="Modo 1 con random_state de 0":
                    ClasificacionBA = RandomForestClassifier(random_state=0)
                    ClasificacionBA.fit(X_train, Y_train)
                    #Clasificación final 
                    Y_ClasificacionBA = ClasificacionBA.predict(X_validation)
                    ValoresMod2 = pd.DataFrame(Y_validation, Y_ClasificacionBA)
                    st.header("Score del modelo")
                    st.write("¿Qué tan fiable es el modelo? En porcentaje.")
                    st.text(accuracy_score(Y_validation, Y_ClasificacionBA)*100)
                    #Matriz de clasificación
                    ModeloClasificacion2 = ClasificacionBA.predict(X_validation)
                    Matriz_Clasificacion2 = pd.crosstab(Y_validation.ravel(),ModeloClasificacion2,rownames=['Reales'],colnames=['Clasificación'])
                    st.header("Matriz de clasificación")
                    st.write(Matriz_Clasificacion2)
                    #st.header("Reporte de la clasificación")
                    #Reporte de la clasificación
                    #st.write('Criterio: \n', ClasificacionBA.criterion)
                    #st.write("Exactitud:", accuracy_score(Y_validation, Y_ClasificacionBA))
                    #st.write(classification_report(Y_validation, Y_ClasificacionBA))
                    ImportanciaMod2 = pd.DataFrame({'Variable': list(datos_sin_clase[datos_sin_clase.columns.values]),'Importancia': ClasificacionBA.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.header("Importancia de cada variable")
                    st.write(ImportanciaMod2)
                    pred=pd.DataFrame(columns=datos_sin_clase.columns.values)
                    for i in datos_sin_clase.columns.values:
                        st.write("Valor para la columna ",i)
                        val=st.number_input('Ingrese valor',value=0.1, step=0.1,key=i)
                        pred.at[0,i]=val
                            
                    #st.write(pred)
                    prediccion=ClasificacionBA.predict(pred)
                    st.write("Predicción del modelo")
                    st.write(prediccion)
                if clas_option=="Modo 2 con parámetros de número de árboles, hojas, muestras y niveles del árbol":
                    st.subheader("Valor de la cantidad de árboles en el bosque aleatorio.")
                    estimators=st.number_input('Valor de n_estimators',value=105, step=1)
                    st.subheader("Valor de la máxima profundidad del árbol")
                    maxdepth=st.number_input('Valor de max_depth',value=10, step=1)
                    st.subheader("Valor de la cantidad mínima de datos para que un nodo de decisión se pueda dividir.")
                    split=st.number_input('Valor de split',value=4, step=1)
                    st.subheader("Valor de la cantidad mínima de datos que debe tener un nodo hoja.")
                    leaf=st.number_input('Valor de leaf',value=2, step=1)
                    st.subheader("Valor de random_state")
                    random=st.number_input('Valor de random_state',value=0, step=1)
                    ClasificacionBA = RandomForestClassifier(n_estimators=estimators,max_depth=maxdepth, min_samples_split=split, min_samples_leaf=leaf,random_state=random)
                    ClasificacionBA.fit(X_train, Y_train)
                    #Clasificación final 
                    Y_ClasificacionBA = ClasificacionBA.predict(X_validation)
                    ValoresMod2 = pd.DataFrame(Y_validation, Y_ClasificacionBA)
                    st.header("Score del modelo")
                    st.write("¿Qué tan fiable es el modelo? En porcentaje")
                    st.text(accuracy_score(Y_validation, Y_ClasificacionBA)*100)
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
                    ImportanciaMod2 = pd.DataFrame({'Variable': list(datos_sin_clase[datos_sin_clase.columns.values]),'Importancia': ClasificacionBA.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.header("Importancia de cada variable")
                    st.write(ImportanciaMod2)
                    pred=pd.DataFrame(columns=datos_sin_clase.columns.values)
                    for i in datos_sin_clase.columns.values:
                        st.write("Valor para la variable ",i)
                        val=st.number_input('Ingrese valor',value=0.1, step=0.1,key=i)
                        pred.at[0,i]=val
                            
                    #st.write(pred)
                    prediccion=ClasificacionBA.predict(pred)
                    st.write("Predicción del modelo")
                    st.write(prediccion)