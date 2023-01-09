import streamlit as st              # Para usar streamlit
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from PIL import Image
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay
import time
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def clasificacionLogistica():
    column11,column12=st.columns([1,3])
    with column11:
        lottie_hola_url="https://assets8.lottiefiles.com/packages/lf20_f7wf1y3d.json"
        lottie_hola=load_lottieurl(lottie_hola_url)
        st_lottie(lottie_hola,key="holaS")
    with column12:
        head = '<h1 style="font-family:Fantasy; color: #1a6563; font-size: 60px;text-align: center;">Clasificación Logística</h1>'
        st.markdown(head, unsafe_allow_html=True)
    colum13,column135=st.columns([3,1])
    with colum13:
        texto1='<p style=" color:black; font-size: 16px; text-align:justify;">Este algoritmo pertenece al Aprendizaje Supervisado de Machine Learning en Clasificación de datos. El objetivo principal es predecir etiquetas de una o más clases de tipo discretas <strong>(0,1,2)</strong> o nominales <strong>(A, B, C; o positivo, negativo; y otros).</strong>Con este algoritmo se construye un modelo a través de un conjunto de datos de entrenamiento<strong>(training).</strong> Los datos de entrenamiento son obtenidos de una fracción de los datos originales, la otra fraccción restante se conoce como datos de validación, apartir de estos se puede comprobar la precisión del modelo.</p>'
        st.markdown(texto1,unsafe_allow_html=True)
    with column135:
        lottie_logic_url="https://assets4.lottiefiles.com/packages/lf20_ym0gbugo.json"
        lottie_logic=load_lottieurl(lottie_logic_url)
        st_lottie(lottie_logic,key="logic")

    title = '<h1 style="font-family:Fantasy; color: #1a6563; font-size: 40px;text-align: center;">Regresión Logística</h1>'
    st.markdown(title, unsafe_allow_html=True)

    texto2='<p style"color:blakc; font-size: 16px; text-align:justify;">La regresión logistica es una transformación de la regresión lineal, ya que la respuesta obtenida se transforma en una probabilidad, para esto se utiliza una función logistica conocida como <strong>Sigmoide.</strong> La siguiente ecuación representa esta función logistica Sigmoide donde X es el modelo lineal:'
    st.markdown(texto2,unsafe_allow_html=True)

    imagen=Image.open('regresion.png')
    st.image(imagen,output_format="PNG")

    imagen=Image.open('sigmoide.jpg')
    st.image(imagen,output_format="JPG")
    st.subheader("Seleción de clasifficación")
    clases=st.selectbox("Tipo de clasificación", ('Dos clases','Tres o más clases'))
    if clases=="Dos clases":
        st.header("Datos")
        st.markdown("Aqui se realiza la carga de tu archivo con extension .cvs")
        upload= st.file_uploader("Archivo",key=3,type="csv") 
        if upload is not None:
            datos=pd.read_csv(upload)
            st.subheader("Tabla de datos")
            st.write(datos)
            st.header("Selección de variables")
            Corr= datos.corr(method='pearson')
            st.subheader("Matriz de correlaciones")
            st.write(Corr)
            st.subheader("Mapa de calor")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.figure(figsize=(14,7))
            MatrizInf = np.triu(Corr)
            sns.heatmap(Corr, cmap='RdBu_r', annot=True, mask=MatrizInf)
            st.pyplot()
            variable_clase=st.selectbox('¿Cuál es la variable clase?',datos.columns.values)
            for col in datos.columns:
                datatype=datos.dtypes[col]
                if datatype=='object' and col!=variable_clase:
                    datos=datos.drop(columns=[col])
            datos_copia=datos.copy()
            datos_copia=datos_copia.drop(columns=[variable_clase])
            variables_sel=st.multiselect('Variables predictoras',datos_copia.columns.values)
            X = np.array(datos[variables_sel])
            listo=False
            pd.DataFrame(X)
            bandera=np.size(X)
            if bandera: 
                st.header('Datos con variables predictoras a utilizar para el modelo')
                st.write(X)
                datatype_clase=datos[variable_clase].dtype
                if datatype_clase=='object':
                    var_clase1=st.text_input('Ingresa primera etiqueta. Esta tendrá el valor de 0.')
                    si_existe1=False
                    for indez, row in datos.iterrows():
                        if row[variable_clase]==var_clase1:
                            si_existe1=True
                    if si_existe1==False:
                        st.error('La etiqueta no se encuentra o no existe. ')
                        listo=False
                    var_clase2=st.text_input('Ingresa segunda etiqueta. Esta tendrá el valor de 1.')
                    si_existe2=False
                    for indez, row in datos.iterrows():
                        if row[variable_clase]==var_clase2:
                            si_existe2=True
                    if si_existe2==False:
                        st.error('La etiqueta no se encuentra o no existe.')
                        listo=False
                    if si_existe1==True and si_existe2==True:
                        datos=datos.replace({var_clase1: 0, var_clase2: 1})
                        st.write(datos[variable_clase])
                        listo=True
                elif datatype=='int64' or datatype=='float64':
                    if datos[variable_clase].max()==1:
                        if datos[variable_clase].min()==0:
                            listo=True
                else:
                    st.error("Tipo de dato en la variable clase inválido")
                    listo=False
                if listo==True:
                    #Variable clase
                    Y = np.array(datos[[variable_clase]])
                    pd.DataFrame
                    testsize=st.slider("Porcentaje de datos de prueba para el modelo (%):",20,30,20)
                    testsize=testsize/100
                    random_state_input=st.number_input("Valor de random_state:",min_value=0,step=1,value=0)
                    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                    test_size = testsize, 
                                                                                    random_state = random_state_input,
                                                                                    shuffle = True)
                    ClasificacionRL = linear_model.LogisticRegression()
                    ClasificacionRL.fit(X_train, Y_train)
                    #Predicciones probabilísticas
                    Probabilidad = ClasificacionRL.predict_proba(X_validation)
                    pd.DataFrame(Probabilidad)
                    st.header("Probabilidad de cada elemento (en porcentaje)")
                    st.write(Probabilidad)
                    #Clasificación final 
                    Y_ClasificacionRL = ClasificacionRL.predict(X_validation)
                    st.subheader("Score del modelo")
                    st.write(accuracy_score(Y_validation, Y_ClasificacionRL)*100)
                    st.header("Validación del modelo")
                    #Matriz de clasificación
                    ModeloClasificacion = ClasificacionRL.predict(X_validation)
                    Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                                       ModeloClasificacion, 
                                                       rownames=['Reales'], 
                                                       colnames=['Clasificación']) 
                    st.subheader("Matriz de clasificación")
                    st.write(Matriz_Clasificacion)
                    #Reporte de la clasificación
                    #print("Exactitud:", accuracy_score(Y_validation, Y_ClasificacionRL))
                    #print(classification_report(Y_validation, Y_ClasificacionRL))
                    CurvaROC = RocCurveDisplay.from_estimator(ClasificacionRL, X_validation, Y_validation, name="Predicción")
                    st.pyplot()
                    st.write("Intercept:", ClasificacionRL.intercept_)
                    st.write('Coeficientes: \n', ClasificacionRL.coef_)
                    st.header("Predicción")
                    st.text("Ingresa los valores de las variables para predecir a la variable clase")
                    pred=pd.DataFrame(columns=variables_sel)
                    for i in pred.columns.values:
                        st.write("Valor para la columna ",i)
                        val=st.number_input('Ingrese valor',value=0.1, step=0.1,key=i)
                        pred.at[0,i]=val
                        
                    #st.write(pred)
                    prediccion=ClasificacionRL.predict(pred)
                    st.write("Predicción del modelo")
                    st.write(prediccion)
    elif clases=="Tres o más clases":
        st.header("Datos")
        st.markdown("Aqui se realiza la carga de tu archivo con extension .cvs")
        upload= st.file_uploader("Archivo",key=3,type="csv") 
        if upload is not None:
            datos=pd.read_csv(upload)
            st.subheader("Tabla de datos")
            st.write(datos)
            st.header("Selección de variables")
            Corr= datos.corr(method='pearson')
            st.subheader("Matriz de correlaciones")
            st.write(Corr)
            st.subheader("Mapa de calor")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.figure(figsize=(14,7))
            MatrizInf = np.triu(Corr)
            sns.heatmap(Corr, cmap='RdBu_r', annot=True, mask=MatrizInf)
            st.pyplot()
            variable_clase=st.selectbox('¿Cuál es la variable clase?',datos.columns.values)
            for col in datos.columns:
                datatype=datos.dtypes[col]
                if datatype=='object' and col!=variable_clase:
                    datos=datos.drop(columns=[col])
            datos_copia=datos.copy()
            datos_copia=datos_copia.drop(columns=[variable_clase])
            variables_sel=st.multiselect('Variables predictoras',datos_copia.columns.values)
            X = np.array(datos[variables_sel])
            pd.DataFrame(X)
            listo=False
            bandera=np.size(X)
            if bandera: 
                st.header('Datos con variables predictoras a utilizar para el modelo')
                st.write(X)
                #estandarizar = StandardScaler()                                # Se instancia el objeto StandardScaler o MinMaxScaler 
                #MEstandarizada = estandarizar.fit_transform(MatrizVariables)   # Sescalan los datos
                #pd.DataFrame(MEstandarizada)
                #st.header('Datos Estandarizados')
                #st.write(MEstandarizada)
                
                datatype_clase=datos[variable_clase].dtype
                if datatype_clase=='object':
                    cant_clases=st.number_input("Ingresa número de etiquetas de tu variable clase.",min_value=3,value=3,step=1)
                    st.markdown("Nota: Comenzará a reemplazar la primera etiqueta por 0, la segunda por un 1, y así sucesivamente.")
                    reemplazo_etiqueta=0
                    key_text=12345
                    band=True
                    while cant_clases>0 and band:
                        var_clase1=st.text_input('Ingresa el valor de la etiqueta',key=key_text)
                        key_text=key_text+1
                        si_existe1=False
                        for indez, row in datos.iterrows():
                            if row[variable_clase]==var_clase1:
                                si_existe1=True
                                band=True
                        if si_existe1==False:
                            st.error('Etiqueta no encontrada en los datos de la variable clase')
                            listo=False
                            band=False
                        if si_existe1==True:
                            datos=datos.replace({var_clase1: reemplazo_etiqueta})
                            reemplazo_etiqueta=reemplazo_etiqueta+1
                            band=True
                            cant_clases=cant_clases-1
                    if band and cant_clases==0:
                        listo=True
                        st.write(datos[variable_clase])
                elif datatype=='int64' or datatype=='float64':
                    listo=True
                else:
                    st.error("Tipo de dato en la variable clase inválido")
                    listo=False
                if listo==True:
                    #Variable clase
                    Y = np.array(datos[[variable_clase]])
                    pd.DataFrame
                    testsize=st.slider("Porcentaje de datos de prueba para el modelo (%):",20,30,20)
                    testsize=testsize/100
                    random_state_input=st.number_input("Valor de random_state:",min_value=0,step=1,value=0)
                    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                    test_size = testsize, 
                                                                                    random_state = random_state_input,
                                                                                    shuffle = True)
                    ClasificacionRL = linear_model.LogisticRegression()
                    ClasificacionRL.fit(X_train, Y_train)
                    #Predicciones probabilísticas
                    Probabilidad = ClasificacionRL.predict_proba(X_validation)
                    pd.DataFrame(Probabilidad)
                    st.header("Probabilidad de cada elemento (en porcentaje)")
                    st.write(Probabilidad)
                    #Clasificación final 
                    Y_ClasificacionRL = ClasificacionRL.predict(X_validation)
                    st.subheader("Score del modelo")
                    st.write(accuracy_score(Y_validation, Y_ClasificacionRL)*100)
                    st.header("Validación del modelo")
                    #Matriz de clasificación
                    ModeloClasificacion = ClasificacionRL.predict(X_validation)
                    Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                                       ModeloClasificacion, 
                                                       rownames=['Reales'], 
                                                       colnames=['Clasificación']) 
                    st.subheader("Matriz de clasificación")
                    st.write(Matriz_Clasificacion)
                    #Reporte de la clasificación
                    #print("Exactitud:", accuracy_score(Y_validation, Y_ClasificacionRL))
                    #print(classification_report(Y_validation, Y_ClasificacionRL))
                    st.write("Intercept:", ClasificacionRL.intercept_)
                    st.write('Coeficientes: \n', ClasificacionRL.coef_)
                    st.header("Predicción")
                    st.text("Ingresa los valores de las variables para predecir a la variable clase")
                    pred=pd.DataFrame(columns=variables_sel)
                    for i in pred.columns.values:
                        st.write("Valor para la columna ",i)
                        val=st.number_input('Ingrese valor',value=0.1, step=0.1,key=i)
                        pred.at[0,i]=val
                        
                    #st.write(pred)
                    prediccion=ClasificacionRL.predict(pred)
                    st.write("Predicción del modelo")
                    st.write(prediccion)

