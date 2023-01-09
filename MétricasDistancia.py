import streamlit as st
from PIL import Image
import pandas as pd                         # Para la manipulación y análisis de datos
import numpy as np                          # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt             # Para generar gráficas a partir de los datos
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def metricas():
    column1,column2,column3=st.columns([1,3,1])
    with column1:
        lottie_hola_url="https://assets8.lottiefiles.com/packages/lf20_f7wf1y3d.json"
        lottie_hola=load_lottieurl(lottie_hola_url)
        st_lottie(lottie_hola,key="holaS")
    with column2:
        tit = '<h1 style="font-family:Fantasy; color: #1a6563; font-size: 60px;text-align: center;">Métricas de Distancia</h1>'
        st.markdown(tit, unsafe_allow_html=True)
    with column3:
        lottie_url_mono="https://assets8.lottiefiles.com/packages/lf20_3vbOcw.json"
        lottie_mono=load_lottieurl(lottie_url_mono)
        st_lottie(lottie_mono,key="mono")
    
    #st.write('Name of option is {}'.format(tabs))
    te1='<p style=" color:black; font-size: 16px; text-align:justify;"> Estás también son conocidas como <strong>búsqueda de similitud vectorial</strong> ya que es una puntuación objetiva que resume la diferencia entre dos objetos, es decir, permiten conocer que tan similares o disimiles son 2 objetos. No es un algoritmo propio de Machine Learning pero si son mediciones que se utilizan para aprender de los datos en Algoritmos de <strong>Aprendizaje Supervisado</strong> y <strong>No Supervisado.</strong></p>'
    st.markdown(te1,unsafe_allow_html=True)
    st.markdown("""Posee las siguientes características:""")
    col5, col6,col7= st.columns([3,2,2])
    with col5:
        st.markdown("- **No negativa**, el valor puede ser mayor o igual a 0.")
        st.markdown("- Es **Simétrica**, la distancia de A a B es la misma que de B a A.")
        st.markdown("- La distancia de dos objetos en un mismo punto es **0**.")      
    with col7:
	    imagen=Image.open('metrica.png')
	    st.image(imagen,output_format="PNG")
    with col6:
        st.markdown("**d(a,b)≥0**")
        st.markdown("**d(a,b)=d(b,a)**")
        st.markdown("**d(a,a)=0**")
    col1, col2= st.columns(2)
    col3,col4 =st.columns(2)
    with col1:
        tit1= '<h1 style="font-family:Fantasy; color:black; font-size: 35px;text-align: center;">Distancia Euclidiana</h1>'
        st.markdown(tit1, unsafe_allow_html=True)
        #st.header("Distancia Euclidiana")
        text1='<p style=" color:black; font-size: 16px; text-align:justify;">Es una de las métricas más utilizadas para calcular la distancia entre dos puntos. También es conocida como <strong>espacio euclidiano</strong>. Esta métrica se basa en la aplicación del <strong>Teorema de Pitágoras</strong>, en donde la distancia es la longitud de la hipotenusa. Para elementos con i dimensiones se utiliza la siguiente expresión:</p>'
        st.markdown(text1,unsafe_allow_html=True)
        imagen=Image.open('euclidiana.png')
        st.image(imagen,output_format="PNG")
    with col2:
        tit2= '<h1 style="font-family:Fantasy; color:black; font-size: 35px;text-align: center;">Distancia de Chebyshev</h1>'
        st.markdown(tit2, unsafe_allow_html=True)
        #st.header("Distancia de Chebyshev")
        text2='<p style=" color:black; font-size: 16px; text-align:justify;">Esta distancia representa el valor máximo absoluto que hay de las diferencias entre las coordenadas de un par de elementos. También se le conoce como <strong>métrica máxima</strong>. Para elementos con i dimensiones se utiliza la siguiente expresión:</p>'
        st.markdown(text2,unsafe_allow_html=True)
        imagen=Image.open('chebyshev.png')
        st.image(imagen,output_format="PNG")
    with col3:
        tit3= '<h1 style="font-family:Fantasy; color:black; font-size: 35px;text-align: center;">Distancia de Manhattan</h1>'
        st.markdown(tit3, unsafe_allow_html=True)
        #st.header("Distancia de Manhattan")
        text3='<p style=" color:black; font-size: 16px; text-align:justify;">Está distancia se utliza siempre que se necesita calcular la distancia entre dos puntos en una ruta similar a una cuadrícula (es decir, información geoespacial). También es conocida como geometría del taxi, distancia de la manzana de la ciudad y distancia rectilínea. Esta inspirada en la ciudad de Manhattan, New York. Para elementos con i dimensiones se utiliza la siguiente expresión:</p>'
        st.markdown(text3,unsafe_allow_html=True)    
        imagen=Image.open('manhattan.png')
        st.image(imagen,output_format="PNG")
    with col4:
        tit4= '<h1 style="font-family:Fantasy; color:black; font-size: 35px;text-align: center;">Distancia de Minkowski</h1>'
        st.markdown(tit4, unsafe_allow_html=True)
        #st.header("Distancia de Minkowski")
        text3='<p style=" color:black; font-size: 16px; text-align:justify;">Esta es la distancia entre dos puntos en un espacio n-dimensional. Está es una métrica generalizada, si el valor de lambda cambia a 1, 2 y 3, estará representando a las distancias Euclidiana, Manhattan y Chebyshev respectivamente. Para fines de esta aplicación usaremos el valor predeterminado de 1.5 para tener una representación equilibrada.</p>'
        st.markdown(text3,unsafe_allow_html=True)
        imagen=Image.open('minkowski.png')
        st.image(imagen,output_format="PNG")
    st.header("Datos")
    st.markdown("Aqui se realiza la carga de tu archivo con extension .cvs")
    metricas = st.file_uploader("Archivo",key=2,type="csv")
    st.header("Algoritmo")
    if metricas is not None:
        datos=pd.read_csv(metricas)
        st.subheader("Tabla de datos")
        st.write(datos)
        st.subheader("Matriz de Correlaciones")
        st.write(datos.corr()) 
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Mapa de Calor")
        plt.figure(figsize=(14,7))
        MatrizInf = np.triu(datos.corr())
        sns.heatmap(datos.corr(), cmap = 'RdBu_r', annot = True, mask = MatrizInf)
        st.pyplot()
        for columna in datos.columns: 
            tipodato=datos.dtypes[columna]
            if tipodato=='object':
                datos=datos.drop(columns=[columna])
        st.header("Selección de variables")
        borrar_columnas=st.multiselect('Seleccione las variables que desea borrar:',datos.columns.values)
        for delete in borrar_columnas:
            datos=datos.drop(columns=[delete])
        st.subheader("Matriz sin las variables eliminadas")
        st.write(datos) 
        st.subheader("Datos estandarizados")
        estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
        MEstandarizada = estandarizar.fit_transform(datos)
        st.write(MEstandarizada)
        st.subheader("Matriz de Distancia")
        option = st.selectbox('Matriz a seleccionar',
        ('Euclidiana', 'Chebyshev', 'Manhattan','Minkowski')) 
        if option=='Euclidiana':
            st.subheader("Matriz Euclidiana")
            DstEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean')
            MEuclidiana = pd.DataFrame(DstEuclidiana)
            st.write(MEuclidiana)
            st.subheader("Distancia entre objetos")
            objeto1 = st.number_input('Seleccione el Objeto 1:',0,100000000,0)
            objeto2 = st.number_input('Seleccione el Objeto 2:',0,100000000,0)
            Objeto1 = MEstandarizada[objeto1]
            Objeto2 = MEstandarizada[objeto2]
            if st.button('Aplicar Métrica de Distancia Euclidiana entre los objetos'):
                dstEuclidiana = distance.euclidean(Objeto1,Objeto2)
                st.write("La distancia entre los 2 objetos es:",dstEuclidiana) 
        elif option=='Chebyshev':
            st.subheader("Matriz Chebyshev")
            DstChebyshev = cdist(MEstandarizada, MEstandarizada, metric='chebyshev')
            MChebyshev = pd.DataFrame(DstChebyshev)
            st.write(MChebyshev)
            st.subheader("Distancia entre objetos")
            objeto1 = st.number_input('Seleccione el Objeto 1:',0,100000000,0)
            objeto2 = st.number_input('Seleccione el Objeto 2:',0,100000000,0)
            Objeto1 = MEstandarizada[objeto1]
            Objeto2 = MEstandarizada[objeto2]
            if st.button('Aplicar Métrica de Distancia Chebyshev entre los objetos'):
                dstChebyshev = distance.chebyshev(Objeto1,Objeto2)
                st.write("La distancia entre los 2 objetos es:",dstChebyshev)
        elif option=='Manhattan':
            st.subheader("Matriz Manhattan")
            DstManhattan = cdist(MEstandarizada, MEstandarizada, metric='cityblock')
            MManhattan = pd.DataFrame(DstManhattan)
            st.write(MManhattan)
            st.subheader("Distancia entre objetos")
            objeto1 = st.number_input('Seleccione el Objeto 1:',0,100000000,0)
            objeto2 = st.number_input('Seleccione el Objeto 2:',0,100000000,0)
            Objeto1 = MEstandarizada[objeto1]
            Objeto2 = MEstandarizada[objeto2]
            if st.button('Aplicar Métrica de Distancia Manhattan entre los objetos'):
                dstManhattan = distance.cityblock(Objeto1,Objeto2)
                st.write("La distancia entre los 2 objetos es:",dstManhattan)
        elif option=='Minkowski':
            st.subheader("Matriz Minkowski")
            DstMinkowski = cdist(MEstandarizada, MEstandarizada, metric='minkowski', p=1.5)
            MMinkowski = pd.DataFrame(DstMinkowski)
            st.write(MMinkowski)
            st.subheader("Distancia entre objetos")
            objeto1 = st.number_input('Seleccione el Objeto 1:',0,100000000,0)
            objeto2 = st.number_input('Seleccione el Objeto 2:',0,100000000,0)
            Objeto1 = MEstandarizada[objeto1]
            Objeto2 = MEstandarizada[objeto2]
            if st.button('Aplicar Métrica de Distancia Monkowski entre los objetos'):
                dstMinkowski = distance.minkowski(Objeto1,Objeto2, p=1.5)
                st.write("La distancia entre los 2 objetos es:",dstMinkowski)