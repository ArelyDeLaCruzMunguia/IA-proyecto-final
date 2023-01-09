import streamlit as st
from PIL import Image
import time
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def principal():
    columna9,columna10, columna13=st.columns([1,3,1])
    with columna9: 
        imagen=Image.open('cai.png')
        st.image(imagen, caption=None, width=200,  use_column_width=1500, clamp=True, channels="RGB", output_format="PNG")
    with columna10:
        st.title('CAI: Herramienta de Aprendizaje de IA')
        st.text('Elaborado por Arely De La Cruz Munguía')
    with columna13:
    	lottie_hola_url="https://assets8.lottiefiles.com/packages/lf20_f7wf1y3d.json"
    	lottie_hola=load_lottieurl(lottie_hola_url)
    	st_lottie(lottie_hola,key="holaS")
    st.header('¿Qué hace CAI?')
    colummna11,columna12=st.columns([3,2])
    with colummna11:
	    texto='<p style="color:black; font-size: 17px; text-align:justify;">CAI es una herramienta de Inteligencia Artificial que ha implementado algoritmos de Machine Learning. Dentro de esta aplicación se han implementado algoritmos de Asociación, Clústering (dónde se han implementado además métricas de distancia) y Selección de Características. El usuario podrá ingresar los datos de acuerdo a las caracteristicas de cada algoritmo para asi realizar un análisis que podrá visualizar por medio de tablas y gráficas que CAI implementó. Cada uno de estos algoritmos indicará que pasos debe seguir para lograr un correcto análisis.</p>'
	    st.markdown(texto, unsafe_allow_html=True)
	    #st.markdown(cuerpo)
	    texto2='<p style=" color:black; font-size: 17px; text-align:justify;">La gran ventaja de CAI es que el usuario podrá elegir y modificar los parámetros de acuerdo a sus necesidades y de acuerdo al algoritmo que desee utilizar, de esta forma podrá ver en acción el funcionamiento de estos y que es lo que representan los resultados que estos obtienen.</p>'
	    st.markdown(texto2,unsafe_allow_html=True)
    with columna12:
    	lottie_hi="https://assets1.lottiefiles.com/packages/lf20_jrpzvtqz.json"
    	lottie_hello=load_lottieurl(lottie_hi)
    	st_lottie(lottie_hello, key="hello")
    st.subheader("¿Qué Algoritmos implementa CAI?")
    st.markdown("""
	- **Reglas de Asociación:** 
	    - Apriori.
	- **Métricas de Distancia:**
	    - Euclidiana 
	    - Chebyshev 
	    - Manhattan 
	    - Minkowski
	- **Clúster:** 
		- Jerarquico 
		- Particional
	- **Clasificación Logística:**
	    - Regresión Logística
	- **Árboles Aleatorios:** 
	    - Pronóstico
	    - Clasificación	
	- **Bosques Aleatorios:**
	    - Pronóstico
	    - Clasificación""")
    st.subheader('¿Cómo hacer uso de CAI?')
    texto3='<p style="color:black; font-size: 17px; text-align:justify;">Del lado derecho se encuentra una barra en donde se tienen las distintas opciones para elegir con que algoritmo trabajar, de esta forma los usuarios podrán elegir con cuál trabajar y en el orden que lo deseen, en cada una de las páginas se visualiza una descripción de como es que trabaja cada algoritmo, que tipo de datos acepta y en que formato se deben subir. Cada uno de los algoritmos tiene parámetros que se pueden modificar según los requerimientos del usuario.</p>'
    st.markdown(texto3,unsafe_allow_html=True)