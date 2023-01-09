import streamlit as st              # Para usar streamlit
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori
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

def asociacion():
    colum1,colum2,colum3=st.columns([1,3,1])
    with colum1:
        lottie_hola_url="https://assets8.lottiefiles.com/packages/lf20_f7wf1y3d.json"
        lottie_hola=load_lottieurl(lottie_hola_url)
        st_lottie(lottie_hola,key="holaS")
    with colum2:
        head = '<h1 style="font-family:Fantasy; color:#1a6563; font-size: 60px;text-align: center;">Reglas de Asociación</h1>'
        st.markdown(head, unsafe_allow_html=True)
        title = '<h1 style="font-family:Fantasy; color: #1a6563; font-size: 40px;text-align: center;">Apriori</h1>'
        st.markdown(title, unsafe_allow_html=True)
    with colum3:
        lottie_ani_url="https://assets8.lottiefiles.com/packages/lf20_4asnmi1e.json"
        lottie_ani=load_lottieurl(lottie_ani_url)
        st_lottie(lottie_ani,key="ani")
    #st.header("Apriori")
    sub1 = '<h1 style="font-family:Century goth; color:black; font-size: 30px;text-align: justify;">¿Cómo funciona el Algoritmo Apriori?</h1>'
    st.markdown(sub1,unsafe_allow_html=True)
    #st.subheader("¿Cómo funciona el Algoritmo Apriori?")
    text='<p style=" color:black; font-family:century goth; font-size: 16px; text-align:justify;">Este es un algoritmo de aprendizaje automático que se basa en reglas, se utiliza para encontrar relaciones ocultas en los datos. <strong>También es conocido como análisis de afinidad</strong>. Básicamente consiste en identificar un conjunto de patrones secuenciales en forma de reglas de tipo:</p>'
    st.markdown(text,unsafe_allow_html=True)
    text1='<p style=" color:black; font-family:century goth; font-size: 16px; text-align:center;"><strong>IF-->THEN</strong>, donde <strong>IF</strong> es el antecedente y <strong>THEN</strong> el consecuente.</p>'
    st.markdown(text1,unsafe_allow_html=True)
    columna91,columna101=st.columns([3,1])
    with columna91:
	    st.markdown("""Este algoritmo trabaja con datos transaccionales, pero **¿Cuáles son los datos transaccionales?**
	    Estos datos tienen como caracteristica lo siguiente: 
- Son datos operativos
- Se emplean para controlar y ejecutar tareas
- Son altamente normalizados y se almacenan en tablas""")
	    st.markdown("""Las reglas de asociación son una proporción probabilistica sobre la ocurrencia
	        de eventos presentes en el conjunto de datos, para generar estas reglas se debe hacer una **poda**
	        con la finalidad de eliminar las reglas menos importantes.""")
    with columna101:
	    imagen=Image.open('apriori.jpg')
	    st.image(imagen,width=350,output_format="JPG")
    sub2 = '<h1 style="font-family:Century Goth; color:black; font-size: 30px;text-align: justify;">Obtención de las reglas significativas</h1>'
    st.markdown(sub2,unsafe_allow_html=True)
    #st.subheader("Obtención de las reglas significativas")
    st.markdown("""Se deben utilizar mediciones para las reglas más significativas, los cuales serán los parámetros
        que podrá modificar el usuario: 
- Soporte: Indica que tan importante es una regla dentro del total de transacciones.
- Confianza: Indica que tan fiable es una regla. 
- Lift (Elevación): Indica el nivel de posibilidad entre el antecedente y el consecuente de la regla. 
    - Si **lift<1** indica una relacion **negativa**. 
    - Si **lift=1** son **independientes**. 
    - Si **lift>1** hay una relación **positiva**.""")
    st.header("Datos")
    st.markdown("Aqui se realiza la carga de tu archivo con extension .cvs")
    reglas = st.file_uploader("Archivo")
    if reglas is not None:
        datos=pd.read_csv(reglas, header=None)
        st.subheader("Tabla de datos")
        st.write(datos)
        #Se incluyen todas las transacciones en una sola lista
        Transacciones = datos.values.reshape(-1).tolist() #-1 significa 'dimensión no conocida'
        #Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
        ListaM = pd.DataFrame(Transacciones)
        ListaM['Frecuencia'] = 1
            #Se agrupa los elementos
        ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
        ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
        ListaM = ListaM.rename(columns={0 : 'Item'})
        # Se genera un gráfico de barras
        st.subheader("Gráfica de frecuencias")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(16,20), dpi=300)
        plt.ylabel('Item')
        plt.xlabel('Frecuencia')
        plt.barh(ListaM['Item'], width=ListaM['Frecuencia'], color='blue')
        st.pyplot()
         #Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
        #level=0 especifica desde el primer índice
        Lista = datos.stack().groupby(level=0).apply(list).tolist()
        st.header("Parámetros")
        soporte = st.number_input('Soporte',0.0,3.0,0.028,0.001)
        confianza = st.number_input('Confianza',0.0,1.0,0.3,0.001)
        elevacion = st.number_input('Elevación',1.1,3.0,1.1,0.001)
        st.write("**Valores elegidos**")
        st.write("Soporte:", soporte)
        st.write("Confianza:", confianza)
        st.write("Elevación:", elevacion)
        if st.button('Aplicar Algoritmo'):
            st.header("Reglas obtenidas con el Algoritmo Apriori")
            Reglas= apriori(Lista,min_support=soporte,min_confidence=confianza,min_lift=elevacion)
            Resultados= list(Reglas)
            st.subheader("Numero de reglas encontradas:")
            st.write((len(Resultados)))
            for item in Resultados:
                  #El primer índice de la lista
                Emparejar = item[0]
                items = [x for x in Emparejar]
                st.write("**Regla**: " + str(item[0]))

                      #El segundo índice de la lista
                st.write("Soporte: " + str(item[1]))

                      #El tercer índice de la lista
                st.write("Confianza: " + str(item[2][0][2]))
                st.write("Lift: " + str(item[2][0][3])) 
                st.write("=====================================")