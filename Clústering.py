import streamlit as st
from PIL import Image
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def cluster():
    column11,column12,colum13=st.columns([2,4,2])
    with column11:
        lottie_hola_url="https://assets8.lottiefiles.com/packages/lf20_f7wf1y3d.json"
        lottie_hola=load_lottieurl(lottie_hola_url)
        st_lottie(lottie_hola,key="holaS")
    with column12:
        titu1= '<h1 style="font-family:Fantasy; color:#1a6563; font-size: 80px;text-align: center;">Clustering</h1>'
        st.markdown(titu1, unsafe_allow_html=True)
    with colum13:
        lottie_compu_url="https://assets8.lottiefiles.com/packages/lf20_kinngfpa.json"
        lottie_compu=load_lottieurl(lottie_compu_url)
        st_lottie(lottie_compu,key="compu")
    #st.write('Name of option is {}'.format(tabs))
    columna33,columna34=st.columns([3,1])
    with columna33:
	    parrafo1='<p style=" color:black; font-size: 16px; text-align:justify;">Es un algoritmo que pertenece al Aprendizaje No Supervisado de Machine Learning, este permite hacer una segmentación y delimitación de grupos de elementos que se unen por características comunes que estos comparten. Estos grupos se crean apartir de los patrones que hay ocultos dentro de los datos, para poder implementar este algoritmo es necesario saber el grado de similitud entre los elementos por lo que para esto se utilizan las métricas de distancia. </p>'
	    st.markdown(parrafo1,unsafe_allow_html=True)
    with columna34:
	    imagen=Image.open('clustering.png')
	    st.image(imagen,width=280,output_format="PNG")
    col1,col2=st.columns(2)
    with col1:
        titu2= '<h1 style="font-family:Fantasy; color:black; font-size: 35px;text-align: center;">Clustering Jerárquico Ascendente</h1>'
        st.markdown(titu2, unsafe_allow_html=True)
        parrafo2='<p style=" color:black; font-size: 16px; text-align:justify;">Este algoritmo organiza a los elementos de manera recursiva en una estructura en forma de árbol, este árbol representa las relaciones de similitud entre los distintos elementos. Este es un algoritmo iterativo, es decir, agrupa en cada iteración a aquellos 2 elementos más cercanos (menor distancia), de esta forma en la siguiente iteración se consideran como un solo elemento o clúster para la siguiente iteración y asi sucesivamente. El árbol se construye de abajo hacia arriba y cada rama representa una relación. Este algoritmo termina cuando se tiene un solo clúster y el número de clústers se define de acuerdo a una cierta altura del árbol.</p>'
        st.markdown(parrafo2,unsafe_allow_html=True)
        imagen=Image.open('dendo.png')
        st.image(imagen,output_format="PNG")
    with col2:
        titu3= '<h1 style="font-family:Fantasy; color:black; font-size: 35px;text-align: center;">Clustering Particional</h1>'
        st.markdown(titu3, unsafe_allow_html=True)
        titu4= '<h1 style="font-family:Fantasy; color:black; font-size: 25px;text-align: center;">KMeans</h1>'
        st.markdown(titu4, unsafe_allow_html=True)
        parrafo3='<p style=" color:black; font-size: 16px; text-align:justify;">El algoritmo Particinal, también conocido como <strong>particiones</strong>, organiza a los elementos dentro de ''k'' clústers. K-means resuelve <strong>problemas de optimización</strong> ya que su función es minimizar la suma de las distancias de cada elemento al centroide de un clúster. Es un algoritmo iterativo y hace como paso inicial una definición aleatoria de centroides, en cada iteración se le asigna el centroide más cercano a cada elemento, cuando ya todos fueron asignados se calcula el promedio de las posiciones de todos los elementos dentro de cada clúster para poder actualizar a este valor a la posición del centroide. El algoritmo finaliza cuando los centroides ya no presentan cambios.</p>'    
        st.markdown(parrafo3,unsafe_allow_html=True)
        imagen=Image.open('kmeans.png')
        st.image(imagen,output_format="PNG")
    st.header("Datos")
    st.markdown("Aqui se realiza la carga de tu archivo con extension .cvs")
    cluster= st.file_uploader("Archivo",key=6,type="csv") 
    if cluster is not None:
        datos=pd.read_csv(cluster)
        st.subheader("Tabla de datos")
        st.write(datos)
        Corr= datos.corr(method='pearson')
        st.header("Selección de variables")
        st.subheader("Matriz de correlaciones")
        st.write(Corr)
        st.subheader("Mapa de calor")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(14,7))
        MatrizInf = np.triu(Corr)
        sns.heatmap(Corr, cmap='RdBu_r', annot=True, mask=MatrizInf)
        st.pyplot()
        añadir_columnas=st.multiselect('Seleccione las variables que desea añadir a la matriz:',datos.columns.values)
        MatrizVariables=np.array(datos[añadir_columnas])
        pd.DataFrame(MatrizVariables)
        flag=np.size(MatrizVariables)
        if flag:
            st.subheader("Matriz con las variables  seleccionadas")
            st.write(MatrizVariables) 
            st.subheader("Datos estandarizados")
            estandarizar = StandardScaler()                                # Se instancia el objeto StandardScaler o MinMaxScaler 
            MEstandarizada = estandarizar.fit_transform(MatrizVariables)   # Sescalan los datos
            pd.DataFrame(MatrizVariables)
            st.write(MEstandarizada)
            st.subheader('Elección de cluster')
            clas=st.selectbox("Clúster que deseas ejecutar:",('Jerárquico','Particional'))
            if clas=="Jerárquico":
                st.header("Cluster Jerárquico") 
                st.subheader("Árbol jerárquico")
                opcion=st.selectbox("Elige la distancia que desea utilizar para calcular el árbol jerárquico:",('Euclidiana', 'Chebyshev', 'Manhattan'))            
                if opcion== 'Euclidiana':
                    plt.figure(figsize=(10, 7))
                    plt.title("Árbol jerárquico")
                    #plt.xlabel('Observaciones')
                    #plt.ylabel('Distancia')
                    plt.Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
                    st.pyplot()
                    #Se crean las etiquetas de los elementos en los clusters
                    numero_clusters=st.number_input("Ingrese el numero de clusters a usar basandose en la gráfica anterior:",1,10,3)
                    MJerarquico = AgglomerativeClustering(n_clusters=numero_clusters, linkage='complete', affinity='euclidean')
                    MJerarquico.fit_predict(MEstandarizada)
                    MJerarquico.labels_
                    datos['clusterH'] = MJerarquico.labels_
                    #st.write(datos)
                    st.subheader("Número de elementos en cada uno de los clusters")
                    st.write(datos.groupby(['clusterH'])['clusterH'].count())
                    CentroidesH = datos.groupby(['clusterH'])[añadir_columnas].mean()
                    if st.button('Obtención de Clústers'):
                        st.subheader("Clústers")
                        st.write(CentroidesH)
                    #st.subheader("Obtención de Clústers")
                    #st.write(CentroidesH)
                elif opcion=='Chebyshev':
                    plt.figure(figsize=(10, 7))
                    plt.title("Árbol jerárquico")
                    #plt.xlabel('Observaciones')
                    #plt.ylabel('Distancia')
                    Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='chebyshev'))
                    st.pyplot()
                    #Se crean las etiquetas de los elementos en los clusters
                    numero_clusters=st.number_input("Ingrese el numero de clusters a usar basandose en la gráfica anterior:",1,10,3)
                    MJerarquico = AgglomerativeClustering(n_clusters=numero_clusters, linkage='complete', affinity='chebyshev')
                    MJerarquico.fit_predict(MEstandarizada)
                    MJerarquico.labels_
                    datos['clusterH'] = MJerarquico.labels_
                    #st.write(datos)
                    st.subheader("Número de elementos en cada uno de los clusters")
                    st.write(datos.groupby(['clusterH'])['clusterH'].count())
                    CentroidesH = datos.groupby(['clusterH'])[añadir_columnas].mean()
                    if st.button('Obtención de Clústers'):
                        st.subheader("Clústers")
                        st.write(CentroidesH)
                    #st.subheader("Obtención de Clústers")
                    #st.write(CentroidesH)
                elif opcion=='Manhattan':
                    plt.figure(figsize=(10, 7))
                    plt.title("Árbol jerárquico")
                    #plt.xlabel('Observaciones')
                    #plt.ylabel('Distancia')
                    Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='cityblock'))
                    st.pyplot()
                    #Se crean las etiquetas de los elementos en los clusters
                    numero_clusters=st.number_input("Ingrese el numero de clusters a usar basandose en la gráfica anterior:",1,10,3)
                    MJerarquico = AgglomerativeClustering(n_clusters=numero_clusters, linkage='complete', affinity='cityblock')
                    MJerarquico.fit_predict(MEstandarizada)
                    MJerarquico.labels_
                    datos['clusterH'] = MJerarquico.labels_
                    #st.write(datos)
                    st.subheader("Número de elementos en cada uno de los clusters")
                    st.write(datos.groupby(['clusterH'])['clusterH'].count())
                    CentroidesH = datos.groupby(['clusterH'])[añadir_columnas].mean()
                    if st.button('Obtención de Clústers'):
                        st.subheader("Clústers")
                        st.write(CentroidesH)
            if clas=="Particional":
                st.header("Cluster Particional")
                st.subheader("Obtención del ''codo'' para KMeans")
                SSE = []
                for i in range(2, 12):
                    km = KMeans(n_clusters=i, random_state=0)
                    km.fit(MEstandarizada)
                    SSE.append(km.inertia_)
                #Se grafica SSE en función de k
                plt.figure(figsize=(10, 7))
                plt.plot(range(2, 12), SSE, marker='o')
                plt.xlabel('Cantidad de clusters *k*')
                plt.ylabel('SSE')
                plt.title('Elbow Method')
                st.pyplot()
                kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
                st.write("La cantidad de clusters es:",kl.elbow)
                plt.style.use('ggplot')
                st.pyplot(kl.plot_knee())
                #Se crean las etiquetas de los elementos en los clusters
                parrafo4='<p style=" color:black; font-size: 16px; text-align:justify;">K-Means utiliza el método del codo para poder calcular y decicir el número de clústers más eficiente. Para esto, obtiene la suma de la distancia al cuadrado entre cada elemento del clúster y su centroide (<strong>SSE o WSS</strong>). La gráfica anterior muestra la relación entre el SSE y los clústers con una k de 2 a 12. </p>'
                st.markdown(parrafo4,unsafe_allow_html=True)
                num_clusters=st.number_input("Ingrese el numero de clusters a usar basandose en la gráfica anterior:",1,12,3)
                MParticional = KMeans(n_clusters=num_clusters, random_state=0).fit(MEstandarizada)
                MParticional.predict(MEstandarizada)
                MParticional.labels_
                datos['clusterP'] = MParticional.labels_
                st.subheader("Número de elementos en cada uno de los clusters")
                st.write(datos.groupby(['clusterP'])['clusterP'].count())
                CentroidesP=datos.groupby(['clusterP'])[añadir_columnas].mean()
                if st.button('Obtención de Clústers'):
                    st.subheader("Clústers")
                    st.write(CentroidesP)
               