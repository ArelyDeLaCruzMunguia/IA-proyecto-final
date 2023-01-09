import streamlit as st              # Para usar streamlit
from PIL import Image
from st_on_hover_tabs import on_hover_tabs
import streamlit.components.v1 as components
import ClasificaciónL
import PaginaPrincipal
import ReglasAsociación
import MétricasDistancia
import Clústering
import ÁrbolesAleatorios
import BosquesAleatorios

st.set_page_config(layout="wide")

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
with st.sidebar:
     tabs = on_hover_tabs(tabName=['Página Principal', 'Reglas de Asociación', 'Métricas de Distancia','Clústering','Clasificación Logística','Árboles de Decisión','Bosques Aleatorios'], 
                              iconName=['home', 'east', 'domain disabled','category','moving','park','forest'],
                              styles = {'navtab': {'background-color':'#1ca8a5',
                                                  'color': '#111',
                                                  'font-size': '18px',
                                                  'fon-family': 'Century Gothic',
                                                  'transition': '.3s',
                                                  'white-space': 'nowrap',
                                                  'text-transform': 'uppercase'},
                                       'tabOptionsStyle': {':hover :hover': {'color': 'white',
                                                                      'cursor': 'pointer'}},
                                       'iconStyle':{'position':'fixed',
                                                    'left':'7.5px',
                                                    'text-align': 'left'},
                                       'tabStyle' : {'list-style-type': 'none',
                                                     'margin-bottom': '30px',
                                                     'padding-left': '30px'}},
                             key="1") ## create tabs for on hover navigation bar
                             
if tabs =='Página Principal':
    PaginaPrincipal.principal()
    #st.write('Name of option is {}'.format(tabs))
elif tabs == 'Reglas de Asociación':
    #PaginaPrincipal.principal()
    ReglasAsociación.asociacion()
    #st.write('Name of option is {}'.format(tabs))    
elif tabs == 'Métricas de Distancia':
    MétricasDistancia.metricas()
    #st.write('Name of option is {}'.format(tabs))
elif tabs== 'Clústering':
	Clústering.cluster()
	#st.write('Name of option is {}'.format(tabs))
elif tabs== 'Clasificación Logística':
 	ClasificaciónL.clasificacionLogistica()
    #st.write('Name of option is {}'.format(tabs))
elif tabs== 'Árboles de Decisión':
	ÁrbolesAleatorios.arboles()
        #st.write('Name of option is {}'.format(tabs))
elif tabs== 'Bosques Aleatorios':
	BosquesAleatorios.bosques()
        #st.write('Name of option is {}'.format(tabs))







