import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st

dados = pd.read_csv('cancerMama.csv')
nomes_colunas = dados.columns.to_list()
features = dados[nomes_colunas[:len(nomes_colunas)-1]]
classes = dados['diagnosis']#0 sem cancer  / 1 = com cancer

#dividir os dados
features_treino,features_teste,classes_treino,classes_teste = train_test_split(features,classes,test_size=0.3,random_state=2)

from sklearn.ensemble import RandomForestClassifier

floresta = RandomForestClassifier(n_estimators=1000)
st.title('Predicao de Cancer de Mama')
#mean_radius      mean_texture     mean_perimeter   mean_area        mean_smoothness  
paciente = []
raio = st.number_input('Digite o raio da lesao:')
paciente.append(raio)
textura = st.number_input('Digite a textura da lesao:')
paciente.append(textura)
perimetro = st.number_input('Digite o perimetro da lesao:')
paciente.append(perimetro)
area = st.number_input('Digite a area da lesao:')
paciente.append(area)
maciez = st.number_input('Digite a maciez da lesao:')
paciente.append(maciez)

resultado = floresta.predict([paciente])
st.write(resultado)
if resultado==0:
  st.write('Paciente sem cancer')
else:
  st.write('Paciente com cancer')
