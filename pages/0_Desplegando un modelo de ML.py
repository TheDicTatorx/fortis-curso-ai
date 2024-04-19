# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import pickle
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import streamlit as st
from streamlit.hello.utils import show_code

@st.cache_data
def load_model():
    # Load the model from the file
    with open('./RFC_diabetes.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

@st.cache_data
def load_scaler():
    # Load the model from the file
    with open('./diabetes_escalador.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

@st.cache_data
def leer_csv():
    return pd.read_csv('./diabetes.csv')

@st.cache_data
def limpieza_datos(data):
    data.drop(columns='ID', inplace=True)
    data.dropna(inplace=True)
    data = data[data['SkinThickness'] < 1000]
    data = data[data['DiabetesPedigreeFunction'] > 0]
    return data

@st.cache_data
def escalar_datos(escalador,data):
    return pd.DataFrame(escalador.transform(data),columns=data.colums)


show_data = st.checkbox('Mostrar tabla')
ml_model = load_model()
custom_scaler = load_scaler()
df = leer_csv()
df_clean = limpieza_datos(df)
# df_escalado = escalar_datos(custom_scaler, df_clean)
df_escalado = pd.DataFrame(custom_scaler.transform(df_clean),columns=df_clean.columns)
df_corr = df_clean.corr()

#===============SIDEBAR===============
sidebar = st.sidebar
sidebar.header('Sección de Filtros')

preg = sidebar.number_input('Valor de Pregnancies')
gluc = sidebar.number_input('Valor de Glucose')
blood = sidebar.number_input('Valor de BloodPressure')
skin = sidebar.number_input('Valor de SkinThickness')
insu = sidebar.number_input('Valor de Insulin')
bmi = sidebar.number_input('Valor de BMI')
pedi = sidebar.number_input('Valor de DiabetesPedigreeFunction')
age = sidebar.number_input('Valor de Age')

valores_input = [preg, gluc, blood, skin, insu, bmi, pedi, age, 0]
#=====================================

st.header('Dashboard de Resultados Modelo ML')

st.write('''Este ejemplo es para demostrar las capacidades de streamlit como visualizador.''')

# Calculamos las predicciones de todo el dataset
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
df_clean['Predicciones'] = ml_model.predict(df_escalado[features])

if show_data:
    st.write(df_clean)

valores_input_escalados = pd.DataFrame(custom_scaler.transform([valores_input]), columns=df_escalado.columns)
st.write(f'''Resultado de una predicción con los valores indicados en el sidebar. 
         Outcome: {ml_model.predict(valores_input_escalados[features])[0]}''')

st.markdown('---')

st.write('Esto es una prueba')

st.write(df_corr)

fig = px.scatter(df_clean, x="SkinThickness", y="BMI", title="Scatter plot")
st.plotly_chart(fig)