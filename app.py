import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

# Configuración de la página - debe ser lo primero
st.set_page_config(
    page_title="Predicción de lluvia en Australia",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título principal
st.title('Predicción de lluvia en Australia')

# Carga de modelos
@st.cache_resource
def load_models():
    models = {
        "Random Search": joblib.load('ModelRS.joblib'),
        "Regresión Logística Inicial": joblib.load('ModelRL.joblib'),
        "Grid Search": joblib.load('ModelGS.joblib'),
        "Red Neuronal": joblib.load('ModelNN.joblib'),
    }
    return models

with st.spinner('Cargando modelos...'):
    models = load_models()
st.success('Modelos cargados exitosamente!')

# Selección de modelo
selected_model = st.sidebar.selectbox(
    "Selecciona el modelo a utilizar",
    options=list(models.keys())
)

# Sidebar
st.sidebar.title("Acerca de")
st.sidebar.info("Esta aplicación predice la probabilidad de lluvia en Australia basada en diversos factores meteorológicos.")

# Opciones para la dirección del viento y sus valores correspondientes
options_dir = {
    "North": "N", "North-Northeast": "NNE", "Northeast": "NE", "East-Northeast": "ENE",
    "East": "E", "East-Southeast": "ESE", "Southeast": "SE", "South-Southeast": "SSE",
    "South": "S", "South-Southwest": "SSW", "Southwest": "SW", "West-Southwest": "WSW",
    "West": "W", "West-Northwest": "WNW", "Northwest": "NW", "North-Northwest": "NNW"   
}

# Temperatura
st.header("Temperatura")
col1, col2, col3 = st.columns(3)
with col1:
    MaxTemp = st.number_input("Temperatura máxima", min_value=-40.0, max_value=50.0, value=20.0, step=0.1)
with col2:
    MinTemp = st.number_input("Temperatura mínima", min_value=-40.0, max_value=50.0, value=0.0, step=0.1)
with col3:
    Temp9am = st.number_input("Temperatura a las 9am", min_value=-40.0, max_value=50.0, value=2.0, step=0.1)
    Temp3pm = st.number_input("Temperatura a las 3pm", min_value=-40.0, max_value=50.0, value=25.0, step=0.1)

# Precipitación y Evaporación
st.header("Precipitación y Evaporación")
col1, col2, col3 = st.columns(3)
with col1:
    Rainfall = st.number_input("Cantidad de lluvia", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
with col2:
    Evaporation = st.number_input("Evaporación", min_value=0.0, max_value=100.0, value=13.5, step=0.1)
with col3:
    Sunshine = st.number_input("Horas de sol", min_value=0.0, max_value=24.0, value=6.8, step=0.1)

# Viento
st.header("Viento")
col1, col2 = st.columns(2)
with col1:
    WindGustDir = st.selectbox("Dirección de la ráfaga de viento", list(options_dir.keys()))
    WindGustSpeed = st.number_input("Velocidad de la ráfaga de viento", min_value=0, max_value=200, value=30)
with col2:
    WindDir9am = st.selectbox("Dirección del viento a las 9am", list(options_dir.keys()))
    WindSpeed9am = st.number_input("Velocidad del viento a las 9am", min_value=0, max_value=200, value=15)
    WindDir3pm = st.selectbox("Dirección del viento a las 3pm", list(options_dir.keys()))
    WindSpeed3pm = st.number_input("Velocidad del viento a las 3pm", min_value=0, max_value=200, value=20)

# Humedad
st.header("Humedad")
col1, col2 = st.columns(2)
with col1:
    Humidity9am = st.number_input("Humedad a las 9am", min_value=0, max_value=100, value=70)
with col2:
    Humidity3pm = st.number_input("Humedad a las 3pm", min_value=0, max_value=100, value=67)

# Presión
st.header("Presión")
col1, col2 = st.columns(2)
with col1:
    Pressure9am = st.number_input("Presión a las 9am", min_value=900, max_value=1100, value=1015)
with col2:
    Pressure3pm = st.number_input("Presión a las 3pm", min_value=900, max_value=1100, value=1010)

# Nubosidad
st.header("Nubosidad")
col1, col2 = st.columns(2)
with col1:
    Cloud9am = st.slider("Nubosidad a las 9am", min_value=0, max_value=9, value=5)
with col2:
    Cloud3pm = st.slider("Nubosidad a las 3pm", min_value=0, max_value=9, value=9)

# Lluvia
st.header("Lluvia")
col1, col2, col3 = st.columns(3)
with col1:
    RainToday = st.selectbox("¿Llovió hoy?", ['No', 'Yes'])
with col2:
    RainTomorrow = st.selectbox("¿Lloverá mañana?", ['No', 'Yes'])
with col3:
    RainfallTomorrow = st.number_input("Cantidad de lluvia para mañana", min_value=0.0, max_value=500.0, value=0.0, step=0.1)

# Botón para realizar la predicción
if st.button("Realizar predicción", type="primary"):
    with st.spinner("Procesando datos y realizando predicción..."):
        
        data = {
            'MaxTemp': [MaxTemp], 'MinTemp': [MinTemp], 'Rainfall': [Rainfall],
            'Evaporation': [Evaporation], 'Sunshine': [Sunshine],
            'WindGustDir': [options_dir[WindGustDir]], 'WindGustSpeed': [WindGustSpeed],
            'WindDir9am': [options_dir[WindDir9am]], 'WindDir3pm': [options_dir[WindDir3pm]],
            'WindSpeed9am': [WindSpeed9am], 'WindSpeed3pm': [WindSpeed3pm],
            'Humidity9am': [Humidity9am], 'Humidity3pm': [Humidity3pm],
            'Pressure9am': [Pressure9am], 'Pressure3pm': [Pressure3pm],
            'Cloud9am': [Cloud9am], 'Cloud3pm': [Cloud3pm],
            'Temp9am': [Temp9am], 'Temp3pm': [Temp3pm],
            'RainToday': [RainToday], 'RainTomorrow': [RainTomorrow],
            'RainfallTomorrow': [RainfallTomorrow]
        }

        df = pd.DataFrame(data)

        file_path= 'weatherAUS.csv'
        weather_data = pd.read_csv(file_path, sep=',',engine='python')

        # Suponiendo que 'df' tiene una fila que deseas agregar
        fila_a_agregar = df.iloc[0]  # Selecciona la primera fila de 'df'

        # Agrega la fila a 'weather_date'
        weather_data.loc[len(weather_data)] = fila_a_agregar

        # Drop unnecessary columns
        weather_data = weather_data.drop(['Unnamed: 0', 'Date', 'Location'], axis=1)

        columns_to_fill = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                        'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
                        'Temp3pm', 'RainfallTomorrow']

        weather_data[columns_to_fill] = weather_data[columns_to_fill].fillna(weather_data[columns_to_fill].median())

        # Crear el estandarizador
        scaler = StandardScaler()

        # Seleccionar las columnas a estandarizar
        columns_to_standardize = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                                'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
                                'Temp3pm', 'RainfallTomorrow']

        # Aplicar la estandarización a las columnas seleccionadas
        weather_data[columns_to_standardize] = scaler.fit_transform(weather_data[columns_to_standardize])

        diccionario = {
            'N': ['N', 'NNW', 'NNE', 'NE', 'NW'],
            'S': ['S', 'SSW', 'SSE', 'SE', 'SW'],
            'E': ['E', 'ENE', 'ESE'],
            'W': ['W', 'WNW', 'WSW'],
        }

        # Invertir el diccionario para el mapeo
        diccionario_invertido = {valor: clave for clave, lista_valores in diccionario.items() for valor in lista_valores}

        # Aplicar la transformación a WindGustDir
        weather_data['WindGustDir'] = weather_data['WindGustDir'].map(diccionario_invertido)

        # Aplicar la transformación a WindDir9am
        weather_data['WindDir9am'] = weather_data['WindDir9am'].map(diccionario_invertido)

        # Aplicar la transformación a WindDir3pm
        weather_data['WindDir3pm'] = weather_data['WindDir3pm'].map(diccionario_invertido)

        columns_to_dummy = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
        weather_data_dummies = pd.get_dummies(weather_data, columns=columns_to_dummy, drop_first=True)
        weather_data_dummies.replace({True: 1, False: 0}, inplace=True)

        last_row = weather_data_dummies.tail(1)
        last_row

        # Drop unnecessary columns
        last_row = last_row.drop(['RainTomorrow_Yes', 'RainfallTomorrow'], axis=1)

        modelo = models[selected_model]
        prediccion = modelo.predict(last_row)
        
        # Manejar diferentes tipos de salida
        if isinstance(prediccion[0], np.ndarray):
            # Para la red neuronal
            probabilidad_lluvia = prediccion[0][0]
            lluvia_manana = "Yes" if probabilidad_lluvia > 0.5 else "No"
        else:
            # Para otros modelos que devuelven 'Yes' o 'No'
            lluvia_manana = prediccion[0]
            probabilidad_lluvia = None
        
    st.header(f"Resultado de la Predicción usando {selected_model}", divider="rainbow")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if lluvia_manana == "Yes":
            st.image("https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExYnVxbHBnM2s5dDY2cnBxdzUyanF6YmdqcjdnODg4YWV0Y2RkZm05ZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/pNn4hlkovWAHfpLRRD/giphy.gif", width=200)
        else:
            st.image("https://media.giphy.com/media/XWXnf6hRiKBJS/giphy.gif", width=200)
    
    with col2:
        st.subheader("Predicción para mañana:")
        st.markdown(f"<h1 style='text-align: center; color: {'#1E90FF' if lluvia_manana == 'Yes' else '#FFD700'};'>{'🌧️ Lloverá' if lluvia_manana == 'Yes' else '☀️ No lloverá'}</h1>", unsafe_allow_html=True)
        
        if probabilidad_lluvia is not None:
            st.metric(label="Probabilidad de lluvia", value=f"{probabilidad_lluvia*100:.1f}%")
        elif hasattr(modelo, 'predict_proba'):
            confianza = modelo.predict_proba(last_row)[0][1] * 100
            st.metric(label="Probabilidad de lluvia", value=f"{confianza:.1f}%")
        else:
            st.write("Probabilidad no disponible para este modelo.")
        
    st.info(f"Esta predicción se basa en los datos meteorológicos proporcionados y el modelo {selected_model} entrenado con datos históricos de Australia.")

