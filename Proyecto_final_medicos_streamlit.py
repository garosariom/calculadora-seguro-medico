import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gspread
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from google.oauth2.service_account import Credentials

def guardar_en_google_sheets(datos_usuario):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

        # 🔥 Transformar correctamente la clave dentro del diccionario
        service_account_info = dict(st.secrets["gcp_service_account"])
        service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

        creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open("datos_seguro").sheet1
        sheet.append_row(datos_usuario)
    except Exception as e:
        st.warning(f"⚠️ No se pudo guardar en Google Sheets: {e}")

# 📌 Cargar y limpiar el dataset
file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv"
column_names = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df = pd.read_csv(file_url, names=column_names, header=None)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["smoker"] = pd.to_numeric(df["smoker"], errors="coerce")

# 📌 Definir variables predictoras y objetivo
X = df[['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region']]
Y = df['charges']

# 📌 Dividir datos en entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# 📌 Aplicar Regresión Ridge con mejor alpha
alpha_values = np.logspace(-3, 2, 10)
ridge_cv = RidgeCV(alphas=alpha_values, store_cv_results=True)
ridge_cv.fit(X_train, Y_train)
best_alpha = ridge_cv.alpha_

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

ridge_best = Ridge(alpha=best_alpha)
ridge_best.fit(X_train_poly, Y_train)

# 📈 Crear la interfaz en Streamlit
st.title("📊 Calculadora de Seguro Médico")
st.markdown("Ingrese los datos en el panel lateral para calcular el costo estimado del seguro médico.")

# 📈 Barra lateral para ingresar datos
st.sidebar.header("📋 Ingrese sus datos")

age = st.sidebar.number_input("Edad", min_value=18, max_value=100, step=1)
gender = st.sidebar.radio("Género", [1, 2], format_func=lambda x: "Femenino" if x == 1 else "Masculino")
weight = st.sidebar.number_input("Peso (en libras)", min_value=80.0, max_value=500.0, step=1.0)
height = st.sidebar.number_input("Altura (en metros)", min_value=1.0, max_value=2.5, step=0.01)
children = st.sidebar.number_input("Número de Hijos", min_value=0, max_value=10, step=1)
smoker = st.sidebar.radio("Fumador", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
region = st.sidebar.selectbox("Región", [1, 2, 3, 4], format_func=lambda x: ["Noroeste", "Noreste", "Suroeste", "Sureste"][x-1])

# 📈 Botón de predicción
if st.sidebar.button("🔍 Calcular Costo"):
    bmi = round((weight / 2.205) / (height ** 2), 2)
    st.markdown(f"🔍 **Tu indice de masa corporal es: {bmi}**")

    nuevo_paciente = np.array([[age, gender, bmi, children, smoker, region]])
    nuevo_paciente_poly = poly.transform(nuevo_paciente)
    prediccion_costo = ridge_best.predict(nuevo_paciente_poly)

    resultado = f"💰 El costo estimado de tu seguro es: **${prediccion_costo[0]:,.2f}**"
    st.success(resultado)

    datos_usuario = [age, gender, bmi, children, smoker, region, round(prediccion_costo[0], 2)]
    guardar_en_google_sheets(datos_usuario)

    # Comparación con la población
    st.subheader("📊 Comparación con otros usuarios")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["charges"], bins=30, kde=True, color="skyblue", ax=ax1)
    ax1.axvline(prediccion_costo[0], color="red", linestyle="--", linewidth=2, label="Tu costo")
    ax1.set_title("Tu costo comparado con la distribución general")
    ax1.set_xlabel("Costo del Seguro (USD)")
    ax1.legend()
    st.pyplot(fig1)

    # Comparación: fumar
    st.subheader("📊 ¿Qué pasaría si...")
    datos_fumar = [age, gender, bmi, children, 0 if smoker == 1 else 1, region]
    costo_fumar = ridge_best.predict(poly.transform([datos_fumar]))[0]
    dif_fumar = costo_fumar - prediccion_costo[0]
    color_fumar = "🟢" if dif_fumar < 0 else "🔴"
    st.markdown(f"{color_fumar} Si {'no fumaras' if smoker == 1 else 'fumaras'}, el costo de tu seguro sería **${costo_fumar:,.2f}** ({'menos' if dif_fumar < 0 else 'más'} que ahora)")

    fig_fumar, ax_fumar = plt.subplots(figsize=(8, 4))
    ax_fumar.bar(["Actual", "Si no fumaras" if smoker == 1 else "Si fumaras"], [prediccion_costo[0], costo_fumar], color=["blue", "green" if dif_fumar < 0 else "red"])
    ax_fumar.set_ylabel("Costo del Seguro (USD)")
    ax_fumar.set_title("Impacto de Fumar en el Costo del Seguro")
    st.pyplot(fig_fumar)

    # Comparación: género
    datos_genero = [age, 2 if gender == 1 else 1, bmi, children, smoker, region]
    costo_genero = ridge_best.predict(poly.transform([datos_genero]))[0]
    dif_genero = costo_genero - prediccion_costo[0]
    color_genero = "🟢" if dif_genero < 0 else "🔴"

    st.subheader("📊 Comparación de Género")
    st.markdown(f"{color_genero} Si fueras del otro género, el costo de tu seguro sería **${costo_genero:,.2f}** ({'menos' if dif_genero < 0 else 'más'} que ahora)")

    # Crear gráfico de línea en vez de barras
    fig_genero, ax_genero = plt.subplots(figsize=(8, 4))
    labels = ["Actual", "Otro género"]
    valores = [prediccion_costo[0], costo_genero]

    ax_genero.plot(labels, valores, marker="o", linestyle="-", color="red" if dif_genero > 0 else "green", linewidth=2)
    ax_genero.set_ylabel("Costo del Seguro (USD)")
    ax_genero.set_title("Impacto del Género en el Costo del Seguro")
    ax_genero.grid(True)
    st.pyplot(fig_genero)


    # Comparación visual del BMI
    st.subheader("📊 Impacto de tu indice de tu indice de masa corporal (IMC) en el Costo")
    st.write("El Índice de Masa Corporal (IMC) es una medida que relaciona el peso con la altura, utilizada para clasificar el peso corporal en categorías como bajo peso, peso saludable, sobrepeso y obesidad. El siguiente gráfico muestra claramente dónde se encuentra tu costo estimado vs el la población general:")
    fig_bmi, ax_bmi = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="bmi", y="charges", alpha=0.6, ax=ax_bmi)
    ax_bmi.scatter(bmi, prediccion_costo[0], color="red", s=100, label="Tu posición")
    ax_bmi.set_title("Relación entre IMC y Costo del Seguro")
    ax_bmi.set_xlabel("IMC")
    ax_bmi.set_ylabel("Costo del Seguro (USD)")
    ax_bmi.legend()
    st.pyplot(fig_bmi)

# 📌 Chatbot de preguntas frecuentes sobre el seguro
faq = {
    "¿Por qué es tan caro mi seguro?": "El precio depende de tu edad, hábitos como fumar y región donde vives.",
    "¿Qué variables afectan el costo?": "Las variables principales son edad, IMC, número de hijos, si fumas y región.",
    "¿Puedo reducir el precio?": "Sí, por ejemplo si dejas de fumar o cambias a un plan con deducible más alto.",
    "¿Qué significa IMC?": "IMC es el Índice de Masa Corporal y afecta el riesgo para las aseguradoras.",
    "Gracias": "¡Con gusto! Si tienes más dudas, aquí estaré 😊",
    "Adiós": "¡Hasta luego! Cuídate mucho.",
}

if "activar_chatbot" not in st.session_state:
    st.session_state.activar_chatbot = False

if st.sidebar.button("¿Tienes dudas sobre tu seguro? 🤔"):
    st.session_state.activar_chatbot = True

if st.session_state.activar_chatbot:
    st.markdown("---")
    st.markdown("### 🤖 Preguntas frecuentes sobre tu seguro médico")
    st.markdown("Haz clic en una pregunta para ver la respuesta:")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for mensaje in st.session_state.messages:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    for pregunta, respuesta in faq.items():
        if st.button(pregunta):
            st.session_state.messages.append({"role": "user", "content": pregunta})
            with st.chat_message("user"):
                st.markdown(pregunta)

            st.session_state.messages.append({"role": "assistant", "content": respuesta})
            with st.chat_message("assistant"):
                st.markdown(respuesta)

