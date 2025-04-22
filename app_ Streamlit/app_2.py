import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Charger le modèle
model = pickle.load(open("C:/Users/HP/Documents/machine learning/TD_2/app_ Streamlit/RF_churn_model.pkl", "rb"))

st.title("Prédiction du Churn Client 📊")

st.write("Veuillez remplir les informations ci-dessous pour prédire le churn :")

# Inputs utilisateur
age = st.number_input("Âge", min_value=18, max_value=100, value=30)
balance = st.number_input("Solde (Balance)", value=0.0)
num_products = st.slider("Nombre de produits")
is_active = st.selectbox("Client Actif ?", options=["Oui", "Non"])
is_active = 1 if is_active == "Oui" else 0

gender = st.selectbox("Sexe", options=["Male", "Female"])
geo = st.selectbox("Géographie", options=["France", "Germany", "Spain"])

# Encodage manuel
gender_female = 1 if gender == "Female" else 0
gender_male = 1 if gender == "Male" else 0

geo_france = 1 if geo == "France" else 0
geo_germany = 1 if geo == "Germany" else 0
geo_spain = 1 if geo == "Spain" else 0

# Bouton de prédiction
if st.button("Prédire"):
    features = np.array([[age, balance, num_products, is_active,
                          geo_france, geo_germany, geo_spain,
                          gender_female, gender_male]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("✅ Le client risque de quitter.")
    else:
        st.info("ℹ️ Le client ne risque pas de quitter.")
