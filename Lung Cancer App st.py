import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")
st.title("🫁 Lung Cancer Risk Prediction App")

# Load trained model and scaler
try:
    with open('models/trained_logreg.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('scaler/trained_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model or scaler file not found. Ensure 'models/trained_logreg.pkl' and 'scaler/trained_scaler.pkl' exist.")
    st.stop()

st.sidebar.header("Patient Details")

# Collect inputs (matching your Flask form fields)
AGE = st.sidebar.slider("Age", 18, 100, 40)
GENDER = st.sidebar.selectbox("Gender", ["Male", "Female"])
SMOKING = st.sidebar.selectbox("Do you smoke?", ["Yes", "No"])
FINGER_DISCOLORATION = st.sidebar.selectbox("Finger Discoloration", ["Yes", "No"])
MENTAL_STRESS = st.sidebar.selectbox("Mental Stress", ["Yes", "No"])
EXPOSURE_TO_POLLUTION = st.sidebar.selectbox("Exposure to Pollution", ["Yes", "No"])
LONG_TERM_ILLNESS = st.sidebar.selectbox("Long-term Illness", ["Yes", "No"])
ENERGY_LEVEL = st.sidebar.slider("Energy Level (0-100)", 0, 100, 50)
IMMUNE_WEAKNESS = st.sidebar.selectbox("Immune Weakness", ["Yes", "No"])
BREATHING_ISSUE = st.sidebar.selectbox("Breathing Issues", ["Yes", "No"])
ALCOHOL_CONSUMPTION = st.sidebar.selectbox("Alcohol Consumption", ["Yes", "No"])
THROAT_DISCOMFORT = st.sidebar.selectbox("Throat Discomfort", ["Yes", "No"])
OXYGEN_SATURATION = st.sidebar.slider("Oxygen Saturation (%)", 70, 100, 95)
CHEST_TIGHTNESS = st.sidebar.selectbox("Chest Tightness", ["Yes", "No"])
FAMILY_HISTORY = st.sidebar.selectbox("Family History of Lung Disease", ["Yes", "No"])
SMOKING_FAMILY_HISTORY = st.sidebar.selectbox("Family Smoking History", ["Yes", "No"])
STRESS_IMMUNE = st.sidebar.selectbox("Stress Affecting Immunity", ["Yes", "No"])

# Prediction Button
if st.button("Predict Lung Cancer Risk"):

    # Mapping inputs as per model expectation
    features = [
        AGE,
        1 if GENDER == "Male" else 0,
        1 if SMOKING == "Yes" else 0,
        1 if FINGER_DISCOLORATION == "Yes" else 0,
        1 if MENTAL_STRESS == "Yes" else 0,
        1 if EXPOSURE_TO_POLLUTION == "Yes" else 0,
        1 if LONG_TERM_ILLNESS == "Yes" else 0,
        ENERGY_LEVEL,
        1 if IMMUNE_WEAKNESS == "Yes" else 0,
        1 if BREATHING_ISSUE == "Yes" else 0,
        1 if ALCOHOL_CONSUMPTION == "Yes" else 0,
        1 if THROAT_DISCOMFORT == "Yes" else 0,
        OXYGEN_SATURATION,
        1 if CHEST_TIGHTNESS == "Yes" else 0,
        1 if FAMILY_HISTORY == "Yes" else 0,
        1 if SMOKING_FAMILY_HISTORY == "Yes" else 0,
        1 if STRESS_IMMUNE == "Yes" else 0
    ]

    features_array = np.array([features])
    scaled_features = scaler.transform(features_array)

    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]

    st.subheader("Prediction Result:")
    if probability >= 0.8:
        st.error(f"✅ Diagnosed with Lung Cancer (Pulmonary Disease)\nProbability: {probability*100:.2f}%")
    elif probability >= 0.6:
        st.warning(f"🟡 Likely to be diagnosed with Lung Cancer (Borderline)\nProbability: {probability*100:.2f}%")
    elif 0.4 < probability < 0.6:
        st.info(f"⚠️ Uncertain Result\nProbability: {probability*100:.2f}%")
    else:
        st.success(f"❌ Healthy\nProbability: {probability*100:.2f}%")

st.caption("Disclaimer: This tool provides risk predictions and is not a diagnostic system. Please consult a healthcare professional.")
