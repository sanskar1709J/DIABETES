import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Clinical Diabetes Risk & 10-Year Simulator", page_icon="🩺", layout="wide")

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

st.title("🩺 Clinical AI-Driven Diabetes Risk & 10-Year Progression Simulator")
st.markdown("Enter exact clinical lab values and patient vitals to calculate an immediate risk score powered by an optimized, probability-calibrated XGBoost model.")

# --- SIDEBAR INPUTS (Exact Clinical Metrics) ---
st.sidebar.header("Patient Clinical Metrics")
pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 0, 200, 117)
bp = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 122, 72)
skin = st.sidebar.slider("Skin Thickness (mm)", 0, 99, 23)
insulin = st.sidebar.slider("Insulin (mu U/ml)", 0, 846, 30)
bmi = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.05, 2.42, 0.37)
age = st.sidebar.slider("Age (Years)", 21, 81, 33)

# --- FEATURE ENGINEERING (Matches training) ---
glucose_insulin_ratio = glucose / (insulin + 1)
bmi_age = bmi * age
glucose_bmi = glucose * bmi

# Assemble input vector matching exact feature order
input_features = np.array([[
    pregnancies, glucose, bp, skin, insulin, bmi, dpf, age,
    glucose_insulin_ratio, bmi_age, glucose_bmi
]])

# Scale and Predict Current Risk
input_scaled = scaler.transform(input_features)
current_risk_proba = model.predict_proba(input_scaled)[0][1] * 100

# --- DASHBOARD LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Current Risk Assessment")
    st.metric(label="Calculated Risk Probability", value=f"{current_risk_proba:.1f}%")
    
    if current_risk_proba > 50:
        st.error("⚠️ **High Risk Status:** Recommended lifestyle intervention or clinical consultation.")
    else:
        st.success("✅ **Low Risk Status:** Maintain healthy habits and routine check-ups.")

with col2:
    st.subheader("💡 Clinical Model Insights")
    st.markdown("This model evaluates exact metabolic markers (like glucose-to-insulin dynamics and BMI-age scaling). Try sliding **Glucose above 180** and **BMI above 35** to see the probability flexibly scale past **90%**!")

st.markdown("---")

# --- 10-YEAR PROGRESSION SIMULATOR ---
st.subheader("📈 10-Year Health Risk Progression Forecast")
st.markdown("This simulation models how your diabetes risk evolves over the next decade assuming slight annual metabolic shifts (incremental age and gradual glucose/BMI increases).")

years = list(range(0, 11))
simulated_risks = []

for year in years:
    sim_age = age + year
    sim_glucose = glucose + (year * 0.4)
    sim_bmi = bmi + (year * 0.1)
    
    sim_gi_ratio = sim_glucose / (insulin + 1)
    sim_bmi_age = sim_bmi * sim_age
    sim_glucose_bmi = sim_glucose * sim_bmi
    
    sim_input = np.array([[
        pregnancies, sim_glucose, bp, skin, insulin, 
        sim_bmi, dpf, sim_age,
        sim_gi_ratio, sim_bmi_age, sim_glucose_bmi
    ]])
    
    sim_scaled = scaler.transform(sim_input)
    risk_val = model.predict_proba(sim_scaled)[0][1] * 100
    simulated_risks.append(risk_val)

# Plotting
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(years, simulated_risks, marker='o', color='#ff4b4b', linewidth=2.5, markersize=6, label='Projected Risk (%)')
ax.axhline(y=50, color='orange', linestyle='--', label='Critical Risk Threshold (50%)')

ax.set_title("Decade Health Trajectory", fontsize=14, fontweight='bold')
ax.set_xlabel("Years into Future", fontsize=11)
ax.set_ylabel("Predicted Risk (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()

st.pyplot(fig)
