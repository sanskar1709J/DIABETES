import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Diabetes Risk & 10-Year Simulator", page_icon="🩺", layout="wide")

# Load model artifacts safely with error handling
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('feature_names.pkl')
        return model, scaler, features, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, feature_names, error_msg = load_artifacts()

st.title("🩺 AI-Driven Diabetes Risk & 10-Year Progression Simulator")

if error_msg:
    st.error(f"⚠️ **Deployment Error:** Could not find or load model files (`.pkl`). Please make sure `diabetes_model.pkl`, `scaler.pkl`, and `feature_names.pkl` are uploaded to your GitHub repository root folder.")
    st.info(f"Technical details: {error_msg}")
else:
    st.markdown("Enter patient clinical metrics below to calculate an immediate risk score and view long-term health projections.")

    # --- SIDEBAR INPUTS ---
    st.sidebar.header("Patient Clinical Metrics")
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose Level (mg/dL)", 0, 200, 117)
    bp = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 122, 72)
    skin = st.sidebar.slider("Skin Thickness (mm)", 0, 99, 23)
    insulin = st.sidebar.slider("Insulin (mu U/ml)", 0, 846, 30)
    bmi = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.05, 2.42, 0.37)
    age = st.sidebar.slider("Age (Years)", 21, 81, 33)

    # --- FEATURE ENGINEERING ---
    glucose_insulin_ratio = glucose / (insulin + 1)
    bmi_age = bmi * age
    glucose_bmi = glucose * bmi

    raw_input_dict = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age,
        'Glucose_Insulin_Ratio': glucose_insulin_ratio,
        'BMI_Age': bmi_age,
        'Glucose_BMI': glucose_bmi
    }

    input_df = pd.DataFrame([[raw_input_dict.get(col, 0) for col in feature_names]], columns=feature_names)

    # Scale and Predict Current Risk
    input_scaled = scaler.transform(input_df)
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
        st.subheader("💡 Model Insights")
        st.markdown("Your risk score is generated using an advanced machine learning model evaluating metabolic metrics and interaction ratios.")

    st.markdown("---")

    # --- 10-YEAR PROGRESSION SIMULATOR ---
    st.subheader("📈 10-Year Health Risk Progression Forecast")
    st.markdown("This simulation models how your diabetes risk evolves over the next decade assuming minor annual metabolic changes.")

    years = list(range(0, 11))
    simulated_risks = []

    for year in years:
        sim_age = age + year
        sim_glucose = glucose + (year * 0.4)
        sim_bmi = bmi + (year * 0.1)
        
        sim_gi = sim_glucose / (insulin + 1)
        sim_ba = sim_bmi * sim_age
        sim_gb = sim_glucose * sim_bmi
        
        sim_dict = {
            'Pregnancies': pregnancies,
            'Glucose': sim_glucose,
            'BloodPressure': bp,
            'SkinThickness': skin,
            'Insulin': insulin,
            'BMI': sim_bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': sim_age,
            'Glucose_Insulin_Ratio': sim_gi,
            'BMI_Age': sim_ba,
            'Glucose_BMI': sim_gb
        }
        
        sim_df = pd.DataFrame([[sim_dict.get(col, 0) for col in feature_names]], columns=feature_names)
        sim_scaled = scaler.transform(sim_df)
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
