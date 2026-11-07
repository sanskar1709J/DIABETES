import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Page setup
st.set_page_config(page_title="Diabetes Risk & Timeline Simulator", page_icon="🩺", layout="wide")

st.title("🩺 Diabetes Risk & 10-Year Progression Timeline Simulator")
st.markdown("Enter patient metrics on the sidebar to predict current diabetes risk and forecast progression over 10 years.")

# --- Synthetic Model Training for Demo App ---
@st.cache_resource
def load_data_and_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=cols)
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, np.nan).fillna(df[col].median())

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, X.columns

model, scaler, feature_names = load_data_and_model()

# --- Sidebar Input Metrics ---
st.sidebar.header("📋 Patient Clinical Profile")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 50, 250, 115)
bp = st.sidebar.slider("Blood Pressure (mmHg)", 40, 140, 70)
skin = st.sidebar.slider("Skin Thickness (mm)", 10, 99, 25)
insulin = st.sidebar.slider("Insulin (mu U/ml)", 15, 840, 80)
bmi = st.sidebar.slider("BMI", 15.0, 60.0, 27.5)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.08, 2.50, 0.35)
age = st.sidebar.slider("Age (Years)", 18, 90, 35)

# --- Current Risk Calculation ---
patient_data = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]
patient_df = pd.DataFrame(patient_data, columns=feature_names)
patient_scaled = scaler.transform(patient_df)

current_prob = model.predict_proba(patient_scaled)[0][1] * 100

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Current Risk Score")
    if current_prob >= 50:
        st.error(f"⚠️ **HIGH RISK**: {current_prob:.1f}% Chance of Diabetes")
    else:
        st.success(f"✅ **LOW RISK**: {current_prob:.1f}% Chance of Diabetes")

    st.metric(label="Calculated Probability", value=f"{current_prob:.1f}%")

with col2:
    st.subheader("📈 10-Year Timeline Progression")

    timeline_data = []
    for yrs in range(0, 11, 2):
        sim_age = age + yrs
        sim_bmi = bmi + (yrs * 0.1)
        sim_glucose = glucose + (yrs * 1.5)
        sim_bp = bp + (yrs * 0.5)

        sim_features = pd.DataFrame([[pregnancies, sim_glucose, sim_bp, skin, insulin, sim_bmi, dpf, sim_age]], columns=feature_names)
        sim_scaled = scaler.transform(sim_features)
        p = model.predict_proba(sim_scaled)[0][1] * 100

        timeline_data.append({"Timeline": f"+{yrs} Yrs" if yrs > 0 else "Current", "Risk": p})

    t_df = pd.DataFrame(timeline_data)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.lineplot(x='Timeline', y='Risk', data=t_df, marker='o', color='#e74c3c', ax=ax, linewidth=2.5)
    ax.axhline(50, color='orange', linestyle='--', label='50% Threshold')
    ax.set_ylim(0, 105)
    ax.set_ylabel("Diabetes Risk (%)")
    ax.set_title("Forecasted Progression Over Time")
    ax.legend()
    st.pyplot(fig)

st.subheader("Data Table")
st.dataframe(t_df)
