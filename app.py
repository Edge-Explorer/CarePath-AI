"""
Streamlit app that uses:
- disease_model.pkl (RandomForest) for prediction & confidence
- llm_pipeline.py (extracted from Notebook 2) for formatting, retrieval, and reasoning (uses Ollama if available)
- patient.db for saving records (existing DB schema)
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sqlite3
import json
from typing import Dict, Any

from llm_pipeline import (
    format_symptoms,
    build_vectorstore,
    retrieve_node,
    predict_disease_state,
    reasoning_node,
    generate_fallback_reasoning,
    PatientState
)

# -------------------------
# Config (matches your notebook)
# -------------------------
MODEL_PATH = "notebooks/disease_model.pkl"  # same as in Notebook 2
DB_PATH = "patient.db"

st.set_page_config(page_title="CarePath AI", layout="wide")

# -------------------------
# Load model (same as notebook)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    model = joblib.load(path)
    feature_names = None
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    return model, feature_names

@st.cache_resource
def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    # assume notebook created table already; otherwise create as notebook does
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS patient_records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symptoms TEXT,
        prediction INTEGER,
        reasoning TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    return conn

@st.cache_resource
def load_vectorstore(db_path: str = DB_PATH):
    # This will return None if required libs are not installed or there are no records
    return build_vectorstore(db_path)

# -------------------------
# UI helpers
# -------------------------
def save_prediction(conn, symptoms: Dict[str, Any], prediction: int, reasoning: str):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO patient_records (symptoms, prediction, reasoning) VALUES (?, ?, ?)",
        (str(symptoms), int(prediction), reasoning)
    )
    conn.commit()
    return cur.lastrowid

# -------------------------
# UI
# -------------------------
st.title("🩺 CarePath AI — (Notebook-2 powered)")

st.markdown("This app uses your `disease_model.pkl` for prediction and the Notebook-2 LangGraph / Ollama reasoning pipeline for explanations.")

model, feature_names = load_model(MODEL_PATH)
conn = init_db(DB_PATH)
vectorstore = load_vectorstore(DB_PATH)

st.info(f"Model loaded. Detected features: {len(feature_names) if feature_names else 'unknown'}")

# Friendly form (same features as Notebook sample)
col1, col2 = st.columns(2)
user_symptoms = {}

with col1:
    user_symptoms["Disease"] = st.selectbox("Known Disease (optional)", ["None", "Diabetes", "Asthma", "Heart Disease", "Hypertension"])
    user_symptoms["Fever"] = 1 if st.radio("Fever", ["No", "Yes"]) == "Yes" else 0
    user_symptoms["Cough"] = 1 if st.radio("Cough", ["No", "Yes"]) == "Yes" else 0
    user_symptoms["Fatigue"] = 1 if st.radio("Fatigue", ["No", "Yes"]) == "Yes" else 0
    user_symptoms["Difficulty Breathing"] = 1 if st.radio("Difficulty Breathing", ["No", "Yes"]) == "Yes" else 0

with col2:
    user_symptoms["Cholesterol Level"] = st.slider("Cholesterol Level (mg/dL)", 100, 400, 180)
    user_symptoms["Age"] = st.slider("Age (years)", 0, 120, 30)
    user_symptoms["Gender"] = st.selectbox("Gender", ["Male", "Female", "Other"])
    user_symptoms["Blood Pressure"] = st.slider("Blood Pressure (systolic mmHg)", 80, 200, 120)

# Ensure categorical string values are mapped to numeric like the notebook expects
def encode_for_model(symptoms):
    s = symptoms.copy()
    gender_map = {"Male": 1, "Female": 0, "Other": 2}
    s["Gender"] = gender_map.get(s.get("Gender", "Female"), 0)
    disease_map = {"None": 0, "Diabetes": 1, "Asthma": 2, "Heart Disease": 3, "Hypertension": 4}
    s["Disease"] = disease_map.get(s.get("Disease", "None"), 0)
    return s

if st.button("🔮 Predict"):
    encoded = encode_for_model(user_symptoms)
    # Build initial state as in Notebook
    state: PatientState = {
        "symptoms": encoded,
        "prediction": None,
        "reasoning": None,
        "retrieved_cases": None
    }

    # 1) predict
    try:
        state = predict_disease_state(state, model, model_feature_names=feature_names)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        raise

    prediction = state.get("prediction")
    st.subheader("✅ Prediction Result")
    # Map classes_ to meaningful label if available
    label = prediction
    if hasattr(model, "classes_"):
        try:
            classes = list(model.classes_)
            # If prediction is an int class index, map; else show as-is
            if isinstance(prediction, (int, np.integer)) and prediction < len(classes):
                label = str(classes[int(prediction)])
            else:
                label = str(prediction)
        except Exception:
            label = str(prediction)
    st.metric("Predicted disease", label)

    # Confidence calculation using predict_proba
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        X = format_symptoms(encoded, model_feature_names=feature_names)
        try:
            probs = model.predict_proba(X)[0]
            pred_index = int(np.argmax(probs))
            confidence = float(probs[pred_index])
        except Exception:
            confidence = 0.0

    st.write(f"Confidence: **{confidence*100:.2f}%**")
    st.progress(min(max(confidence, 0.0), 1.0))

    # 2) retrieval (optional)
    state = retrieve_node(state, vectorstore)

    # 3) reasoning (this uses your notebook logic and Ollama if available)
    with st.spinner("Generating reasoning via Notebook-2 pipeline (Ollama if available)..."):
        state = reasoning_node(state, vectorstore=vectorstore)

    st.subheader("📖 Reasoning (from Notebook-2)")
    st.markdown(state.get("reasoning", generate_fallback_reasoning(encoded, prediction)))

    # Save if user wants
    if st.checkbox("💾 Save this prediction to DB (same as notebook)", value=True):
        rec_id = save_prediction(conn, encoded, prediction, state.get("reasoning", ""))
        st.success(f"Saved to DB (id={rec_id})")

# Show last 5 records
with st.expander("Recent records (patient.db)"):
    try:
        df = pd.read_sql_query("SELECT id, symptoms, prediction, reasoning, created_at FROM patient_records ORDER BY created_at DESC LIMIT 5", conn)
        st.dataframe(df)
    except Exception as ex:
        st.write("Could not read DB:", ex)


