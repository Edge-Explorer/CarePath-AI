# llm_pipeline.py
"""
Extracted pipeline functions from your Notebook 02_model_integration.ipynb
- format_symptoms
- init vectorstore (optional)
- retrieve_node
- predict_disease
- reasoning_node (uses langchain_ollama if available; else fallback)
- generate_fallback_reasoning
This mirrors Notebook 2 behavior but packaged as callable functions/classes.
"""

import sqlite3
import pandas as pd
import re
import json
from typing import TypedDict, Optional, Dict, Any

# Attempt imports used in Notebook 2
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    FAISS = None
    HuggingFaceEmbeddings = None

try:
    from langchain_ollama import OllamaLLM
except Exception:
    OllamaLLM = None

# -------------------------
# Type for state (mirrors notebook)
# -------------------------
class PatientState(TypedDict):
    symptoms: dict
    prediction: Optional[int]
    reasoning: Optional[str]
    retrieved_cases: Optional[list]

# -------------------------
# Format symptoms (same as notebook)
# -------------------------
def format_symptoms(symptoms: dict, model_feature_names: Optional[list] = None):
    """
    Reorders and fills missing features so input matches the model requirements.
    If model_feature_names is None, simply returns the dict wrapped as DataFrame.
    """
    if model_feature_names is None:
        return pd.DataFrame([symptoms])

    required_features = list(model_feature_names)
    formatted = {feature: symptoms.get(feature, 0) for feature in required_features}
    return pd.DataFrame([formatted])

# -------------------------
# Vectorstore builder (optional) using patient.db
# -------------------------
def build_vectorstore(db_path: str = "patient.db"):
    """
    Build a small FAISS vectorstore from patient.db using HuggingFaceEmbeddings.
    Returns `vectorstore` or None on failure.
    """
    if FAISS is None or HuggingFaceEmbeddings is None:
        # required libs not installed
        return None

    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT id, symptoms, prediction, reasoning FROM patient_records", conn)
        conn.close()
        if len(df) == 0:
            return None

        df["case_text"] = df.apply(
            lambda row: f"Symptoms: {row['symptoms']} | Prediction:{row['prediction']} | Reasoning: {row['reasoning']}",
            axis=1
        )
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(df["case_text"].tolist(), embeddings)
        return vectorstore
    except Exception:
        return None

# -------------------------
# Retrieval node (same logic as notebook)
# -------------------------
def retrieve_node(state: dict, vectorstore):
    """
    Retrieve similar cases from the vectorstore (k=3).
    If vectorstore is None, sets retrieved_cases to [].
    """
    if vectorstore is None:
        state["retrieved_cases"] = []
        return state

    query = str(state["symptoms"])
    try:
        results = vectorstore.similarity_search(query, k=3)
        state["retrieved_cases"] = [r.page_content for r in results]
    except Exception:
        state["retrieved_cases"] = []
    return state

# -------------------------
# Predict node (wraps model.predict)
# -------------------------
def predict_disease_state(state: PatientState, model, model_feature_names: Optional[list] = None) -> PatientState:
    """
    Takes a PatientState dict, formats symptoms and predicts using the sklearn model.
    Mutates and returns state with 'prediction' set (int).
    """
    symptoms = state["symptoms"]
    features = format_symptoms(symptoms, model_feature_names)
    prediction = model.predict(features)[0]
    state["prediction"] = int(prediction)
    return state

# -------------------------
# Ollama init & reasoning_node
# -------------------------
# initialize Ollama LLM if available similarly to your notebook
ollama_available = False
llm = None
if OllamaLLM is not None:
    try:
        # Using the same defaults as in notebook (model="mistral", streaming=False)
        llm = OllamaLLM(model="mistral", streaming=False)
        ollama_available = True
    except Exception:
        ollama_available = False

def generate_fallback_reasoning(symptoms: Dict[str, Any], prediction):
    """
    Notebook-style fallback reasoning: creates a readable summary
    """
    outcome = "Positive" if int(prediction) == 1 else "Negative"
    key_symptoms = []
    if symptoms.get("Fever", 0) == 1:
        key_symptoms.append("fever")
    if symptoms.get("Cough", 0) == 1:
        key_symptoms.append("cough")
    if symptoms.get("Fatigue", 0) == 1:
        key_symptoms.append("fatigue")
    if symptoms.get("Difficulty Breathing", 0) == 1:
        key_symptoms.append("difficulty breathing")

    reasoning = f"""
Analysis Summary:
• Patient Age: {symptoms.get('Age', 'N/A')} years
• Key Symptoms Present: {', '.join(key_symptoms) if key_symptoms else 'None reported'}
• Prediction: {outcome}

Reasoning:
• Based on the symptom profile and patient demographics
• The model indicates a {outcome.lower()} outcome
• {"Multiple symptoms suggest medical attention may be needed." if outcome == "Positive" else "Symptoms appear mild based on current assessment."}

Note: This is an automated assessment. Please consult healthcare professionals for proper medical advice.
"""
    # clean up whitespace
    reasoning = re.sub(r'\n\s+\n', '\n\n', reasoning).strip()
    return reasoning

def reasoning_node(state: dict, vectorstore=None):
    """
    Generate reasoning for the prediction using Ollama if available (mirrors Notebook 2).
    - state should have 'symptoms' and 'prediction'
    - sets state['reasoning'] to a string
    """
    symptoms = state.get("symptoms", {})
    prediction = state.get("prediction", None)

    # Prepare retrieved_text from vectorstore if present
    retrieved = state.get("retrieved_cases", [])
    retrieved_text = "\n".join(retrieved) if retrieved else "No similar cases found."

    if ollama_available and llm is not None:
        # Notebook-style prompt (kept faithful to the original)
        prompt = f"""
You are a medical assistant.

Here are some similar past cases:
{retrieved_text}

Now, based on the symptoms: {symptoms} 
and the prediction: {prediction},

Write a professional, short, readable explanation. 
- Use point-wise format.
- Include a clear conclusion at the end.
- Keep sentences concise and natural.
- Add relevant emojis for readability.
"""
        try:
            # In notebook they used llm.invoke(prompt)
            # We use the same call; if API differs in your env adjust accordingly
            reasoning_text = llm.invoke(prompt)
            # Clean up the text using same regexes as notebook
            reasoning_text = re.sub(r'\b(\w+)\s+\1\b', r'\1', reasoning_text)
            reasoning_text = re.sub(r'\s+([,.;:])', r'\1', reasoning_text)
            reasoning_text = re.sub(r'\n+', '\n', reasoning_text).strip()
        except Exception:
            reasoning_text = generate_fallback_reasoning(symptoms, prediction)
    else:
        reasoning_text = generate_fallback_reasoning(symptoms, prediction)

    state["reasoning"] = reasoning_text
    return state
