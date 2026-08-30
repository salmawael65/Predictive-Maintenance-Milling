# ============================================================
# STREAMLIT GUI - MACHINE FAILURE PREDICTION
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# ============================================================
# CONFIGURATION & DATASET LOADING
# ============================================================

MODEL_PATH = "Best_XGBoost_Dataset_A.pkl"
DATASET_PATH = "predictive_maintenance.csv"  # ضع المسار الصحيح لملف البيانات لديك

# Load Main XGBoost model
import os
import joblib

# تحديد المسار المطلق لمجلد المشروع على السحابة
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Best_XGBoost_Dataset_A.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset_a_final_results (1).csv")


@st.cache_resource
def load_main_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ ملف النموذج غير موجود بالمسار: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

best_xgb = load_main_model()

# Load Dataset for failure-type training
@st.cache_data
def load_data():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return None


df = load_data()

# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FAILURE_TYPES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

# ============================================================
# TRAIN FAILURE TYPE MODELS
# ============================================================


@st.cache_resource
def train_failure_type_models():
    if df is None:
        return {}

    type_models = {}
    required_columns = list(FAILURE_TYPES.keys())

    if not all(col in df.columns for col in required_columns):
        return {}

    X_type = df[FEATURES].copy()

    for failure_code, failure_name in FAILURE_TYPES.items():
        y_type = df[failure_code].astype(int)

        X_train_type, X_test_type, y_train_type, y_test_type = (
            train_test_split(
                X_type, y_type, test_size=0.20, random_state=42, stratify=y_type
            )
        )

        model = XGBClassifier(
            n_estimators=200,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )

        categorical = ["Type"]
        numerical = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical,
                ),
                ("numerical", "passthrough", numerical),
            ]
        )

        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )

        pipeline.fit(X_train_type, y_train_type)
        type_models[failure_code] = pipeline

    return type_models


# ============================================================
# PREDICTION FUNCTION
# ============================================================


def predict_machine(input_data):
    input_df = pd.DataFrame([input_data])

    # STEP 1: MACHINE FAILURE
    failure_prediction = best_xgb.predict(input_df)[0]
    failure_probability = best_xgb.predict_proba(input_df)[0][1]

    # STEP 2: FAILURE TYPE
    failure_types_detected = []
    type_probabilities = {}

    type_models = train_failure_type_models()

    if failure_prediction == 1 and len(type_models) > 0:
        for code, model in type_models.items():
            probability = model.predict_proba(input_df)[0][1]
            type_probabilities[code] = probability

            if probability >= 0.50:
                failure_types_detected.append(FAILURE_TYPES[code])

    return (
        failure_prediction,
        failure_probability,
        failure_types_detected,
        type_probabilities,
    )


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Machine Failure Prediction", page_icon="⚙️", layout="wide"
)

st.title("⚙️ Machine Failure Prediction System")
st.write(
    "Enter the machine operating parameters to predict whether the machine will experience a failure and identify the possible failure type."
)
st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Machine Parameters")

machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])

air_temperature = st.sidebar.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=300.0,
    step=0.1,
)

process_temperature = st.sidebar.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=310.0,
    step=0.1,
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed [rpm]",
    min_value=500,
    max_value=5000,
    value=1500,
    step=10,
)

torque = st.sidebar.number_input(
    "Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0, step=0.1
)

tool_wear = st.sidebar.number_input(
    "Tool Wear [min]", min_value=0, max_value=300, value=100, step=1
)

# INPUT DATA
input_data = {
    "Type": machine_type,
    "Air temperature [K]": air_temperature,
    "Process temperature [K]": process_temperature,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
}

# ============================================================
# DISPLAY INPUT & PREDICT
# ============================================================

st.subheader("Machine Input")
input_display = pd.DataFrame(
    {"Parameter": list(input_data.keys()), "Value": list(input_data.values())}
)
st.dataframe(input_display, use_container_width=True, hide_index=True)

if st.button("🔍 Predict Machine Status", use_container_width=True):
    with st.spinner("Analyzing machine condition..."):
        (
            failure_prediction,
            failure_probability,
            detected_types,
            type_probabilities,
        ) = predict_machine(input_data)

    st.divider()
    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)

    with col1:
        if failure_prediction == 0:
            st.success("✅ NO FAILURE")
            st.metric("Failure Probability", f"{failure_probability * 100:.2f}%")
            st.write(
                "The model predicts that the machine is operating under a non-failure condition."
            )
        else:
            st.error("🚨 MACHINE FAILURE DETECTED")
            st.metric("Failure Probability", f"{failure_probability * 100:.2f}%")
            st.write(
                "The model predicts that the machine is likely to experience a failure."
            )

    with col2:
        st.subheader("Failure Type")
        if failure_prediction == 1:
            if len(detected_types) > 0:
                for failure_type in detected_types:
                    st.warning(f"⚠️ {failure_type}")
            else:
                st.info(
                    "Failure detected, but no specific failure type exceeded the classification threshold."
                )
        else:
            st.success("No failure type detected.")

    if failure_prediction == 1 and len(type_probabilities) > 0:
        st.divider()
        st.subheader("Failure Type Probabilities")

        probability_df = pd.DataFrame(
            {
                "Failure Type": [
                    FAILURE_TYPES[code] for code in type_probabilities.keys()
                ],
                "Probability": [
                    f"{prob * 100:.2f}%"
                    for prob in type_probabilities.values()
                ],
            }
        )
        st.dataframe(
            probability_df, use_container_width=True, hide_index=True
        )

        chart_data = pd.DataFrame(
            {
                "Failure Type": [
                    FAILURE_TYPES[code] for code in type_probabilities.keys()
                ],
                "Probability": list(type_probabilities.values()),
            }
        ).set_index("Failure Type")

        st.bar_chart(chart_data)

# ============================================================
# INFORMATION
# ============================================================

st.divider()
st.subheader("📌 Failure Types")

failure_info = pd.DataFrame(
    {
        "Code": ["TWF", "HDF", "PWF", "OSF", "RNF"],
        "Failure Type": [
            "Tool Wear Failure",
            "Heat Dissipation Failure",
            "Power Failure",
            "Overstrain Failure",
            "Random Failure",
        ],
        "Description": [
            "Failure related to excessive tool wear.",
            "Failure caused by insufficient heat dissipation.",
            "Failure related to power conditions.",
            "Failure caused by excessive mechanical load.",
            "Random or unexpected machine failure.",
        ],
    }
)

st.dataframe(failure_info, use_container_width=True, hide_index=True)
