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

# تحديد المسار المطلق لمجلد المشروع
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
