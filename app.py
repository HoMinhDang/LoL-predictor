import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from src.config import EARLY_FEATURES, ROOT_DIR
from src.data.loader import load_csv, filter_team_rows
from src.data.preprocessor import get_nan_league_report
from src.models.svm_model import SVMModel
from src.models.rf_model import RFModel
from src.models.xgb_model import XGBModel

MODELS_DIR = os.path.join(ROOT_DIR, 'models_saved')

FEATURE_LABELS = {
    'golddiffat15': 'Gold Diff @15min',
    'xpdiffat15': 'XP Diff @15min',
    'csdiffat15': 'CS Diff @15min',
    'killsat15': 'Kills @15min',
    'assistsat15': 'Assists @15min',
    'deathsat15': 'Deaths @15min',
    'opp_killsat15': 'Opp Kills @15min',
    'opp_assistsat15': 'Opp Assists @15min',
    'opp_deathsat15': 'Opp Deaths @15min',
    'firstblood': 'First Blood',
    'firstdragon': 'First Dragon',
    'firstherald': 'First Herald',
    'firsttower': 'First Tower',
    'firstmidtower': 'First Mid Tower',
    'firsttothreetowers': 'First to 3 Towers',
    'void_grubs': 'Void Grubs',
}

BINARY_FEATURES = {
    'firstblood', 'firstdragon', 'firstherald',
    'firsttower', 'firstmidtower', 'firsttothreetowers',
}

MODEL_WRAPPERS = {
    'svm': SVMModel,
    'rf': RFModel,
    'xgb': XGBModel,
}


@st.cache_resource
def load_saved_models():
    available = {}
    metadata = None
    scaler = None

    meta_path = os.path.join(MODELS_DIR, 'metadata.json')
    if not os.path.exists(meta_path):
        return available, metadata, scaler

    with open(meta_path) as f:
        metadata = json.load(f)

    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

    for model_name in metadata.get('available_models', ['svm', 'rf', 'xgb']):
        model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
        if os.path.exists(model_path):
            wrapper = MODEL_WRAPPERS[model_name]()
            wrapper.load(model_path)
            available[model_name] = wrapper

    return available, metadata, scaler


@st.cache_data
def load_data():
    df = load_csv()
    return filter_team_rows(df)


def render_predict_tab(available_models, scaler, metadata):
    st.header("Predict Winner")

    if not available_models:
        st.warning("No trained models found in models_saved/. Run `python -m src.train` first.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Early Game Stats (≤15 min)")
        input_values = {}
        for feat in EARLY_FEATURES:
            label = FEATURE_LABELS.get(feat, feat)
            if feat in BINARY_FEATURES:
                input_values[feat] = 1 if st.checkbox(label, value=False) else 0
            else:
                input_values[feat] = st.number_input(label, value=0, step=1)

    with col2:
        st.subheader("Model")
        model_name = st.selectbox(
            "Select Model",
            list(available_models.keys()),
            index=0,
        )

        if metadata:
            st.caption(f"Trained on: {metadata.get('league', 'all')} "
                       f"({metadata.get('n_samples', '?')} samples)")
            if model_name in metadata:
                m = metadata[model_name]
                st.info(
                    f"Accuracy: {m['metrics']['accuracy']:.3f}\n"
                    f"ROC-AUC: {m['metrics']['roc_auc']:.3f}"
                )

    if st.button("Predict", type="primary"):
        model = available_models[model_name]
        input_array = np.array([[input_values[feat] for feat in EARLY_FEATURES]])
        input_scaled = scaler.transform(input_array)

        proba = model.predict_proba(input_scaled)[0]
        pred = model.predict(input_scaled)[0]

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            if pred == 1:
                st.markdown("<h2 style='color:green;text-align:center'>WIN 🏆</h2>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color:red;text-align:center'>LOSS 💀</h2>",
                            unsafe_allow_html=True)
        with col_b:
            confidence = proba[pred] * 100
            st.metric("Confidence", f"{confidence:.1f}%")

        chart_df = pd.DataFrame({
            'Outcome': ['Win', 'Loss'],
            'Probability': [proba[1], proba[0]],
        })
        st.bar_chart(chart_df.set_index('Outcome'))


def render_data_tab():
    st.header("Data Explorer")
    df = load_data()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Leagues", df['league'].nunique())
    with col3:
        st.metric("Win Rate", f"{df['result'].mean()*100:.1f}%")

    league_report = get_nan_league_report(df)
    nan_leagues = league_report[league_report['rows_with_nan'] > 0]
    if len(nan_leagues) > 0:
        with st.expander("NaN Report by League"):
            st.dataframe(league_report, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="LoL Early Game Winner Prediction",
        page_icon="🏆",
        layout="wide",
    )

    available_models, metadata, scaler = load_saved_models()

    st.title("LoL Early Game Winner Prediction")
    st.caption("Predict match winners using early-game stats (< 15 min)")

    if not available_models:
        st.error(
            "No trained models found. "
            "Run `python -m src.train` to train and save models "
            "in `models_saved/`."
        )
        st.stop()

    tab1, tab2 = st.tabs(["Predict Winner", "Data Explorer"])

    with tab1:
        render_predict_tab(available_models, scaler, metadata)
    with tab2:
        render_data_tab()


if __name__ == '__main__':
    main()
