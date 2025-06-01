import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import io

# ===== Page Setup =====
st.set_page_config(page_title="Loan Default Prediction", page_icon="💳", layout="centered")

# Load models and scaler
xgb_model = joblib.load('xgboost_model.pkl')
lgbm_model = joblib.load('lightgbm_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature names
feature_names = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
                 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

# ===== Title & Intro =====
st.markdown("""
    <h1 style='text-align: center; color: #0A79DF;'>💳 Loan Default Prediction Comparison App</h1>
    <p style='text-align: center; color: #6c757d;'>Enter the applicant’s data below. This app compares loan default risk using 3 machine learning models.</p>
""", unsafe_allow_html=True)

# ===== Input Form =====
st.markdown("### 🧾 Applicant Information")
user_input = []
for feature in feature_names:
    val = st.number_input(f"➤ {feature}", min_value=0.0, format="%.2f")
    user_input.append(val)

# ===== Prediction Button =====
if st.button("📊 Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # ===== Model Predictions =====
    results = []
    for name, model in [('XGBoost', xgb_model), ('LightGBM', lgbm_model), ('Random Forest', rf_model)]:
        pred = int(model.predict(input_scaled)[0])
        proba = round(model.predict_proba(input_scaled)[0][1] * 100, 2)
        results.append({
            'Model': name,
            'Prediction': f"{pred} ({'Default' if pred == 1 else 'No Default'})",
            'Probability (%)': f"{proba}%"
        })

    results_df = pd.DataFrame(results)

    st.markdown("### 📈 Prediction Results")
    st.success("✅ Prediction saved to Excel successfully!")
    st.dataframe(results_df.style.set_properties(**{'text-align': 'center'}))

    # ===== Save to Excel =====
    prediction_record = {
        'Input': str(user_input),
        'XGBoost_Prediction': int(xgb_model.predict(input_scaled)[0]),
        'XGBoost_Probability': round(xgb_model.predict_proba(input_scaled)[0][1] * 100, 2),
        'LightGBM_Prediction': int(lgbm_model.predict(input_scaled)[0]),
        'LightGBM_Probability': round(lgbm_model.predict_proba(input_scaled)[0][1] * 100, 2),
        'RandomForest_Prediction': int(rf_model.predict(input_scaled)[0]),
        'RandomForest_Probability': round(rf_model.predict_proba(input_scaled)[0][1] * 100, 2),
    }

    df_row = pd.DataFrame([prediction_record])
    if os.path.exists("prediction_log.xlsx"):
        df_old = pd.read_excel("prediction_log.xlsx")
        df_all = pd.concat([df_old, df_row], ignore_index=True)
    else:
        df_all = df_row

    df_all.to_excel("prediction_log.xlsx", index=False)

    # ===== Excel Download Button =====
    excel_buffer = io.BytesIO()
    df_all.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)
    st.download_button(
        label="📥 Download Excel File",
        data=excel_buffer,
        file_name="prediction_log.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ===== Model Performance Table =====
st.markdown("### 📊 Model Performance Summary")
df_perf = pd.DataFrame({
    "Model": ["XGBoost", "LightGBM", "Random Forest"],
    "Accuracy": [0.935, 0.934, 0.928],
    "F1-score": [0.30, 0.32, 0.28],
    "ROC AUC": [0.86, 0.87, 0.84]
})
st.dataframe(df_perf.style.set_properties(**{'text-align': 'center'}))

    