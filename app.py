mport streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# === Load models ===
xgb_model = joblib.load('xgboost_model.pkl')
lgbm_model = joblib.load('lightgbm_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# === Title Section with Bank Logo and Background ===
st.markdown(
    """
    <div style="background-color:#e8f0fe;padding:20px;border-radius:10px;display:flex;justify-content:space-between;align-items:center;">
        <div style="font-size:32px;font-weight:bold;">Loan Default Prediction Comparison App</div>
        <img src="https://img.icons8.com/ios-filled/50/000000/bank.png" width="40"/>
    </div>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <div style="margin-top:10px;margin-bottom:20px;color:#333;">
        Try different feature values to simulate a loan applicant and see predictions across models.
    </div>
    """, unsafe_allow_html=True
)

# === Input Panel ===
with st.expander("Adjust Input Features"):
    RevolvingUtilizationOfUnsecuredLines = st.slider("Revolving Utilization of Unsecured Lines", 0.0, 1.0, 0.1)
    age = st.slider("Age", 18, 100, 35)
    NumberOfTime30_59DaysPastDueNotWorse = st.slider("Number of Times 30-59 Days Past Due Not Worse", 0, 10, 0)
    DebtRatio = st.slider("Debt Ratio", 0.0, 10.0, 0.5)
    MonthlyIncome = st.slider("Monthly Income", 0.0, 50000.0, 5000.0)
    NumberOfOpenCreditLinesAndLoans = st.slider("Number of Open Credit Lines and Loans", 0, 20, 5)
    NumberOfTimes90DaysLate = st.slider("Number of Times 90 Days Late", 0, 10, 0)
    NumberRealEstateLoansOrLines = st.slider("Number of Real Estate Loans or Lines", 0, 10, 1)
    NumberOfTime60_89DaysPastDueNotWorse = st.slider("Number of Times 60-89 Days Past Due Not Worse", 0, 10, 0)
    NumberOfDependents = st.slider("Number of Dependents", 0, 10, 0)

# === Predict Button ===
if st.button("Predict"):
    input_data = pd.DataFrame([[
        RevolvingUtilizationOfUnsecuredLines,
        age,
        NumberOfTime30_59DaysPastDueNotWorse,
        DebtRatio,
        MonthlyIncome,
        NumberOfOpenCreditLinesAndLoans,
        NumberOfTimes90DaysLate,
        NumberRealEstateLoansOrLines,
        NumberOfTime60_89DaysPastDueNotWorse,
        NumberOfDependents
    ]])

    input_scaled = scaler.transform(input_data)

    models = {
        "XGBoost": xgb_model,
        "LightGBM": lgbm_model,
        "Random Forest": rf_model
    }

    results = []
    for name, model in models.items():
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] * 100
        results.append({
            "Model": name,
            "Prediction": f"{int(pred)} ({'Default' if pred == 1 else 'No Default'})",
            "Probability (%)": f"{prob:.2f}%"
        })

    df_results = pd.DataFrame(results)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:24px;font-weight:bold;'>Prediction Comparison</div>", unsafe_allow_html=True)
    st.dataframe(df_results)

    # Bar Chart
    st.markdown("<div style='font-size:22px;font-weight:bold;margin-top:25px;'>Model Probability Comparison</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    ax.bar(df_results['Model'], [float(x.strip('%')) for x in df_results['Probability (%)']], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Probability of Default (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    # CSV download
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name='loan_predictions.csv', mime='text/csv')
