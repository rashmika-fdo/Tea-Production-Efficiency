# Kalubowitiyana Tea Yield Optimizer

**Tea Production Decision Support System | ML-Powered Efficiency Tool**

🔗 **Live App**: [Tea Yield Optimizer – Kalubowitiyana CTC Factory](https://tea-yield-optimizer-demo.streamlit.app/)


---

## 🔹 Project Overview

This project is a **Machine Learning-powered decision support system** designed for the Kalubowitiyana CTC Tea Factory. It helps factory operators and managers **optimize daily tea processing parameters** to achieve the **maximum possible yield** while staying within safe operational constraints.

Using historical production data, daily **input tea amount** (kg) and **rainfall** (mm), the app predicts:

- Optimal process parameters for fermentation, humidity, inlet temperature, and outlet temperature.
- Expected tea yield (kg) and yield percentage.
- Traffic-light indicators for profitability assessment.

The system allows **real-time operator adjustments** with immediate feedback on predicted yields.

---

## 🔹 Problem Statement

Tea production at CTC factories involves multiple process variables that directly affect yield. Operators often rely on experience or heuristics, leading to:

- Sub-optimal yield
- Yield variability due to environmental factors (e.g., rainfall)
- Difficulty in quantifying trade-offs between process settings and output

This app provides a **data-driven, reproducible solution** to optimize yield.

---

## 🔹 Key Features

1. **Daily Inputs**
   - Input tea amount (kg)
   - Daily rainfall (mm)

2. **Optimal Settings Prediction**
   - Suggests **ideal fermentation, humidity, inlet & outlet temperatures**
   - Shows predicted output and yield percentage
   - Traffic-light indicator for profitability:
     - 🟢 Green: Yield ≥ 21.5% (profit-maximizing)
     - 🟡 Yellow: Yield 20–21.5% (acceptable)
     - 🔴 Red: Yield < 20% (low)

3. **Operator Controls**
   - Sliders to manually adjust process variables
   - Real-time predicted yield for adjusted settings
   - Comparison with model-suggested optimal yield

4. **Interactive Visual Feedback**
   - Metrics display for output and yield
   - Profitability guidance via light indicators
   - Comparison of adjusted vs optimal settings

---

## 🔹 Technical Workflow

### **Data & Model**
- **Dataset:** production data (3,000 rows) with columns: `Input`, `Rain`, `Ferment`, `Humidity`, `Inlet`, `Outlet`, `Output`.
- **ML Model:** XGBoost Regressor trained to predict `Output` based on input variables.
- **Evaluation Metrics:** R², RMSE, MAPE, 5-fold Cross Validation.
- **Optimization Function:** Randomized search within factory-defined ranges to suggest optimal parameters.

### **Mechanism**
1. Users input **daily tea amount and rainfall**.
2. ML model predicts output for **randomized sets of process parameters** within factory constraints.
3. **Top-performing settings** are displayed as model-recommended optimal parameters.
4. Operators can **fine-tune the process** using sliders.
5. The app **predicts yield for the adjusted settings** and compares it with optimal.
6. Traffic-light system indicates profitability threshold.

---

## 🔹 Technology Stack

- **Frontend & App Deployment:** [Streamlit](https://streamlit.io)
- **Machine Learning:** [XGBoost](https://xgboost.ai), [Scikit-learn](https://scikit-learn.org)
- **Data Handling:** Pandas, Numpy
- **Model Serialization:** Joblib
- **Version Control & Deployment:** GitHub & Streamlit Community Cloud
- **Visualization:** Streamlit metrics and custom HTML/CSS for traffic-light indicators

---
