# ===============================
# SYSTEM + THREAD CONTROL
# ===============================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




# ===============================
# IMPORTS
# ===============================
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Tea Yield Optimizer",
    layout="wide"
)


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_image = get_base64_image("assets/tea_bg.jpg")

st.markdown(
    f"""
    <style>
    .title-container {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        padding: 40px;
        border-radius: 12px;
        margin-bottom: 20px;
    }}
    .title-container h1 {{
        color: white;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
    }}
    .title-container p {{
        color: #f0f0f0;
        font-size: 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)







# ===============================
# LOAD MODEL (DETERMINISTIC MODEL)
# ===============================
model = joblib.load("xgboost_output_model.pkl")

# ===============================
# SESSION STATE
# ===============================
if "optimal_result" not in st.session_state:
    st.session_state.optimal_result = None

if "optimized_df" not in st.session_state:
    st.session_state.optimized_df = None

if "predicted_result" not in st.session_state:
    st.session_state.predicted_result = None

# ===============================
# CONTROL RANGES (FACTORY LIMITS)
# ===============================
FERMENT_RANGE = (22.0, 28.0)
HUMIDITY_RANGE = (78.0, 81.0)
INLET_RANGE = (96.0, 130.0)
OUTLET_RANGE = (35.0, 45.0)

# ===============================
# OPTIMIZATION FUNCTION
# ===============================
def optimize_output(model, input_val, rainfall_val, n_trials=3000, seed=42):
    rng = np.random.default_rng(seed)
    results = []

    for _ in range(n_trials):
        ferment = rng.uniform(*FERMENT_RANGE)
        humidity = rng.uniform(*HUMIDITY_RANGE)
        inlet = rng.uniform(*INLET_RANGE)
        outlet = rng.uniform(*OUTLET_RANGE)

        X_new = pd.DataFrame([{
            "Input": float(input_val),
            "Rain": float(rainfall_val),
            "Ferment": ferment,
            "Humidity": humidity,
            "Inlet": inlet,
            "Outlet": outlet
        }])

        pred_output = model.predict(X_new)[0]
        yield_pct = (pred_output / input_val) * 100

        results.append(
            (ferment, humidity, inlet, outlet, pred_output, yield_pct)
        )

    df_res = pd.DataFrame(
        results,
        columns=[
            "Ferment", "Humidity", "Inlet", "Outlet",
            "Pred_Output", "Yield_pct"
        ]
    )

    return df_res.sort_values(by="Pred_Output", ascending=False)


# ===============================
# CACHE OPTIMIZATION (CRITICAL)
# ===============================
@st.cache_data(show_spinner=False)
def run_optimization(_model, input_tea, rainfall):
    # call optimizer with _model
    return optimize_output(_model, input_tea, rainfall).iloc[0  ]


# ===============================
# TRAFFIC LIGHT LOGIC
# ===============================
def traffic_light(yield_pct):
    if yield_pct >= 21.5:
        st.success("🟢 GREEN: Profit-maximizing yield (≥ 21.5%)")
    elif yield_pct >= 20:
        st.warning("🟡 YELLOW: Acceptable yield (20–21.5%)")
    else:
        st.error("🔴 RED: Low yield (< 20%)")

# ===============================
# TITLE
# ===============================
st.markdown(
    """
    <div class="title-container">
        <h1>Kalubowitiyana CTC Tea Factory</h1>
        <p>Tea Yield Optimizer • ML Decision Support System</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ===============================
# DAILY INPUTS
# ===============================
st.subheader("📥 Daily Inputs (Fixed for the Day)")

col1, col2 = st.columns(2)

with col1:
    input_tea = st.number_input(
        "Tea Input (kg)",
        min_value=5000.0,
        step=100.0,
        value=15000.0
    )

with col2:
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        step=0.5,
        value=12.0
    )

# ===============================
# OPTIMAL SETTINGS (MODEL IDEAL)
# ===============================
st.divider()
st.subheader("⚙️ Model-Suggested Optimal Settings (Ideal Case)")

if st.button("🔍 Find Best Yield Settings for Today"):
    with st.spinner("Optimizing for best yield..."):
        optimized_df = run_optimization(model, input_tea, rainfall)
        st.session_state.optimized_df = optimized_df
        st.session_state.optimal_result = optimized_df

if st.session_state.optimal_result is not None:
    best = st.session_state.optimal_result

    colL, colR = st.columns(2)

    with colL:
        st.metric("Optimal Output (kg)", f"{best['Pred_Output']:.2f}")
        st.metric("Optimal Yield (%)", f"{best['Yield_pct']:.2f}")
        traffic_light(best["Yield_pct"])

    with colR:
        st.write("**Recommended Best Process Settings**")
        st.write(f"• Ferment: **{best['Ferment']:.2f} °C**")
        st.write(f"• Humidity: **{best['Humidity']:.2f} %**")
        st.write(f"• Inlet: **{best['Inlet']:.2f} °C**")
        st.write(f"• Outlet: **{best['Outlet']:.2f} °C**")

# ===============================
# OPERATOR CONTROLS
# ===============================
st.divider()
st.subheader("🎛 Operator Adjustment (Within Factory Limits)")

c1, c2, c3, c4 = st.columns(4)

with c1:
    ferment_val = st.slider("Ferment (°C)", *FERMENT_RANGE, 26.0)

with c2:
    humidity_val = st.slider("Humidity (%)", *HUMIDITY_RANGE, 80.0)

with c3:
    inlet_val = st.slider("Inlet (°C)", *INLET_RANGE, 115.0)

with c4:
    outlet_val = st.slider("Outlet (°C)", *OUTLET_RANGE, 36.0)


st.markdown(
    "<p style='font-size:14px; color:gray;'>"
    "📌 <b>Target:</b> Maintain tea yield <b>≥ 21.5%</b> for optimal profitability."
    "</p>",
    unsafe_allow_html=True
)
# ===============================
# PREDICTION BUTTON
# ===============================
st.divider()
st.subheader("📈 Predicted Yield for Selected Settings")

if st.button("📊 Predict Yield"):
    X_user = pd.DataFrame([{
        "Input": float(input_tea),
        "Rain": float(rainfall),
        "Ferment": ferment_val,
        "Humidity": humidity_val,
        "Inlet": inlet_val,
        "Outlet": outlet_val
    }])

    pred_output = model.predict(X_user)[0]
    yield_pct = (pred_output / input_tea) * 100

    st.session_state.predicted_result = (pred_output, yield_pct)

# ===============================
# DISPLAY PREDICTION
# ===============================
if st.session_state.predicted_result is not None:
    pred_output, yield_pct = st.session_state.predicted_result

    colA, colB = st.columns(2)
    with colA:
        st.metric("Predicted Output (kg)", f"{pred_output:.2f}")
    with colB:
        st.metric("Predicted Yield (%)", f"{yield_pct:.2f}")

    traffic_light(yield_pct)

    if st.session_state.optimal_result is not None:
        loss = st.session_state.optimal_result["Yield_pct"] - yield_pct
        st.info(f"📉 Yield loss compared to optimal: **{loss:.2f}%**")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("🔬 Machine Learning based Decision Support Demo")
