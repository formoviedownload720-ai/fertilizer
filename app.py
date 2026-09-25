import streamlit as st
import numpy as np

# ============================================================
# STAGE A: Sensor Calibration (Ordinary Least Squares)
# Coefficients from Thesis Table 4.2 / Equations 4.3-4.5
# ============================================================
def calibrate_sensor(n_raw: float, p_raw: float, k_raw: float):
    """
    Apply the fitted univariate OLS calibration equations.
    Returns calibrated N (%), P (mg/kg), K (meq/100g)
    """
    # N_corrected = 0.8255 * N_sensor + 0.05026
    n_cal = 0.8255 * n_raw + 0.05026

    # P_corrected = 0.07761 * P_sensor + 14.55
    p_cal = 0.07761 * p_raw + 14.55

    # K_corrected = 0.5840 * K_sensor + 0.2508
    k_cal = 0.5840 * k_raw + 0.2508

    return n_cal, p_cal, k_cal


# ============================================================
# STAGE B: Fertilizer Recommendation (Closed-form of FRG-2018)
# From Thesis Equation 3.7 and Table 3.6
# D = D_max * max(0, 1 - x / x0)
# x0 = 1.000 (urea), 0.833 (MoP), 0.667 (TSP)
# ============================================================

# Crop-specific maximum doses at soil-test index = 0 (g per decimal)
# Taken directly from Thesis Table 3.6
CROP_MAX_DOSES = {
    "Aus (BRRI 27)":          {"urea": 773,  "mop": 486, "tsp": 324},
    "Aus (BRRI 42)":          {"urea": 773,  "mop": 486, "tsp": 324},
    "B. Aman":                {"urea": 422,  "mop": 324, "tsp": 243},
    "Boro (BRRI 28)":         {"urea": 1687, "mop": 972, "tsp": 567},
    "Boro (BRRI 29)":         {"urea": 2109, "mop": 1231,"tsp": 648},
    "Boro (BRRI 36)":         {"urea": 1406, "mop": 810, "tsp": 486},
    "T. Aman (BRRI 25)":      {"urea": 843,  "mop": 648, "tsp": 324},
    "T. Aman (BRRI 51)":      {"urea": 1054, "mop": 810, "tsp": 405},
    "T. Aman (Binadhan 9)":   {"urea": 633,  "mop": 486, "tsp": 243},
}

# Critical limits (approximate from FRG / thesis context)
# Used to map calibrated values → soil test index x ∈ [0, 1+]
CRIT_N = 0.12   # %
CRIT_P = 8.0    # mg kg⁻¹
CRIT_K = 0.12   # meq 100 g⁻¹

# Upper bounds for normalization (roughly “very high” class)
HIGH_N = 0.30
HIGH_P = 40.0
HIGH_K = 0.50


def soil_test_index(cal_n: float, cal_p: float, cal_k: float) -> float:
    """
    Map calibrated nutrient values to a single soil-test index x ∈ [0, ≈1.2].
    Thesis notes the two continuous features are collinear in the training data,
    so a single index is used (consistent with the deterministic FRG mapping).
    Higher index → higher soil fertility → lower fertilizer dose.
    """
    # Normalize each nutrient to [0, 1] relative to critical → high range
    x_n = np.clip((cal_n - 0.03) / (HIGH_N - 0.03), 0.0, 1.2)
    x_p = np.clip((cal_p - 2.0) / (HIGH_P - 2.0), 0.0, 1.2)
    x_k = np.clip((cal_k - 0.05) / (HIGH_K - 0.05), 0.0, 1.2)

    # Weighted average (N and P/K carry most weight in FRG rice tables)
    x = 0.45 * x_n + 0.30 * x_p + 0.25 * x_k
    return float(np.clip(x, 0.0, 1.3))


def recommend_doses(crop: str, x: float):
    """
    Closed-form recommendation (Equation 3.7 of the thesis).
    Returns urea, mop, tsp in g per decimal.
    """
    dmax = CROP_MAX_DOSES[crop]

    # x0 values from thesis
    x0_urea = 1.000
    x0_mop  = 0.833
    x0_tsp  = 0.667

    urea = dmax["urea"] * max(0.0, 1.0 - x / x0_urea)
    mop  = dmax["mop"]  * max(0.0, 1.0 - x / x0_mop)
    tsp  = dmax["tsp"]  * max(0.0, 1.0 - x / x0_tsp)

    return urea, mop, tsp


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="ML Fertilizer Recommendation | RUET Thesis",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🌾 Machine Learning-Based Fertilizer Recommendation")
st.markdown("""
**Two-stage pipeline** (from the RUET BSc Thesis):  
1. **Stage A – Sensor Calibration** (Ordinary Least Squares)  
2. **Stage B – Dose Prediction** (FRG-2018 closed-form / Random-Forest surrogate)
""")

st.divider()

# ---------- Sidebar: Inputs ----------
st.sidebar.header("📥 Sensor Input (Raw Readings)")
st.sidebar.caption("Enter values exactly as reported by the 7-in-1 RS485/Modbus NPK sensor")

n_raw = st.sidebar.number_input(
    "Nitrogen sensor reading  (%)",
    min_value=0.000, max_value=0.200, value=0.005, step=0.001, format="%.3f"
)
p_raw = st.sidebar.number_input(
    "Phosphorus sensor reading  (mg kg⁻¹)",
    min_value=0.0, max_value=500.0, value=120.0, step=1.0, format="%.1f"
)
k_raw = st.sidebar.number_input(
    "Potassium sensor reading  (meq 100 g⁻¹)",
    min_value=0.000, max_value=2.000, value=0.300, step=0.001, format="%.3f"
)

st.sidebar.header("🌱 Crop & Area")
crop = st.sidebar.selectbox(
    "Select crop / variety",
    options=list(CROP_MAX_DOSES.keys()),
    index=4  # default Boro (BRRI 29)
)

land_area = st.sidebar.number_input(
    "Land area (decimal)",
    min_value=0.1, max_value=1000.0, value=1.0, step=0.1
)

# ---------- Main computation ----------
if st.sidebar.button("🚀 Calculate Recommendation", type="primary", use_container_width=True):

    # Stage A
    n_cal, p_cal, k_cal = calibrate_sensor(n_raw, p_raw, k_raw)

    # Stage B
    x = soil_test_index(n_cal, p_cal, k_cal)
    urea_g, mop_g, tsp_g = recommend_doses(crop, x)

    # Scale to land area
    urea_total = urea_g * land_area
    mop_total  = mop_g  * land_area
    tsp_total  = tsp_g  * land_area

    # ---------- Display results ----------
    st.subheader("① Stage A – Calibrated Sensor Values")
    col1, col2, col3 = st.columns(3)
    col1.metric("Calibrated N (%)", f"{n_cal:.4f}")
    col2.metric("Calibrated P (mg kg⁻¹)", f"{p_cal:.1f}")
    col3.metric("Calibrated K (meq 100 g⁻¹)", f"{k_cal:.3f}")

    st.info(
        f"**Soil-test index x = {x:.3f}**  \n"
        "(0 = very low fertility → maximum dose; ≈1 = high fertility → near-zero dose)"
    )

    st.subheader("② Stage B – Recommended Fertilizer Doses")
    st.caption(f"Crop: **{crop}**  |  Area: **{land_area} decimal**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Urea", f"{urea_total:.1f} g", help="Total for the entered land area")
    c2.metric("MoP (Muriate of Potash)", f"{mop_total:.1f} g")
    c3.metric("TSP (Triple Superphosphate)", f"{tsp_total:.1f} g")

    # Also show per-decimal for reference
    with st.expander("Show per-decimal values (as in thesis)"):
        st.write(f"- Urea: **{urea_g:.1f} g/decimal**")
        st.write(f"- MoP: **{mop_g:.1f} g/decimal**")
        st.write(f"- TSP: **{tsp_g:.1f} g/decimal**")

    # Important disclaimer from thesis conclusions
    st.warning("""
    **Important limitations (from the thesis)**  
    - Potassium channel calibrates usefully; Nitrogen weakly; **Phosphorus does not calibrate** over the tested range.  
    - The system is validated only for the nine rice crop/variety combinations listed.  
    - Calibration was performed on a single High Barind Tract soil (AEZ 26).  
    - Treat the three doses with different confidence – especially phosphorus.
    """)

else:
    st.markdown("""
    ### How to use
    1. Enter the **raw** readings from your 7-in-1 soil NPK sensor in the sidebar.  
    2. Choose the rice crop / variety.  
    3. Optionally adjust the land area (in decimal).  
    4. Click **Calculate Recommendation**.

    The app first applies the **OLS calibration equations** derived in the thesis,  
    then computes fertilizer doses using the **closed-form FRG-2018 mapping**  
    (the same relationship that the Random Forest model learned).
    """)

# Footer
st.divider()
st.caption(
    "Based on the BSc Thesis: *Machine Learning-Based Fertilizer Recommendation System "
    "Using Calibrated Soil NPK Sensor Data for Bangladesh*  \n"
    "Md. Rezwanus Samam Toha (2002132) & Tasbir Alam (2002154)  |  "
    "Supervised by Dr. Md. Rokunuzzaman  |  RUET, Rajshahi"
)
