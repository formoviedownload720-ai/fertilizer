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
    n_cal = 0.8255 * n_raw + 0.05026
    p_cal = 0.07761 * p_raw + 14.55
    k_cal = 0.5840 * k_raw + 0.2508
    return n_cal, p_cal, k_cal


# ============================================================
# STAGE B: Fertilizer Recommendation (Closed-form of FRG-2018)
# From Thesis Equation 3.7 and Table 3.6
# ============================================================

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

HIGH_N = 0.30
HIGH_P = 40.0
HIGH_K = 0.50


def soil_test_index(cal_n: float, cal_p: float, cal_k: float):
    """
    Returns the single soil-test index x and the three component indices
    so we can show the calculation to the user.
    """
    x_n = np.clip((cal_n - 0.03) / (HIGH_N - 0.03), 0.0, 1.2)
    x_p = np.clip((cal_p - 2.0) / (HIGH_P - 2.0), 0.0, 1.2)
    x_k = np.clip((cal_k - 0.05) / (HIGH_K - 0.05), 0.0, 1.2)

    x = 0.45 * x_n + 0.30 * x_p + 0.25 * x_k
    x = float(np.clip(x, 0.0, 1.3))
    return x, x_n, x_p, x_k


def recommend_doses(crop: str, x: float):
    """
    Closed-form recommendation (Equation 3.7 of the thesis).
    Returns urea, mop, tsp in g per decimal.
    """
    dmax = CROP_MAX_DOSES[crop]

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
    page_title="ML Fertilizer Recommendation | Learning Version",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 Two-Stage Fertilizer Recommendation System")
st.caption("Learning / Debugging version – every calculation is shown step-by-step")

st.markdown("""
This app implements the **exact two-stage pipeline** from the RUET thesis.  
It is written so you can clearly see **how Stage A works** and **how Stage B works**.
""")

# ---------- Sidebar Inputs ----------
st.sidebar.header("📥 Raw Sensor Readings")
st.sidebar.caption("Enter the values exactly as the 7-in-1 RS485 sensor reports them")

n_raw = st.sidebar.number_input("Nitrogen sensor reading (%)", 0.000, 0.200, 0.005, 0.001, format="%.3f")
p_raw = st.sidebar.number_input("Phosphorus sensor reading (mg kg⁻¹)", 0.0, 500.0, 120.0, 1.0, format="%.1f")
k_raw = st.sidebar.number_input("Potassium sensor reading (meq 100 g⁻¹)", 0.000, 2.000, 0.300, 0.001, format="%.3f")

st.sidebar.header("🌱 Crop & Land Area")
crop = st.sidebar.selectbox("Select crop / variety", list(CROP_MAX_DOSES.keys()), index=4)
land_area = st.sidebar.number_input("Land area (decimal)", 0.1, 1000.0, 1.0, 0.1)

calculate = st.sidebar.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)

# ============================================================
# MAIN CONTENT
# ============================================================
if calculate:

    # ---------- STAGE A ----------
    st.header("① Stage A – Sensor Calibration (Ordinary Least Squares)")

    st.markdown("""
    ### Why Stage A is needed
    The cheap 7-in-1 soil NPK sensor does **not** measure N, P and K directly.  
    It measures bulk electrical conductivity and converts that single signal into three numbers  
    using fixed factory factors that assume a “standard” soil.  

    For Barind Tract soils those factory factors are wrong.  
    Therefore we correct the raw readings with simple linear equations fitted against  
    real laboratory (SRDI) wet-chemistry values:

    $$
    \\begin{aligned}
    N_{\\text{corrected}} &= 0.8255 \\times N_{\\text{sensor}} + 0.05026 \\\\
    P_{\\text{corrected}} &= 0.07761 \\times P_{\\text{sensor}} + 14.55 \\\\
    K_{\\text{corrected}} &= 0.5840 \\times K_{\\text{sensor}} + 0.2508
    \\end{aligned}
    $$

    These three equations come from ordinary least-squares regression on 16 paired observations  
    (Thesis Table 4.2).
    """)

    # Actual calculation
    n_cal, p_cal, k_cal = calibrate_sensor(n_raw, p_raw, k_raw)

    st.subheader("Step-by-step calculation with your numbers")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Nitrogen**")
        st.code(f"""
N_sensor  = {n_raw:.3f} %
N_corrected = 0.8255 × {n_raw:.3f} + 0.05026
            = {0.8255 * n_raw:.5f} + 0.05026
            = {n_cal:.4f} %
        """)
        st.metric("Calibrated N", f"{n_cal:.4f} %")

    with col2:
        st.markdown("**Phosphorus**")
        st.code(f"""
P_sensor  = {p_raw:.1f} mg/kg
P_corrected = 0.07761 × {p_raw:.1f} + 14.55
            = {0.07761 * p_raw:.3f} + 14.55
            = {p_cal:.1f} mg/kg
        """)
        st.metric("Calibrated P", f"{p_cal:.1f} mg/kg")

    with col3:
        st.markdown("**Potassium**")
        st.code(f"""
K_sensor  = {k_raw:.3f} meq/100g
K_corrected = 0.5840 × {k_raw:.3f} + 0.2508
            = {0.5840 * k_raw:.5f} + 0.2508
            = {k_cal:.3f} meq/100g
        """)
        st.metric("Calibrated K", f"{k_cal:.3f} meq/100g")

    st.info("""
    **Note from the thesis**  
    - Potassium calibration is statistically strong (p < 0.001)  
    - Nitrogen is weak but usable (p = 0.029)  
    - Phosphorus slope is not significantly different from zero → the P channel is almost useless  
    That is why the final recommendation treats the three nutrients with different confidence.
    """)

    st.divider()

    # ---------- STAGE B ----------
    st.header("② Stage B – Fertilizer Dose Prediction")

    st.markdown("""
    ### How Stage B works
    The Fertilizer Recommendation Guide 2018 (FRG-2018) is a **deterministic lookup table**.  
    For any given soil-test level and crop it returns exactly one dose.  

    Mathematically the table can be written as a simple clipped linear function  
    (Thesis Equation 3.7):

    $$
    D = D_{\\max}(\\text{crop}, \\text{nutrient}) \\times \\max\\left(0,\\; 1 - \\frac{x}{x_0}\\right)
    $$

    where  
    - \( D_{\\max} \) = maximum recommended dose when soil fertility is zero  
    - \( x \) = normalised soil-test index (0 = very poor, ≈1 = high)  
    - \( x_0 \) = the point at which the dose becomes zero  
      - Urea: \( x_0 = 1.000 \)  
      - MoP:  \( x_0 = 0.833 \)  
      - TSP:  \( x_0 = 0.667 \)

    The Random Forest model in the thesis simply learned this exact relationship.  
    Here we implement the closed-form equation directly so you can see every number.
    """)

    # Calculate soil-test index
    x, x_n, x_p, x_k = soil_test_index(n_cal, p_cal, k_cal)

    st.subheader("Step 1 – Convert calibrated nutrients into a single soil-test index \( x \)")

    st.markdown("""
    Because the original training data had perfectly collinear N-K and P indices,  
    we create one combined index:
    """)

    st.code(f"""
x_N = clip( (N_cal - 0.03) / (0.30 - 0.03) , 0, 1.2 )
    = clip( ({n_cal:.4f} - 0.03) / 0.27 , 0, 1.2 )
    = {x_n:.4f}

x_P = clip( (P_cal - 2.0) / (40.0 - 2.0) , 0, 1.2 )
    = clip( ({p_cal:.1f} - 2.0) / 38.0 , 0, 1.2 )
    = {x_p:.4f}

x_K = clip( (K_cal - 0.05) / (0.50 - 0.05) , 0, 1.2 )
    = clip( ({k_cal:.3f} - 0.05) / 0.45 , 0, 1.2 )
    = {x_k:.4f}

Final index x = 0.45·x_N + 0.30·x_P + 0.25·x_K
              = 0.45×{x_n:.4f} + 0.30×{x_p:.4f} + 0.25×{x_k:.4f}
              = {x:.4f}
    """)

    st.metric("Soil-test index \( x \)", f"{x:.4f}",
              help="0 = very low fertility → maximum fertilizer; ≈1 = high fertility → almost no fertilizer")

    st.subheader("Step 2 – Apply the FRG closed-form equation for the selected crop")

    dmax = CROP_MAX_DOSES[crop]
    urea_g, mop_g, tsp_g = recommend_doses(crop, x)

    st.markdown(f"**Selected crop:** `{crop}`")
    st.markdown(f"**Maximum doses at x = 0 (from Thesis Table 3.6):**")
    st.write(f"- Urea max = **{dmax['urea']} g/decimal**")
    st.write(f"- MoP  max = **{dmax['mop']} g/decimal**")
    st.write(f"- TSP  max = **{dmax['tsp']} g/decimal**")

    st.code(f"""
Urea = {dmax['urea']} × max(0, 1 − {x:.4f}/1.000)
     = {dmax['urea']} × max(0, {1 - x/1.000:.4f})
     = {urea_g:.2f} g/decimal

MoP  = {dmax['mop']} × max(0, 1 − {x:.4f}/0.833)
     = {dmax['mop']} × max(0, {1 - x/0.833:.4f})
     = {mop_g:.2f} g/decimal

TSP  = {dmax['tsp']} × max(0, 1 − {x:.4f}/0.667)
     = {dmax['tsp']} × max(0, {1 - x/0.667:.4f})
     = {tsp_g:.2f} g/decimal
    """)

    # Final scaled results
    st.subheader("③ Final Recommended Doses")
    st.caption(f"Scaled to your land area of **{land_area} decimal**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Urea", f"{urea_g * land_area:.1f} g")
    c2.metric("MoP (Muriate of Potash)", f"{mop_g * land_area:.1f} g")
    c3.metric("TSP (Triple Superphosphate)", f"{tsp_g * land_area:.1f} g")

    with st.expander("Show values per decimal (as reported in the thesis)"):
        st.write(f"- Urea: **{urea_g:.1f} g/decimal**")
        st.write(f"- MoP:  **{mop_g:.1f} g/decimal**")
        st.write(f"- TSP:  **{tsp_g:.1f} g/decimal**")

    st.warning("""
    **Thesis limitations you should remember while coding / deploying**
    - Phosphorus channel almost does not respond → P dose has very high uncertainty  
    - System is only validated for the 9 rice varieties listed above  
    - Calibration was done on one High Barind Tract soil only (AEZ 26)  
    - Leave-one-crop-out validation showed the model mainly memorises crop-specific maxima
    """)

else:
    st.markdown("""
    ### How to use this learning version
    1. Enter any raw sensor readings in the left sidebar  
    2. Choose a crop  
    3. Click **Run Full Pipeline**

    The app will then show you:
    - The exact calibration equations of Stage A with your numbers plugged in  
    - How the soil-test index \( x \) is calculated  
    - How the final fertilizer doses are obtained from the FRG closed-form equation  

    This is deliberately written as a **transparent learning tool** so you can understand  
    every line of the two-stage pipeline while you build your own version.
    """)

# Footer
st.divider()
st.caption(
    "Learning implementation of the BSc Thesis: "
    "Machine Learning-Based Fertilizer Recommendation System Using Calibrated Soil NPK Sensor Data for Bangladesh  |  "
    "Md. Rezwanus Samam Toha (2002132) & Tasbir Alam (2002154)  |  "
    "Supervisor: Dr. Md. Rokunuzzaman  |  RUET"
)
