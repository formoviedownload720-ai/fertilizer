import streamlit as st
import numpy as np

# ============================================================
# STAGE A: Sensor Calibration (Ordinary Least Squares)
# ============================================================
def calibrate_sensor(n_raw: float, p_raw: float, k_raw: float):
    n_cal = 0.8255 * n_raw + 0.05026
    p_cal = 0.07761 * p_raw + 14.55
    k_cal = 0.5840 * k_raw + 0.2508
    return n_cal, p_cal, k_cal


# ============================================================
# STAGE B: Fertilizer Recommendation (Closed-form of FRG-2018)
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

HIGH_N, HIGH_P, HIGH_K = 0.30, 40.0, 0.50


def soil_test_index(cal_n, cal_p, cal_k):
    x_n = np.clip((cal_n - 0.03) / (HIGH_N - 0.03), 0.0, 1.2)
    x_p = np.clip((cal_p - 2.0) / (HIGH_P - 2.0), 0.0, 1.2)
    x_k = np.clip((cal_k - 0.05) / (HIGH_K - 0.05), 0.0, 1.2)
    x = 0.45 * x_n + 0.30 * x_p + 0.25 * x_k
    return float(np.clip(x, 0.0, 1.3)), x_n, x_p, x_k


def recommend_doses(crop, x):
    dmax = CROP_MAX_DOSES[crop]
    urea = dmax["urea"] * max(0.0, 1.0 - x / 1.000)
    mop  = dmax["mop"]  * max(0.0, 1.0 - x / 0.833)
    tsp  = dmax["tsp"]  * max(0.0, 1.0 - x / 0.667)
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
st.caption("Learning version – every model and calculation is explained in simple language")

# ---------- Sidebar ----------
st.sidebar.header("📥 Raw Sensor Readings")
n_raw = st.sidebar.number_input("Nitrogen sensor reading (%)", 0.000, 0.200, 0.005, 0.001, format="%.3f")
p_raw = st.sidebar.number_input("Phosphorus sensor reading (mg kg⁻¹)", 0.0, 500.0, 120.0, 1.0, format="%.1f")
k_raw = st.sidebar.number_input("Potassium sensor reading (meq 100 g⁻¹)", 0.000, 2.000, 0.300, 0.001, format="%.3f")

st.sidebar.header("🌱 Crop & Land Area")
crop = st.sidebar.selectbox("Select crop / variety", list(CROP_MAX_DOSES.keys()), index=4)
land_area = st.sidebar.number_input("Land area (decimal)", 0.1, 1000.0, 1.0, 0.1)

calculate = st.sidebar.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)

# ============================================================
if calculate:

    # ========================================================
    # STAGE A
    # ========================================================
    st.header("① Stage A – Sensor Calibration")
    st.subheader("How the Regression Model works here")

    st.markdown("""
    ### What is Ordinary Least Squares (OLS) Regression?

    Imagine you have many pairs of numbers:
    - X-axis → what the cheap sensor reads
    - Y-axis → what the real laboratory (SRDI) measured

    You want a straight line that best fits all those points.

    The equation of a straight line is:

    $$
    y = a \\times x + b
    $$

    - \( a \) = slope (how much the real value changes when sensor changes)
    - \( b \) = intercept (the value when sensor reads zero)

    **Ordinary Least Squares** finds the best \( a \) and \( b \) by minimising the sum of squared errors  
    (the vertical distances from each point to the line).

    In this thesis, three separate lines were fitted:

    | Nutrient   | Slope \( a \) | Intercept \( b \) | Meaning |
    |------------|---------------|-------------------|---------|
    | Nitrogen   | 0.8255        | 0.05026           | Sensor slightly over-responds |
    | Phosphorus | 0.07761       | 14.55             | Almost no response (bad channel) |
    | Potassium  | 0.5840        | 0.2508            | Good, usable response |

    These numbers came from 16 paired observations after a controlled fertilizer experiment.
    """)

    n_cal, p_cal, k_cal = calibrate_sensor(n_raw, p_raw, k_raw)

    st.subheader("Live calculation with your sensor readings")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Nitrogen**")
        st.code(f"""N_sensor = {n_raw:.3f}
N_corrected = 0.8255 × {n_raw:.3f} + 0.05026
            = {n_cal:.4f} %""")
        st.metric("Calibrated N", f"{n_cal:.4f} %")

    with c2:
        st.markdown("**Phosphorus**")
        st.code(f"""P_sensor = {p_raw:.1f}
P_corrected = 0.07761 × {p_raw:.1f} + 14.55
            = {p_cal:.1f} mg/kg""")
        st.metric("Calibrated P", f"{p_cal:.1f} mg/kg")

    with c3:
        st.markdown("**Potassium**")
        st.code(f"""K_sensor = {k_raw:.3f}
K_corrected = 0.5840 × {k_raw:.3f} + 0.2508
            = {k_cal:.3f} meq/100g""")
        st.metric("Calibrated K", f"{k_cal:.3f} meq/100g")

    st.info("""
    **Thesis findings about the regression**  
    - Potassium: strong relationship (p < 0.001) → trustworthy  
    - Nitrogen: weak but usable (p = 0.029)  
    - Phosphorus: slope almost zero → the P channel is almost useless  
    This is why calibration error dominates the final dose error.
    """)

    st.divider()

    # ========================================================
    # STAGE B
    # ========================================================
    st.header("② Stage B – Fertilizer Dose Prediction")
    st.subheader("How Random Forest works here (and why we can replace it)")

    st.markdown("""
    ### What is Random Forest?

    Random Forest is an **ensemble** of many Decision Trees.

    1. Each tree is trained on a random sample of the data and a random subset of features.
    2. When predicting, every tree gives its own answer.
    3. The final prediction is the **average** of all the trees (for regression).

    In the thesis they trained four models:
    - Multiple Linear Regression
    - Single Decision Tree
    - **Random Forest** ← selected as best
    - Gradient Boosting

    Random Forest won because the FRG-2018 recommendation table has a very special shape:
    - The dose falls in a straight line as soil fertility increases
    - Then it is clipped at zero (never becomes negative)

    This creates an **axis-aligned, non-smooth** surface — exactly the kind of pattern  
    that tree-based models are excellent at learning.

    ### Important discovery in the thesis

    When they looked carefully at the training data they found:

    $$
    D = D_{\\max}(\\text{crop}) \\times \\max\\left(0,\\; 1 - \\frac{x}{x_0}\\right)
    $$

    This simple formula reproduces **all 999 tabulated doses** with R² ≈ 0.999999.

    That means the Random Forest was essentially **memorising this closed-form equation**.

    Therefore, in this learning app we implement the closed-form directly.  
    You get the same numbers the Random Forest would have produced,  
    but you can also see every intermediate step.
    """)

    x, x_n, x_p, x_k = soil_test_index(n_cal, p_cal, k_cal)

    st.subheader("Step 1 – Build the soil-test index \( x \)")

    st.markdown("""
    The original FRG table uses a normalised fertility index.  
    Higher \( x \) = better soil = less fertilizer needed.
    """)

    st.code(f"""
x_N = clip( ({n_cal:.4f} - 0.03) / 0.27 ) = {x_n:.4f}
x_P = clip( ({p_cal:.1f} - 2.0) / 38.0 )  = {x_p:.4f}
x_K = clip( ({k_cal:.3f} - 0.05) / 0.45 ) = {x_k:.4f}

x = 0.45·x_N + 0.30·x_P + 0.25·x_K = {x:.4f}
    """)

    st.metric("Soil-test index x", f"{x:.4f}")

    st.subheader("Step 2 – Apply the closed-form (what Random Forest learned)")

    dmax = CROP_MAX_DOSES[crop]
    urea_g, mop_g, tsp_g = recommend_doses(crop, x)

    st.markdown(f"**Crop selected:** `{crop}`")
    st.write(f"Maximum doses when soil is completely depleted (x = 0):")
    st.write(f"- Urea max = **{dmax['urea']} g/decimal**")
    st.write(f"- MoP  max = **{dmax['mop']} g/decimal**")
    st.write(f"- TSP  max = **{dmax['tsp']} g/decimal**")

    st.code(f"""
Urea = {dmax['urea']} × max(0, 1 - {x:.4f}/1.000) = {urea_g:.2f} g/decimal
MoP  = {dmax['mop']} × max(0, 1 - {x:.4f}/0.833) = {mop_g:.2f} g/decimal
TSP  = {dmax['tsp']} × max(0, 1 - {x:.4f}/0.667) = {tsp_g:.2f} g/decimal
    """)

    st.subheader("③ Final Recommended Doses")
    st.caption(f"For your land area of **{land_area} decimal**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Urea", f"{urea_g * land_area:.1f} g")
    col2.metric("MoP", f"{mop_g * land_area:.1f} g")
    col3.metric("TSP", f"{tsp_g * land_area:.1f} g")

    with st.expander("Show values per decimal"):
        st.write(f"- Urea: **{urea_g:.1f} g/decimal**")
        st.write(f"- MoP:  **{mop_g:.1f} g/decimal**")
        st.write(f"- TSP:  **{tsp_g:.1f} g/decimal**")

    st.warning("""
    **Key learning points from the thesis**
    - Random Forest looked very accurate under normal cross-validation  
      but failed when tested with leave-one-crop-out → it was mainly memorising crop maxima.
    - Calibration error (Stage A) is ~20 times larger than model error (Stage B) in money terms.
    - Phosphorus channel is the weakest link in the whole pipeline.
    """)

else:
    st.markdown("""
    ### How to use this learning app

    1. Enter any raw sensor readings on the left  
    2. Choose a rice variety  
    3. Click **Run Full Pipeline**

    You will see:
    - How **Ordinary Least Squares regression** calibrates the sensor (Stage A)
    - How **Random Forest** works and why the closed-form equation is equivalent (Stage B)
    - Every intermediate number and formula
    """)

st.divider()
st.caption(
    "Learning implementation of the RUET BSc Thesis  |  "
    "Md. Rezwanus Samam Toha (2002132) & Tasbir Alam (2002154)  |  "
    "Supervisor: Dr. Md. Rokunuzzaman"
)
