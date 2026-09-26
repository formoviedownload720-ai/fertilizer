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

    In this thesis, three separate lines were fitted from 16 paired observations:

    | Nutrient   | Slope \( a \) | Intercept \( b \) | p-value   | Meaning |
    |------------|---------------|-------------------|-----------|---------|
    | Nitrogen   | 0.8255        | 0.05026           | 0.029     | Weak but usable |
    | Phosphorus | 0.07761       | 14.55             | 0.458     | Almost no relationship |
    | Potassium  | 0.5840        | 0.2508            | **< 0.001** | Strong & trustworthy |
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

    # ---------- Detailed explanation of p-value ----------
    with st.expander("🔍 Why is Potassium (p < 0.001) called trustworthy? Click to understand", expanded=False):
        st.markdown("""
        ### What the p-value means

        When we fit the line \( y = a x + b \), we test this hypothesis:

        - **Null hypothesis (H₀)**: slope \( a = 0 \)  
          (the sensor has **no real relationship** with true laboratory potassium)
        - **Alternative**: slope \( a \neq 0 \)

        The **p-value** is the probability of getting a slope as large as 0.5840  
        **if the null hypothesis were true** (i.e., if there was really no relationship).

        | Nutrient   | Slope   | p-value   | Interpretation |
        |------------|---------|-----------|----------------|
        | Potassium  | 0.5840  | **< 0.001** | Less than 0.1% chance this happened by luck → **very strong evidence** |
        | Nitrogen   | 0.8255  | 0.029     | Significant but weak |
        | Phosphorus | 0.07761 | 0.458     | No evidence of real relationship |

        ### Why p < 0.001 is considered strong

        In science:
        - p < 0.05  → statistically significant
        - p < 0.01  → highly significant
        - **p < 0.001 → extremely strong evidence**

        Because the p-value for potassium is far below 0.001, we can be confident  
        that the sensor **does respond** to real changes in exchangeable potassium.

        Additional supporting evidence from the thesis:
        - The potassium channel tracked added MoP better than the other channels (R² ≈ 0.925)
        - The slope is reasonably large and its standard error is relatively small
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
    """)

    x, x_n, x_p, x_k = soil_test_index(n_cal, p_cal, k_cal)

    st.subheader("Step 1 – Build the soil-test index \( x \)")

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
    st.write("Maximum doses when soil is completely depleted (x = 0):")
    st.write(f"- Urea max = **{dmax['urea']} g/decimal**")
    st.write(f"- MoP  max = **{dmax['mop']} g/decimal**")
    st.write(f"- TSP  max = **{dmax['tsp']} g/decimal**")

    # ---------- Detailed explanation of D_max source ----------
    with st.expander("🔍 Where do these maximum dose values come from? Click to understand", expanded=False):
        st.markdown("""
        ### Source of the maximum doses

        These numbers come directly from **Table 3.6** of the thesis:

        > Composition of the recommendation dataset derived from the guide

        | Crop and variety       | Max Urea | Max MoP | Max TSP |
        |------------------------|----------|---------|---------|
        | Aus (BRRI 27)          | 773      | 486     | 324     |
        | Aus (BRRI 42)          | 773      | 486     | 324     |
        | B. Aman                | 422      | 324     | 243     |
        | Boro (BRRI 28)         | 1687     | 972     | 567     |
        | **Boro (BRRI 29)**     | **2109** | **1231**| **648** |
        | Boro (BRRI 36)         | 1406     | 810     | 486     |
        | T. Aman (BRRI 25)      | 843      | 648     | 324     |
        | T. Aman (BRRI 51)      | 1054     | 810     | 405     |
        | T. Aman (Binadhan 9)   | 633      | 486     | 243     |

        ### What do these numbers mean?

        They are the **maximum recommended fertilizer doses** when the soil is considered  
        completely depleted (soil-test index \( x = 0 \)).

        The original source is the official **Fertilizer Recommendation Guide 2018 (FRG-2018)**  
        published by the Bangladesh Agricultural Research Council (BARC).

        The thesis authors extracted these values from the FRG-2018 tables for the nine rice varieties.

        In the closed-form equation:

        $$
        D = D_{\\max} \\times \\max\\left(0,\\; 1 - \\frac{x}{x_0}\\right)
        $$

        - When \( x = 0 \), the dose becomes exactly \( D_{\\max} \)
        - That is why the app shows these values as “maximum doses”
        """)

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

    # ---------- Three important explanations ----------
    with st.expander("🔍 What is Leave-One-Crop-Out validation? Click to understand", expanded=False):
        st.markdown("""
        ### Normal cross-validation vs Leave-One-Crop-Out

        **Normal 5-fold cross-validation** (what most papers report):
        - Randomly split the 333 rows into 5 groups
        - Train on 4 groups, test on 1 group
        - Because the same crops appear in both train and test, the model can simply memorise  
          the maximum dose of each crop and look very accurate (R² ≈ 0.99)

        **Leave-One-Crop-Out (LOCO)**:
        - Completely remove **all 37 rows** of one crop (e.g. all Boro BRRI 29 data)
        - Train the model only on the other 8 crops
        - Then test on the held-out crop

        Result from the thesis (Random Forest):

        | Validation method       | Urea R² | Urea MAE (g/decimal) |
        |-------------------------|---------|----------------------|
        | Normal 5-fold           | 0.996   | 26.25                |
        | Leave-One-Crop-Out      | 0.252   | **210.05**           |

        The error jumped **8 times**.  

        This proves the model was mostly **memorising the nine crop-specific maximum doses**  
        rather than learning a real general relationship between soil nutrients and fertilizer need.

        That is why the thesis says the system is valid **only for the nine rice varieties**  
        that were in the training table.
        """)

    with st.expander("🔍 Why is Calibration error ~20× larger than Model error (in money)? Click to understand", expanded=False):
        st.markdown("""
        ### The numbers from the thesis (Section 4.4)

        For Boro rice (BRRI 29) the full recommended fertilizer costs about **Tk 24,478 per hectare**.

        When they propagated the errors through the whole pipeline they found:

        | Error source              | Money loss (Tk / hectare) | Share of fertilizer bill |
        |---------------------------|---------------------------|--------------------------|
        | **Calibration error** (Stage A) | **1,430 – 4,476**     | 5.8% – 18.3%             |
        | Model error (Stage B)     | only **64**               | 0.3%                     |

        → Calibration error is roughly **20–70 times larger** than the model error.

        ### Why this happens

        1. The Random Forest (or the closed-form equation) is almost perfect at reproducing  
           the FRG-2018 table → very small model error.
        2. But the input to that model is the **calibrated sensor reading**.
        3. The sensor (especially phosphorus and nitrogen) has large calibration residual.
        4. A small error in the soil-test index \( x \) is multiplied by the large maximum doses  
           (2109 g urea, 1231 g MoP, etc.) and becomes a big error in fertilizer amount.
        5. That fertilizer error, multiplied by the price of urea/TSP/MoP, becomes a large money error.

        **Conclusion from the thesis**:  
        Improving the recommendation model (Stage B) gives almost no benefit.  
        The real bottleneck is the quality of the sensor calibration (Stage A).
        """)

    with st.expander("🔍 If N and P are not accurate, what is the significance of this sensor? Click to understand", expanded=False):
        st.markdown("""
        ### Honest answer from the thesis

        The sensor is **not equally good for all three nutrients**:

        | Nutrient   | Calibration quality      | Practical usefulness                          |
        |------------|--------------------------|-----------------------------------------------|
        | Potassium  | Strong (p < 0.001)       | **Quantitatively usable**                     |
        | Nitrogen   | Weak (p = 0.029)         | Only rough indication                         |
        | Phosphorus | Almost none (p = 0.458)  | **Not usable for quantitative dose**          |

        ### So what is the real significance of the sensor?

        1. **It is still useful for Potassium**  
           The K channel is the only one that calibrates reliably.  
           For potassium dose the system is quantitatively supported.

        2. **It can give a fertility class, not a precise dose, for N and P**  
           Even if the exact number is wrong, the sensor can still tell you  
           “this soil is low / medium / high” in a rough sense.  
           That is better than applying a completely blind blanket dose.

        3. **Cost and coverage advantage**  
           - Laboratory test costs Tk 440 (economic cost) and takes days  
           - This sensor costs almost nothing per reading and gives instant result  
           - National laboratory capacity reaches only a tiny fraction of 16.88 million farm households  
           - A shared sensor (Tk 283 per farmer in a group) can reach far more people

        4. **It forces honesty about uncertainty**  
           The thesis contribution is not “the sensor is perfect”.  
           The contribution is: we measured the uncertainty of each nutrient separately  
           and declared that only potassium is currently trustworthy.

        ### Bottom line

        The sensor is **serviceable as shared measurement infrastructure for one nutrient (K)**,  
        informative for a second (N), and not yet trustworthy for the third (P).  

        That is still better than no measurement at all for the majority of Bangladeshi farmers  
        who currently receive no soil-test-based recommendation.
        """)

else:
    st.markdown("""
    ### How to use this learning app

    1. Enter any raw sensor readings on the left  
    2. Choose a rice variety  
    3. Click **Run Full Pipeline**

    You will see explanations for:
    - How **Ordinary Least Squares regression** calibrates the sensor (Stage A)
    - Why **Potassium p < 0.001** is considered trustworthy
    - How **Random Forest** works and why the closed-form is equivalent (Stage B)
    - Where the maximum dose values (2109, 1231, 648…) come from
    - What **Leave-One-Crop-Out** validation means and why it is important
    - Why **Calibration error is ~20× larger** than model error in money terms
    - If N and P are weak, **what is still the significance of this sensor**
    """)

st.divider()
st.caption(
    "Learning implementation of the RUET BSc Thesis  |  "
    "Md. Rezwanus Samam Toha (2002132) & Tasbir Alam (2002154)  |  "
    "Supervisor: Dr. Md. Rokunuzzaman"
)
