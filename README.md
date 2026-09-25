# Machine Learning-Based Fertilizer Recommendation System

Two-stage pipeline from the RUET BSc Thesis:

1. **Stage A – Sensor Calibration**  
   Ordinary Least Squares (OLS) equations fitted against SRDI wet-chemistry reference values.

2. **Stage B – Fertilizer Dose Prediction**  
   Closed-form implementation of the FRG-2018 mapping (the deterministic function that the Random Forest model learned).

## Live Demo

After you deploy, your Vercel / Streamlit Cloud URL will appear here.

## Local Run

```bash
git clone https://github.com/YOUR_USERNAME/fertilizer-recommendation-app.git
cd fertilizer-recommendation-app
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (Recommended)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Sign in with GitHub → “New app”.
4. Select this repo, branch `main`, main file path `app.py`.
5. Click Deploy.

(Streamlit Cloud is free and the easiest way to host a Streamlit app.)

## Deploy to Vercel (Alternative)

Vercel is optimized for frontend frameworks. Pure Streamlit needs a continuous server, so the recommended path is:

**Option A – Use Streamlit Cloud** (simplest).

**Option B – Convert to FastAPI + static frontend** (more work, fully serverless-friendly).

If you still want to try Streamlit on Vercel you can use a community adapter, but it is not officially supported and may have cold-start issues.

## Project Structure

```
fertilizer-recommendation-app/
├── app.py              # Main Streamlit application (Stage A + Stage B)
├── requirements.txt
└── README.md
```

## Calibration Equations Used (Stage A)

```
N_corrected = 0.8255 × N_sensor + 0.05026     [%]
P_corrected = 0.07761 × P_sensor + 14.55      [mg kg⁻¹]
K_corrected = 0.5840 × K_sensor + 0.2508      [meq 100 g⁻¹]
```

## Supported Crops (Stage B)

- Aus (BRRI 27)
- Aus (BRRI 42)
- B. Aman
- Boro (BRRI 28)
- Boro (BRRI 29)
- Boro (BRRI 36)
- T. Aman (BRRI 25)
- T. Aman (BRRI 51)
- T. Aman (Binadhan 9)

## Important Limitations (from the thesis)

- Potassium calibrates usefully, Nitrogen weakly, **Phosphorus does not calibrate** over the tested range.
- Valid only for the nine rice crop/variety combinations listed above.
- Calibration performed on a single High Barind Tract soil (AEZ 26).
- No field yield trial was conducted.

## Thesis Credits

**Title:** Machine Learning-Based Fertilizer Recommendation System Using Calibrated Soil NPK Sensor Data for Bangladesh  

**Authors:**  
Md. Rezwanus Samam Toha (Roll 2002132)  
Tasbir Alam (Roll 2002154)  

**Supervisor:** Dr. Md. Rokunuzzaman  
Professor, Department of Mechanical Engineering  
Rajshahi University of Engineering & Technology (RUET)
