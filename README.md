# Two-Stage Fertilizer Recommendation System (Learning Version)

This is a **transparent learning implementation** of the RUET BSc thesis pipeline.

It is deliberately written so you can see **exactly how Stage A and Stage B work**, including every formula and intermediate calculation.

## What the app shows you

### Stage A – Sensor Calibration (Ordinary Least Squares)
- Why the cheap NPK sensor needs calibration
- The three linear equations fitted in the thesis
- Live calculation with the numbers you enter

### Stage B – Fertilizer Dose Prediction
- How the FRG-2018 recommendation table is turned into a simple mathematical formula
- How the soil-test index x is calculated from calibrated N, P, K
- How the final Urea / MoP / TSP doses are obtained

## Files

```
app.py              ← Main Streamlit application (with full explanations)
requirements.txt
README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Recommended: Streamlit Community Cloud (share.streamlit.io)

## Thesis Credits

**Title:** Machine Learning-Based Fertilizer Recommendation System Using Calibrated Soil NPK Sensor Data for Bangladesh  

**Authors:**  
Md. Rezwanus Samam Toha (2002132)  
Tasbir Alam (2002154)  

**Supervisor:** Dr. Md. Rokunuzzaman  
Department of Mechanical Engineering, RUET
