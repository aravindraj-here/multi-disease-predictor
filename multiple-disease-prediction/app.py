import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Multi-Disease Predictor", layout="centered")
st.title("🧠 Multi-Disease Prediction App")

disease = st.selectbox("Select Disease to Predict", ["Kidney Disease", "Liver Disease", "Parkinson's Disease"])

# KIDNEY DISEASE

if disease == "Kidney Disease":
    st.subheader("🩺 Kidney Disease Prediction")

    age = st.number_input("Age", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    specific_gravity = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
    albumin = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
    sugar = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
    rbc = st.selectbox("Red Blood Cells", [0, 1])
    pc = st.selectbox("Pus Cell", [0, 1])
    pcc = st.selectbox("Pus Cell Clumps", [0, 1])
    ba = st.selectbox("Bacteria", [0, 1])
    bgr = st.number_input("Blood Glucose Random", min_value=0.0)
    bu = st.number_input("Blood Urea", min_value=0.0)
    sc = st.number_input("Serum Creatinine", min_value=0.0)
    sod = st.number_input("Sodium", min_value=0.0)
    pot = st.number_input("Potassium", min_value=0.0)
    hemo = st.number_input("Hemoglobin", min_value=0.0)
    pcv = st.number_input("Packed Cell Volume", min_value=0.0)
    wc = st.number_input("WBC Count", min_value=0.0)
    rc = st.number_input("RBC Count", min_value=0.0)
    htn = st.selectbox("Hypertension", [0, 1])
    dm = st.selectbox("Diabetes Mellitus", [0, 1])
    cad = st.selectbox("Coronary Artery Disease", [0, 1])
    appet = st.selectbox("Appetite", [0, 1])
    pe = st.selectbox("Pedal Edema", [0, 1])
    ane = st.selectbox("Anemia", [0, 1])

    input_data = np.array([[
        age, blood_pressure, specific_gravity, albumin, sugar, rbc, pc, pcc, ba,
        bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane
    ]])

    if st.button("Predict"):
        model = joblib.load("models/kidney_model.pkl")
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.success("✅ The patient is likely healthy.")
        else:
            st.error("⚠️ The patient is likely to have kidney disease.")

# LIVER DISEASE

elif disease == "Liver Disease":
    st.subheader("🩸 Liver Disease Prediction")

    age = st.number_input("Age", min_value=1)
    gender = st.selectbox("Gender (Male=1, Female=0)", [1, 0])
    total_bilirubin = st.number_input("Total Bilirubin")
    direct_bilirubin = st.number_input("Direct Bilirubin")
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase")
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase")
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase")
    total_proteins = st.number_input("Total Proteins")
    albumin = st.number_input("Albumin")
    ag_ratio = st.number_input("Albumin and Globulin Ratio")

    input_data = np.array([[
        age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
        alamine_aminotransferase, aspartate_aminotransferase, total_proteins,
        albumin, ag_ratio
    ]])

    if st.button("Predict"):
        model = joblib.load("models/liver_model.pkl")
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("✅ The patient is likely healthy.")
        else:
            st.error("⚠️ The patient is likely to have liver disease.")

# PARKINSON’S DISEASE

elif disease == "Parkinson's Disease":
    st.subheader("🧠 Parkinson's Disease Prediction")

    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter_percent = st.number_input("MDVP:Jitter(%)")
    jitter_abs = st.number_input("MDVP:Jitter(Abs)")
    rap = st.number_input("MDVP:RAP")
    ppq = st.number_input("MDVP:PPQ")
    ddp = st.number_input("Jitter:DDP")
    shimmer = st.number_input("MDVP:Shimmer")
    shimmer_db = st.number_input("MDVP:Shimmer(dB)")
    apq3 = st.number_input("Shimmer:APQ3")
    apq5 = st.number_input("Shimmer:APQ5")
    apq = st.number_input("MDVP:APQ")
    dda = st.number_input("Shimmer:DDA")
    nhr = st.number_input("NHR")
    hnr = st.number_input("HNR")
    rpde = st.number_input("RPDE")
    dfa = st.number_input("DFA")
    spread1 = st.number_input("Spread1")
    spread2 = st.number_input("Spread2")
    d2 = st.number_input("D2")
    ppe = st.number_input("PPE")

    input_data = np.array([[
        fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
        shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
        rpde, dfa, spread1, spread2, d2, ppe
    ]])

    if st.button("Predict"):
        model = joblib.load("models/parkinsons_model.pkl")
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.success("✅ The patient is likely healthy.")
        else:
            st.error("⚠️ The patient is likely to have Parkinson's disease.")


