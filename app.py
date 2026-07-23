import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI-Based Student Burnout Detection System",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

# -------------------------------
# Title
# -------------------------------
st.title("🎓 AI-Based Student Burnout Detection System")

st.markdown("""
This system predicts the burnout risk of students using a **Random Forest Machine Learning Model**.
Fill in the student details and click **Predict Burnout Risk**.
""")

st.divider()

st.header("📝 Student Information")
# -------------------------------
# Input Form
# -------------------------------

col1, col2 = st.columns(2)

with col1:

    study_hours = st.slider(
        "Study Hours per Day",
        1,
        12,
        6
    )

    sleep_hours = st.slider(
        "Sleep Hours per Day",
        3,
        10,
        7
    )

    screen_time = st.slider(
        "Screen Time (Hours)",
        1,
        12,
        5
    )

    attendance = st.slider(
        "Attendance (%)",
        50,
        100,
        80
    )

    age = st.slider(
        "Age",
        18,
        30,
        20
    )

with col2:

    cgpa = st.slider(
        "CGPA",
        5.0,
        10.0,
        8.0
    )

    stress_level = st.selectbox(
        "Stress Level",
        ["Low", "Medium", "High"]
    )

    physical_activity = st.selectbox(
        "Physical Activity",
        ["Low", "Medium", "High"]
    )

    assignment_load = st.selectbox(
        "Assignment Load",
        ["Low", "Medium", "High"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

st.divider()

predict = st.button(
    "🔍 Predict Burnout Risk",
    use_container_width=True
)
# -------------------------------
# Prediction
# -------------------------------

if predict:

    # Encode categorical inputs
    stress = encoders["Stress_Level"].transform([stress_level])[0]
    activity = encoders["Physical_Activity"].transform([physical_activity])[0]
    assignment = encoders["Assignment_Load"].transform([assignment_load])[0]
    gender_encoded = encoders["Gender"].transform([gender])[0]

    input_data = pd.DataFrame([{
        "Study_Hours": study_hours,
        "Sleep_Hours": sleep_hours,
        "Screen_Time": screen_time,
        "Stress_Level": stress,
        "Physical_Activity": activity,
        "Assignment_Load": assignment,
        "Attendance": attendance,
        "CGPA": cgpa,
        "Age": age,
        "Gender": gender_encoded
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    burnout = encoders["Burnout"].inverse_transform([prediction])[0]
    confidence = np.max(probabilities) * 100

    st.divider()
    st.header("📊 Prediction Result")

    if burnout == "Low":
        st.success(f"✅ Burnout Risk : {burnout}")
    elif burnout == "Moderate":
        st.warning(f"⚠️ Burnout Risk : {burnout}")
    else:
        st.error(f"🚨 Burnout Risk : {burnout}")

    st.metric("Confidence Score", f"{confidence:.2f}%")

    st.subheader("💡 Personalized Recommendations")

    if burnout == "Low":

        st.success("""
✔ Maintain your current study routine.

✔ Continue regular physical activity.

✔ Sleep at least 7–8 hours daily.

✔ Maintain a healthy work-life balance.
""")

    elif burnout == "Moderate":

        st.warning("""
✔ Reduce excessive screen time.

✔ Take short breaks while studying.

✔ Improve your sleep schedule.

✔ Exercise regularly.

✔ Manage academic workload effectively.
""")

    else:

        st.error("""
✔ Consult your academic mentor or counsellor.

✔ Reduce study overload.

✔ Sleep at least 7–8 hours.

✔ Practice stress management techniques.

✔ Increase physical activity.

✔ Maintain a balanced lifestyle.
""")
