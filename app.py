iimport streamlit as st
import numpy as np
import pickle

model = pickle.load(open("student_model.pkl", "rb"))

st.title("🎓 Student Result Prediction App")

study_hours = st.number_input("Study Hours per Day", 0.0, 24.0)
attendance = st.number_input("Attendance (%)", 0.0, 100.0)
previous_marks = st.number_input("Previous Marks (%)", 0.0, 100.0)

if st.button("Predict"):
    data = np.array([[study_hours, attendance, previous_marks]])
    result = model.predict(data)

    if result[0] == 1:
        st.success("✅ Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")

