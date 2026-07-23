import pandas as pd
import numpy as np

np.random.seed(42)

num_samples = 1000

study_hours = np.random.randint(1, 11, num_samples)
sleep_hours = np.random.randint(3, 10, num_samples)
screen_time = np.random.randint(1, 11, num_samples)

stress_level = np.random.choice(["Low", "Medium", "High"], num_samples)
physical_activity = np.random.choice(["Low", "Medium", "High"], num_samples)
assignment_load = np.random.choice(["Low", "Medium", "High"], num_samples)

attendance = np.random.randint(50, 101, num_samples)
cgpa = np.round(np.random.uniform(5.0, 10.0, num_samples), 2)

age = np.random.randint(18, 26, num_samples)
gender = np.random.choice(["Male", "Female"], num_samples)

burnout = []

for i in range(num_samples):
    score = 0

    if study_hours[i] > 8:
        score += 2
    elif study_hours[i] > 6:
        score += 1

    if sleep_hours[i] < 5:
        score += 2
    elif sleep_hours[i] < 7:
        score += 1

    if screen_time[i] > 8:
        score += 2
    elif screen_time[i] > 6:
        score += 1

    if stress_level[i] == "High":
        score += 2
    elif stress_level[i] == "Medium":
        score += 1

    if physical_activity[i] == "Low":
        score += 2
    elif physical_activity[i] == "Medium":
        score += 1

    if assignment_load[i] == "High":
        score += 2
    elif assignment_load[i] == "Medium":
        score += 1

    if attendance[i] < 70:
        score += 1

    if cgpa[i] < 6:
        score += 1

    if score >= 9:
        burnout.append("High")
    elif score >= 5:
        burnout.append("Moderate")
    else:
        burnout.append("Low")

df = pd.DataFrame({
    "Study_Hours": study_hours,
    "Sleep_Hours": sleep_hours,
    "Screen_Time": screen_time,
    "Stress_Level": stress_level,
    "Physical_Activity": physical_activity,
    "Assignment_Load": assignment_load,
    "Attendance": attendance,
    "CGPA": cgpa,
    "Age": age,
    "Gender": gender,
    "Burnout": burnout
})

df.to_csv("burnout_dataset.csv", index=False)

print("Dataset generated successfully!")
print(df.head())