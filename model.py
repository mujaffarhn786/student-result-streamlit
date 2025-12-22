import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("student_data.csv")

X = data[['StudyHours', 'Attendance', 'PreviousMarks']]
y = data['Result']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("student_model.pkl", "wb"))

print("Model trained and saved successfully!")

