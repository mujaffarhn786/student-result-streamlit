import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load clean CSV
data = pd.read_csv("student_data.csv")

print("Columns:", data.columns)

X = data[['StudyHours', 'Attendance', 'PreviousMarks']]
y = data['Result']

model = LogisticRegression()
model.fit(X, y)

pickle.dump(model, open("student_model.pkl", "wb"))
print("Model saved successfully!")




