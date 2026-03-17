import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

print("Loading dataset...")

df = pd.read_csv("students_data.csv")

print("Dataset Loaded Successfully!")
print(df.head())


# -----------------------------
# Remove Student_ID
# -----------------------------
if "Student_ID" in df.columns:
    df = df.drop("Student_ID", axis=1)


# -----------------------------
# Convert categorical data
# -----------------------------
df = pd.get_dummies(df)


# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("Final_Exam_Score", axis=1)
y = df["Final_Exam_Score"]

print("\nFeatures used for training:")
print(X.columns)


# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

print("\nTraining the model...")

model.fit(X_train, y_train)


# -----------------------------
# Accuracy
# -----------------------------
accuracy = model.score(X_test, y_test)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# -----------------------------
# Create models folder
# -----------------------------
if not os.path.exists("models"):
    os.makedirs("models")


# -----------------------------
# Save Model
# -----------------------------
model_path = "models/student_model.pkl"
joblib.dump(model, model_path)


# -----------------------------
# Save feature columns
# -----------------------------
columns_path = "models/model_columns.pkl"
joblib.dump(X.columns, columns_path)


print("\nModel trained and saved successfully!")
print("Model file location:", model_path)
print("Feature columns saved:", columns_path)