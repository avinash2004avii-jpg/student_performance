print("Flask app starting...")

from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import sqlite3
import smtplib

app = Flask(__name__)

DATA_FILE = "students_data.csv"


# -------------------------
# Email Alert Function
# -------------------------
def send_email_alert():

    sender_email = "yourgmail@gmail.com"
    receiver_email = "teacher@email.com"
    app_password = "your_app_password"

    message = """Subject: ⚠ Student Risk Alert

Warning!

A student has been detected as AT RISK.

Please check the dashboard immediately.

AI Student Performance System
"""

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, app_password)
    server.sendmail(sender_email, receiver_email, message)
    server.quit()

    print("Email alert sent!")


# -------------------------
# Load ML Model
# -------------------------
model = joblib.load("models/student_model.pkl")


# -------------------------
# HOME PAGE
# -------------------------
@app.route("/")
def home():
    return render_template("home.html")


# -------------------------
# LOGIN PAGES
# -------------------------
@app.route("/teacher-login")
def teacher_login_page():
    return render_template("teacher_login.html")


@app.route("/admin-login")
def admin_login_page():
    return render_template("admin_login.html")


@app.route("/student-login")
def student_login_page():
    return render_template("student_login.html")


# -------------------------
# Prediction Form
# -------------------------
@app.route("/prediction-form")
def prediction_form():
    return render_template("prediction_form.html")


# -------------------------
# Login Authentication
# -------------------------
@app.route("/login", methods=["POST"])
def login():

    username = request.form["username"]
    password = request.form["password"]

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT role FROM users WHERE username=? AND password=?",
        (username, password)
    )

    user = cursor.fetchone()
    conn.close()

    if user:

        role = user[0]

        if role == "admin":
            return redirect(url_for("admin_dashboard"))

        elif role == "teacher":
            return redirect(url_for("teacher_dashboard"))

        elif role == "student":
            return redirect(url_for("student_dashboard"))

    return "Invalid Username or Password"


# -------------------------
# Admin Dashboard
# -------------------------
@app.route("/admin")
def admin_dashboard():

    df = pd.read_csv(DATA_FILE)

    df["risk"] = np.where(
        df["Final_Exam_Score"] < 50,
        "⚠ At Risk",
        "Safe"
    )

    total_students = len(df)
    risk_count = len(df[df["risk"] == "⚠ At Risk"])

    students = df.to_dict(orient="records")

    return render_template(
        "admin_dashboard.html",
        students=students,
        total_students=total_students,
        risk_count=risk_count
    )


# -------------------------
# Teacher Dashboard
# -------------------------
@app.route("/teacher")
def teacher_dashboard():

    df = pd.read_csv(DATA_FILE)

    df["risk"] = np.where(
        df["Final_Exam_Score"] < 50,
        "⚠ At Risk",
        "Safe"
    )

    students = df.to_dict(orient="records")

    risk_students = df[df["risk"] == "⚠ At Risk"]
    risk_students = risk_students.to_dict(orient="records")

    return render_template(
        "teacher_dashboard.html",
        students=students,
        risk_students=risk_students
    )


# -------------------------
# Student Dashboard
# -------------------------
@app.route("/student")
def student_dashboard():

    df = pd.read_csv(DATA_FILE)

    students = df.to_dict(orient="records")

    return render_template(
        "student_dashboard.html",
        students=students
    )


# -------------------------
# View All Students
# -------------------------
@app.route("/students")
def students_page():

    search_id = request.args.get("search_id")

    df = pd.read_csv(DATA_FILE)

    df["risk"] = np.where(
        df["Final_Exam_Score"] < 50,
        "⚠ At Risk",
        "Safe"
    )

    if search_id:
        df = df[df["Student_ID"].astype(str).str.contains(search_id)]

    students = df.to_dict(orient="records")

    return render_template("students.html", students=students)


# -------------------------
# Add Student Page
# -------------------------
@app.route("/add-student")
def add_student():
    return render_template("add_student.html")


# -------------------------
# Save Student
# -------------------------
@app.route("/save-student", methods=["POST"])
def save_student():

    df = pd.read_csv(DATA_FILE)

    new_row = {
        "Student_ID": request.form["student_id"],
        "Class": request.form["class"],
        "Section": request.form["section"],
        "Age": int(request.form["age"]),
        "Gender": request.form["gender"],
        "Study_hours": float(request.form["study_hours"]),
        "Sleep_hours": float(request.form["sleep_hours"]),
        "Parent_Education_Level": request.form["parent_education"],
        "Health_Issues": request.form["health"],
        "Internet_Access": request.form["internet"],
        "Attendence": float(request.form["attendance"]),
        "internal_test1": float(request.form["internal1"]),
        "internal_test2": float(request.form["internal2"]),
        "Assignment_score": float(request.form["assignment"]),
        "Extracurricular_Activities": request.form["extra"],
        "Previous_Exam_Score": float(request.form["previous"]),
        "Final_Exam_Score": float(request.form["final_score"])
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df = df.drop_duplicates(subset=["Student_ID"], keep="first")

    df.to_csv(DATA_FILE, index=False)

    return redirect(url_for("students_page"))


# -------------------------
# Upload Students
# -------------------------
@app.route("/upload-students", methods=["POST"])
def upload_students():

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    if file.filename.endswith(".csv"):
        df_new = pd.read_csv(file)

    elif file.filename.endswith(".xlsx"):
        df_new = pd.read_excel(file)

    else:
        return "Unsupported file format"

    df_existing = pd.read_csv(DATA_FILE)

    df_all = pd.concat([df_existing, df_new], ignore_index=True)

    df_all = df_all.drop_duplicates(subset=["Student_ID"], keep="first")

    df_all.to_csv(DATA_FILE, index=False)

    return redirect(url_for("students_page"))


# -------------------------
# Delete Student
# -------------------------
@app.route("/delete-student/<student_id>")
def delete_student(student_id):

    df = pd.read_csv(DATA_FILE)

    df = df[df["Student_ID"].astype(str) != student_id]

    df.to_csv(DATA_FILE, index=False)

    return redirect(url_for("students_page"))


# -------------------------
# Prediction (FIXED)
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():

    attendance = float(request.form["attendance"])
    internal1 = float(request.form["internal1"])
    internal2 = float(request.form["internal2"])
    previous_score = float(request.form["previous_score"])

    # Average internal marks
    internal_avg = (internal1 + internal2) / 2

    # Convert to model features
    study_hours = internal_avg / 10
    assignment_score = internal_avg
    sleep_hours = 7  # default sleep hours

    # Model expects 5 features
    data = [[study_hours, attendance, previous_score, assignment_score, sleep_hours]]

    prediction = model.predict(data)

    score = round(prediction[0], 2)

    if score < 50:
        result = "⚠ At Risk Student"
        send_email_alert()
    else:
        result = "Good Performance"

    return render_template("prediction.html", result=result, score=score)


# -------------------------
# Run Flask
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)