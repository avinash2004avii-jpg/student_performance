"""
app.py  —  Student Performance Prediction System
"""

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, send_file)
import pandas as pd
import numpy as np
import joblib, os, io
import database as db

app = Flask(__name__)
app.secret_key = "sps_secret_key_change_in_production"

# ── Paths ──────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE, "data",   "students_data.csv")
BULK_OUT  = os.path.join(BASE, "data",   "bulk_results.csv")
MDL_DIR   = os.path.join(BASE, "models")

# ── Load model ────────────────────────────────────────────────────
def load_model():
    p = os.path.join(MDL_DIR, "student_model.pkl")
    if not os.path.exists(p):
        return None, None, None
    return (joblib.load(os.path.join(MDL_DIR, "student_model.pkl")),
            joblib.load(os.path.join(MDL_DIR, "model_columns.pkl")),
            joblib.load(os.path.join(MDL_DIR, "le_health.pkl")))

model, model_columns, le_health = load_model()

db.create_tables()

# ── Helpers ───────────────────────────────────────────────────────
def load_csv():
    df = pd.read_csv(DATA_FILE)
    df["Health_Issues"] = df["Health_Issues"].fillna("None")
    return df

def fval(form, key, default=0):
    """Safe float from form — handles empty strings gracefully."""
    v = form.get(key, "").strip()
    try:
        return float(v) if v != "" else float(default)
    except (ValueError, TypeError):
        return float(default)

def ival(form, key, default=0):
    """Safe int from form — handles empty strings gracefully."""
    v = form.get(key, "").strip()
    try:
        return int(v) if v != "" else int(default)
    except (ValueError, TypeError):
        return int(default)

def risk_label(score):
    if score is None: return "Unknown"
    if score < 70:    return "At Risk"
    if score < 80:    return "Average"
    return "Safe"

def safe_encode_health(val):
    v = str(val) if val else "None"
    if v not in le_health.classes_: v = "None"
    return int(le_health.transform([v])[0])

def build_features(row):
    it1  = float(row.get("internal_test 1",  row.get("internal1",  0)))
    it2  = float(row.get("internal_test 2",  row.get("internal2",  0)))
    asgn = float(row.get("Assignment_score", row.get("assignment", 0)))
    prev = float(row.get("Previous_Exam_Score", row.get("previous_score", 0)))
    att  = float(row.get("Attendence",  row.get("attendance",  75)))
    sh   = float(row.get("Study_hours", row.get("study_hours", 3)))
    slp  = float(row.get("Sleep_hours", row.get("sleep_hours", 7)))
    hlth = row.get("Health_Issues", row.get("health", "None"))
    feats = {
        "Study_hours": sh, "Health_Issues": safe_encode_health(hlth),
        "Attendence": att, "internal_test 1": it1, "internal_test 2": it2,
        "Assignment_score": asgn, "Previous_Exam_Score": prev,
        "internal_avg": (it1+it2)/2, "internal_diff": abs(it1-it2),
        "academic_score": (it1+it2+asgn)/3,
        "study_x_attendance": sh*att/100,
        "total_score": it1+it2+asgn+prev,
        "study_efficiency": sh/(slp+1),
        "high_study": 1 if sh>4 else 0,
    }
    return pd.DataFrame([feats])[model_columns]

def predict_score(row):
    if model is None: return None
    return round(float(model.predict(build_features(row))[0]), 1)

def generate_suggestions(score, row):
    """Return list of personalised improvement tips based on student data."""
    tips = []
    att  = float(row.get("Attendence", row.get("attendance", 100)))
    sh   = float(row.get("Study_hours", row.get("study_hours", 3)))
    it1  = float(row.get("internal_test 1", row.get("internal1", 0)))
    it2  = float(row.get("internal_test 2", row.get("internal2", 0)))
    asgn = float(row.get("Assignment_score", row.get("assignment", 0)))
    slp  = float(row.get("Sleep_hours", row.get("sleep_hours", 7)))

    if att < 75:
        tips.append(("📅 Attendance", f"Attendance is {att}% — below the 75% minimum. "
                     "Missing class directly correlates with lower scores. "
                     "Aim for at least 85% attendance."))
    if sh < 2:
        tips.append(("📚 Study Time", f"Only {sh} hours of study per day is very low. "
                     "Increasing to at least 3–4 hours daily can significantly improve performance."))
    elif sh < 3:
        tips.append(("📚 Study Time", f"{sh} hours of study is below average. "
                     "Try adding one focused study session per day."))
    if it2 < it1:
        tips.append(("📉 Declining Trend", f"Internal Test 2 ({it2}) is lower than Internal Test 1 ({it1}). "
                     "Performance is declining — review test 2 topics thoroughly and seek teacher help."))
    if asgn < 50:
        tips.append(("📝 Assignments", f"Assignment score is only {asgn}/100. "
                     "Completing assignments consistently is one of the easiest ways to improve your grade."))
    if slp < 6:
        tips.append(("😴 Sleep", f"Only {slp} hours of sleep affects memory and focus. "
                     "7–8 hours of sleep helps retain what you study."))
    if it1 < 50 and it2 < 50:
        tips.append(("📖 Core Concepts", "Both internal tests are below 50. "
                     "Focus on understanding fundamentals rather than memorising — consider extra tutoring."))
    if score < 70 and not tips:
        tips.append(("💡 General", "Review your weakest subjects first. "
                     "Create a weekly study schedule and stick to it."))
    return tips

def login_required(role=None):
    """Decorator factory for route protection."""
    from functools import wraps
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in to continue.", "warning")
                return redirect(url_for("login_page"))
            if role and session.get("role") != role:
                flash("Access denied.", "danger")
                return redirect(url_for("login_page"))
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════════════
@app.route("/")
def home():
    if "user_id" in session:
        role = session.get("role")
        if role == "admin":   return redirect(url_for("admin_dashboard"))
        if role == "teacher": return redirect(url_for("teacher_dashboard"))
        if role == "student": return redirect(url_for("student_dashboard"))
    return render_template("home.html")


# ════════════════════════════════════════════════════════════════════
# AUTH — Login / Signup / Logout
# ════════════════════════════════════════════════════════════════════
@app.route("/login", methods=["GET", "POST"])
def login_page():
    """Generic login — redirects to role-specific page."""
    return render_template("login.html")

@app.route("/login/admin", methods=["GET", "POST"])
def login_admin():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = db.login_user(username, password)
        if user and user["role"] == "admin":
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            return redirect(url_for("admin_dashboard"))
        flash("Invalid credentials or not an admin account.", "danger")
    return render_template("login_admin.html")

@app.route("/login/teacher", methods=["GET", "POST"])
def login_teacher():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = db.login_user(username, password)
        if user and user["role"] == "teacher":
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            return redirect(url_for("teacher_dashboard"))
        flash("Invalid credentials or not a teacher account.", "danger")
    return render_template("login_teacher.html")

@app.route("/login/student", methods=["GET", "POST"])
def login_student():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = db.login_user(username, password)
        if user and user["role"] == "student":
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            return redirect(url_for("student_dashboard"))
        flash("Invalid credentials or not a student account.", "danger")
    return render_template("login_student.html")


@app.route("/signup/teacher", methods=["GET", "POST"])
def signup_teacher():
    if request.method == "POST":
        username = request.form["username"].strip()
        email    = request.form["email"].strip()
        password = request.form["password"]
        confirm  = request.form["confirm"]
        name     = request.form["name"].strip()
        subject  = request.form.get("subject", "General").strip()

        if password != confirm:
            flash("Passwords do not match.", "danger")
        elif db.username_exists(username):
            flash("Username already taken.", "danger")
        elif db.email_exists(email):
            flash("Email already registered.", "danger")
        else:
            ok, msg = db.signup_teacher(username, email, password, name, subject)
            if ok:
                flash("Account created! Please log in.", "success")
                return redirect(url_for("login_teacher"))
            flash(msg, "danger")
    return render_template("signup_teacher.html")


@app.route("/signup/student", methods=["GET", "POST"])
def signup_student():
    teachers = db.get_all_teachers_simple()
    if request.method == "POST":
        username     = request.form["username"].strip()
        email        = request.form["email"].strip()
        password     = request.form["password"]
        confirm      = request.form["confirm"]
        name         = request.form["name"].strip()
        student_code = request.form["student_code"].strip()
        class_       = request.form.get("class_", "")
        section      = request.form.get("section", "")
        teacher_id   = request.form.get("teacher_id") or None

        if password != confirm:
            flash("Passwords do not match.", "danger")
        elif db.username_exists(username):
            flash("Username already taken.", "danger")
        elif db.email_exists(email):
            flash("Email already registered.", "danger")
        else:
            ok, msg = db.signup_student(username, email, password, name,
                                        student_code, class_, section, teacher_id)
            if ok:
                flash("Account created! Please log in.", "success")
                return redirect(url_for("login_student"))
            # Give a clear message for the most common failure
            if "student_code" in msg or "UNIQUE" in msg:
                flash("That Student ID is already registered. Check your roll number.", "danger")
            else:
                flash(msg, "danger")
    return render_template("signup_student.html", teachers=teachers)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ════════════════════════════════════════════════════════════════════
# ADMIN
# ════════════════════════════════════════════════════════════════════
@app.route("/admin")
@login_required("admin")
def admin_dashboard():
    df = load_csv()
    df["risk"] = np.where(df["Final_Exam_Score"] < 70, "At Risk", "Safe")
    return render_template("admin_dashboard.html",
        total_students=len(df),
        risk_count=len(df[df["risk"] == "At Risk"]),
        total_teachers=len(db.get_all_teachers()),
        users=db.get_all_users(),
        teachers=db.get_all_teachers(),
    )

@app.route("/admin/delete-user/<int:uid>")
@login_required("admin")
def admin_delete_user(uid):
    if uid == session["user_id"]:
        flash("You cannot delete your own account.", "danger")
    else:
        db.delete_user(uid)
        flash("User deleted.", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/students")
@login_required("admin")
def admin_students():
    q = request.args.get("q", "")
    df = load_csv()
    df["risk"] = np.where(df["Final_Exam_Score"] < 70, "At Risk", "Safe")
    if q:
        df = df[df["Student_ID"].astype(str).str.contains(q, na=False)]
    return render_template("students_table.html",
                           students=df.to_dict(orient="records"), q=q)

@app.route("/admin/add-student", methods=["GET", "POST"])
@login_required("admin")
def admin_add_student():
    if request.method == "POST":
        df = load_csv()
        new = {
            "Student_ID": request.form["student_id"],
            "Class": request.form["class_"], "section": request.form["section"],
            "Age": int(request.form.get("age", 14)),
            "Gender": request.form.get("gender", "Male"),
            "Study_hours": float(request.form.get("study_hours", 3)),
            "Sleep_hours": float(request.form.get("sleep_hours", 7)),
            "Parent_Education_Level": request.form.get("parent_edu", "High School"),
            "Health_Issues": request.form.get("health", "None"),
            "Internet_Access": request.form.get("internet", "Yes"),
            "Attendence": float(request.form.get("attendance", 75)),
            "internal_test 1": float(request.form.get("internal1", 0)),
            "internal_test 2": float(request.form.get("internal2", 0)),
            "Assignment_score": float(request.form.get("assignment", 0)),
            "Extracurricular_Activities": request.form.get("extra", "No"),
            "Previous_Exam_Score": float(request.form.get("previous", 0)),
            "Final_Exam_Score": float(request.form.get("final_score", 0)),
        }
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        df = df.drop_duplicates(subset=["Student_ID"], keep="last")
        df.to_csv(DATA_FILE, index=False)
        flash(f"Student {new['Student_ID']} added.", "success")
        return redirect(url_for("admin_students"))
    return render_template("add_student.html")

@app.route("/admin/delete-student/<sid>")
@login_required("admin")
def admin_delete_student(sid):
    df = load_csv()
    df = df[df["Student_ID"].astype(str) != sid]
    df.to_csv(DATA_FILE, index=False)
    flash("Student removed.", "success")
    return redirect(url_for("admin_students"))

@app.route("/admin/upload-students", methods=["POST"])
@login_required("admin")
def admin_upload_students():
    file = request.files.get("file")
    if not file:
        flash("No file uploaded.", "danger")
        return redirect(url_for("admin_students"))
    new_df = pd.read_csv(file) if file.filename.endswith(".csv") else pd.read_excel(file)
    new_df["Health_Issues"] = new_df["Health_Issues"].fillna("None")
    existing = load_csv()
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Student_ID"], keep="last")
    combined.to_csv(DATA_FILE, index=False)
    flash(f"Uploaded {len(new_df)} rows.", "success")
    return redirect(url_for("admin_students"))


# ════════════════════════════════════════════════════════════════════
# TEACHER
# ════════════════════════════════════════════════════════════════════
@app.route("/teacher")
@login_required("teacher")
def teacher_dashboard():
    teacher = db.get_teacher_by_user_id(session["user_id"])
    df = load_csv()
    df["risk"] = np.where(df["Final_Exam_Score"] < 70, "At Risk", "Safe")
    avg_score  = round(df["Final_Exam_Score"].mean(), 1)
    avg_att    = round(df["Attendence"].mean(), 1)
    risk_pct   = round((df["risk"] == "At Risk").mean() * 100, 1)
    return render_template("teacher_dashboard.html",
        teacher=teacher,
        total=len(df),
        at_risk=len(df[df["risk"] == "At Risk"]),
        avg_score=avg_score, avg_att=avg_att, risk_pct=risk_pct,
        risk_students=df[df["risk"] == "At Risk"].head(10).to_dict(orient="records"),
    )

@app.route("/teacher/predict", methods=["GET", "POST"])
@login_required("teacher")
def teacher_predict():
    result = None
    score  = None
    suggestions = []
    if request.method == "POST":
        try:
            score  = predict_score(request.form)
            result = risk_label(score)
            if score is not None and score < 80:
                suggestions = generate_suggestions(score, request.form)
        except Exception as e:
            flash(f"Prediction error: {e}", "danger")
    return render_template("predict_single.html",
                           result=result, score=score, suggestions=suggestions)

@app.route("/teacher/bulk-predict", methods=["GET", "POST"])
@login_required("teacher")
def teacher_bulk_predict():
    if request.method == "GET":
        return render_template("bulk_predict.html")

    if model is None:
        flash("Model not found. Run train_model.py first.", "danger")
        return render_template("bulk_predict.html")

    file = request.files.get("file")
    if not file:
        flash("No file uploaded.", "danger")
        return render_template("bulk_predict.html")

    try:
        df = pd.read_csv(file) if file.filename.lower().endswith(".csv") else pd.read_excel(file)
        df["Health_Issues"] = df["Health_Issues"].fillna("None")
    except Exception as e:
        flash(f"Could not read file: {e}", "danger")
        return render_template("bulk_predict.html")

    results = []
    for _, row in df.iterrows():
        try:
            score = predict_score(row.to_dict())
        except Exception:
            score = None
        risk = risk_label(score)
        results.append({
            "Student_ID":      row.get("Student_ID", "—"),
            "Predicted_Score": score if score is not None else "Error",
            "Risk":            risk,
            "Attendance":      row.get("Attendence", "—"),
            "Study_Hours":     row.get("Study_hours", "—"),
            "Internal_1":      row.get("internal_test 1", "—"),
            "Internal_2":      row.get("internal_test 2", "—"),
            "Assignment":      row.get("Assignment_score", "—"),
            "Previous":        row.get("Previous_Exam_Score", "—"),
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(BULK_OUT, index=False)

    valid_scores = [r["Predicted_Score"] for r in results if isinstance(r["Predicted_Score"], float)]
    avg_score = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else "—"

    return render_template("bulk_predict.html",
        results=results, show_results=True,
        total=len(results),
        at_risk=sum(1 for r in results if r["Risk"] == "At Risk"),
        avg_score=avg_score,
    )

@app.route("/teacher/bulk-predict/download")
@login_required("teacher")
def bulk_download():
    if not os.path.exists(BULK_OUT):
        flash("No results to download yet.", "warning")
        return redirect(url_for("teacher_bulk_predict"))
    return send_file(BULK_OUT, as_attachment=True, download_name="predictions.csv")

@app.route("/teacher/students")
@login_required("teacher")
def teacher_students():
    q = request.args.get("q", "")
    df = load_csv()
    df["risk"] = np.where(df["Final_Exam_Score"] < 70, "At Risk", "Safe")
    if q:
        df = df[df["Student_ID"].astype(str).str.contains(q, na=False)]
    return render_template("students_table.html",
                           students=df.to_dict(orient="records"), q=q)


# ════════════════════════════════════════════════════════════════════
# TEACHER — Add student(s)
# ════════════════════════════════════════════════════════════════════
@app.route("/teacher/add-student", methods=["GET", "POST"])
@login_required("teacher")
def teacher_add_student():
    if request.method == "POST":
        df = load_csv()
        f = request.form
        new = {
            "Student_ID":              f["student_id"].strip(),
            "Class":                   f.get("class_", "9th"),
            "section":                 f.get("section", "A"),
            "Age":                     ival(f, "age", 14),
            "Gender":                  f.get("gender", "Male"),
            "Study_hours":             fval(f, "study_hours", 3),
            "Sleep_hours":             fval(f, "sleep_hours", 7),
            "Parent_Education_Level":  f.get("parent_edu", "High School"),
            "Health_Issues":           f.get("health", "None"),
            "Internet_Access":         f.get("internet", "Yes"),
            "Attendence":              fval(f, "attendance", 75),
            "internal_test 1":         fval(f, "internal1", 0),
            "internal_test 2":         fval(f, "internal2", 0),
            "Assignment_score":        fval(f, "assignment", 0),
            "Extracurricular_Activities": f.get("extra", "No"),
            "Previous_Exam_Score":     fval(f, "previous", 0),
            "Final_Exam_Score":        fval(f, "final_score", 0),
        }
        sid = new["Student_ID"]
        if not sid:
            flash("Student ID is required.", "danger")
            return render_template("teacher_add_student.html")
        if sid in df["Student_ID"].astype(str).values:
            flash(f"Student ID {sid} already exists.", "danger")
            return render_template("teacher_add_student.html")
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        flash(f"Student {sid} added successfully.", "success")
        return redirect(url_for("teacher_students"))
    return render_template("teacher_add_student.html")

@app.route("/teacher/upload-students", methods=["POST"])
@login_required("teacher")
def teacher_upload_students():
    file = request.files.get("file")
    if not file:
        flash("No file uploaded.", "danger")
        return redirect(url_for("teacher_students"))
    try:
        new_df = pd.read_csv(file) if file.filename.endswith(".csv") else pd.read_excel(file)
        new_df["Health_Issues"] = new_df["Health_Issues"].fillna("None")
    except Exception as e:
        flash(f"Could not read file: {e}", "danger")
        return redirect(url_for("teacher_students"))
    existing = load_csv()
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Student_ID"], keep="last")
    combined.to_csv(DATA_FILE, index=False)
    flash(f"Uploaded {len(new_df)} student(s).", "success")
    return redirect(url_for("teacher_students"))


# ════════════════════════════════════════════════════════════════════
# STUDENT
# ════════════════════════════════════════════════════════════════════
@app.route("/student")
@login_required("student")
def student_dashboard():
    student = db.get_student_by_user_id(session["user_id"])
    df = load_csv()

    row = None
    score = None
    suggestions = []
    result = None

    if student:
        match = df[df["Student_ID"].astype(str) == str(student["student_code"])]
        if not match.empty:
            row   = match.iloc[0].to_dict()
            score = predict_score(row)
            result = risk_label(score)
            if score is not None and score < 80:
                suggestions = generate_suggestions(score, row)

    return render_template("student_dashboard.html",
        student=student, row=row, score=score,
        result=result, suggestions=suggestions,
    )


# ════════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, port=5000)