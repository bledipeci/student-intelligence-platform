import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline, get_feature_columns
from src.models import load_models
from src.engine import classify_risk, generate_recommendations
from src.report import generate_pdf

st.set_page_config(page_title="Student Intelligence Platform", layout="wide")

RISK_COLORS = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}


def load_data():
    df = run_pipeline()
    lr, _, dt = load_models()

    X = df[lr.feature_names_in_].apply(pd.to_numeric, errors="coerce").fillna(0)

    df["predicted_grade"] = lr.predict(X).round(1)
    df["pass_fail"]       = (df["predicted_grade"] >= 10).map({True: "Pass", False: "Fail"})
    df["category"]        = dt.predict(X)
    df["risk_level"]      = df.apply(
        lambda r: classify_risk(r["predicted_grade"], 1 if r["pass_fail"] == "Pass" else 0, r["category"]),
        axis=1
    )
    return df


if "df" not in st.session_state or st.session_state.get("df_version") != 3:
    st.session_state.df = load_data()
    st.session_state.df_version = 3

df = st.session_state.df

# ── Sidebar navigation ─────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Dashboard"], label_visibility="collapsed")

st.sidebar.divider()

# ── Home page ──────────────────────────────────────────────────────────────
if page == "Home":
    st.title("Student Intelligence Platform")
    st.subheader("An end-to-end ML system for student analytics and risk assessment")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### What it does")
        st.markdown("""
- Loads real student data from a **SQL database**
- Engineers features and runs an **ETL pipeline** with pandas
- Trains **3 machine learning models** to predict performance
- Classifies each student into a **risk level** with actionable recommendations
- Generates a downloadable **PDF report** per student
        """)

        st.markdown("### Tech Stack")
        st.markdown("""
| Layer | Technology |
|---|---|
| Database | SQLite |
| Data processing | pandas |
| Machine learning | scikit-learn |
| Dashboard | Streamlit + Plotly |
| PDF generation | ReportLab |
        """)

    with col2:
        st.markdown("### Models")
        st.markdown("""
**Linear Regression** — predicts the student's final grade (0–20)

**Logistic Regression** — classifies whether the student will pass or fail

**Decision Tree** — categorises performance as Low, Medium, or High
        """)

        st.markdown("### Dataset")
        st.markdown("""
Real-world data from the [UCI Student Performance Dataset](https://www.kaggle.com/datasets/whenamancodes/student-performance)

- **1,048 students** across Maths and Portuguese
- **33 features** including grades, attendance, study habits, and family background
        """)

    st.divider()
    st.markdown("### How to use")
    col_a, col_b, col_c = st.columns(3)
    col_a.info("**1. Overview**\nSee grade distribution, risk breakdown, and pass rates across all students.")
    col_b.info("**2. Student Browser**\nSearch and filter all students by name, subject, or risk level.")
    col_c.info("**3. Student Report**\nSelect any student to see their predictions, risk level, and download a PDF report.")

    st.stop()

# ── Dashboard filters (only shown when on Dashboard page) ──────────────────
st.sidebar.markdown("**Filters**")
subject_filter = st.sidebar.selectbox("Subject", ["All", "Maths", "Portuguese"])
risk_filter = st.sidebar.multiselect(
    "Risk Level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"]
)
if not risk_filter:
    st.sidebar.warning("Select at least one risk level.")
    risk_filter = ["HIGH", "MEDIUM", "LOW"]

filtered = df.copy()
if subject_filter != "All":
    filtered = filtered[filtered["subject"] == subject_filter]
filtered = filtered[filtered["risk_level"].isin(risk_filter)]

# ── Header ─────────────────────────────────────────────────────────────────
st.title("Student Intelligence Platform")
st.caption("Predictive analytics and risk assessment powered by Machine Learning")
st.divider()

# ── Top metrics ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students",       len(filtered))
col2.metric("Avg Predicted Grade",  f"{filtered['predicted_grade'].mean():.1f} / 20")
col3.metric("Pass Rate",            f"{(filtered['pass_fail'] == 'Pass').mean():.1%}")
col4.metric("High Risk Students",   int((filtered["risk_level"] == "HIGH").sum()))

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Student Browser", "Student Report", "Live Prediction"])


# ── Tab 1: Overview ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Feature Importance (Random Forest)")
    st.caption("How much each input contributes to predicting student performance category.")
    _, _, rf = load_models()
    feat_imp = pd.DataFrame({
        "Feature":    get_feature_columns(),
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=True).tail(15)
    fig = px.bar(
        feat_imp, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
    )
    fig.update_layout(
        margin=dict(t=10, b=20),
        coloraxis_showscale=False,
        yaxis_title="",
        xaxis_title="Importance Score",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk Level Breakdown")
        risk_counts = filtered["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig = px.pie(
            risk_counts, names="Risk Level", values="Count",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            hole=0.4,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=14)
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Grade Distribution")
        fig = px.histogram(
            filtered, x="predicted_grade", nbins=20,
            color_discrete_sequence=["#3498db"],
            labels={"predicted_grade": "Predicted Final Grade"},
        )
        fig.update_layout(bargap=0.05, margin=dict(t=20, b=20), yaxis_title="Number of Students")
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Pass vs Fail by Subject")
        pf = filtered.groupby(["subject", "pass_fail"]).size().reset_index(name="count")
        fig = px.bar(
            pf, x="subject", y="count", color="pass_fail",
            color_discrete_map={"Pass": "#2ecc71", "Fail": "#e74c3c"},
            barmode="group",
            labels={"subject": "Subject", "count": "Students", "pass_fail": "Result"},
        )
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Avg Predicted Grade by Past Failures")
        fail_avg = (
            filtered.groupby("failures")["predicted_grade"]
            .mean().round(1).reset_index()
        )
        fail_avg.columns = ["Failures", "Avg Grade"]
        fail_avg["Failures"] = fail_avg["Failures"].astype(int).astype(str)
        fig = px.bar(
            fail_avg, x="Failures", y="Avg Grade",
            text="Avg Grade",
            color="Avg Grade",
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            labels={"Avg Grade": "Avg Predicted Grade", "Failures": "Past Failures"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            margin=dict(t=30, b=20),
            yaxis=dict(range=[0, 20], title="Avg Predicted Grade"),
            xaxis_title="Number of Past Failures",
            coloraxis_showscale=False,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Student Browser ─────────────────────────────────────────────────
with tab2:
    st.subheader(f"Showing {len(filtered)} students")

    search = st.text_input("Search by name", placeholder="Type a name...")
    if search:
        filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]

    display_df = filtered[["name", "subject", "G1", "G2", "G3", "predicted_grade", "pass_fail", "category", "risk_level"]].copy()
    display_df = display_df.rename(columns={
        "predicted_grade": "Predicted Grade",
        "pass_fail":       "Pass/Fail",
        "risk_level":      "Risk",
        "category":        "Category",
    })
    display_df = display_df.reset_index(drop=True)
    display_df.index += 1

    def highlight_risk(val):
        styles = {
            "HIGH":   "background-color:#8B0000; color:#ffffff; font-weight:bold",
            "MEDIUM": "background-color:#8B6914; color:#ffffff; font-weight:bold",
            "LOW":    "background-color:#1a5c38; color:#ffffff; font-weight:bold",
        }
        return styles.get(val, "")

    st.dataframe(
        display_df.style.map(highlight_risk, subset=["Risk"]),
        use_container_width=True,
        height=520,
    )


# ── Tab 3: Student Report ──────────────────────────────────────────────────
with tab3:
    st.subheader("Individual Student Report")

    all_names = df["name"].tolist()
    selected_name = st.selectbox("Select a student", all_names)

    student_row = df[df["name"] == selected_name].iloc[0]
    recommendations = generate_recommendations(student_row, student_row["risk_level"])

    st.divider()

    # ── Predictions row ────────────────────────────────────────────────────
    st.markdown("### Predictions")
    risk  = student_row["risk_level"]
    color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}[risk]

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Predicted Grade", f"{student_row['predicted_grade']} / 20",
              delta=f"{student_row['predicted_grade'] - student_row['G2']:.1f} vs Period 2")
    p2.metric("Pass / Fail",          student_row["pass_fail"])
    p3.metric("Performance Category", student_row["category"])
    p4.markdown("**Risk Level**")
    p4.markdown(f":{color}[**{risk}**]")

    st.divider()

    # ── Profile — 3 horizontal groups ─────────────────────────────────────
    st.markdown("### Profile")

    def _decode_job(val):
        str_map = {"at_home": "At Home", "other": "Other", "services": "Services", "health": "Health", "teacher": "Teacher"}
        int_map = {1: "At Home", 2: "Other", 3: "Services", 4: "Health", 5: "Teacher"}
        try:
            return int_map.get(int(float(val)), "Other")
        except (ValueError, TypeError):
            return str_map.get(str(val).lower(), str(val).title())

    study_label = {1: "<2h", 2: "2-5h", 3: "5-10h", 4: ">10h"}

    def field(label, value):
        st.markdown(f"**{label}**")
        st.markdown(str(value))

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("##### Personal")
        field("Name",         student_row["name"])
        field("Subject",      student_row["subject"])
        field("School",       "Gabriel Pereira" if student_row["school"] == 1 else "Mousinho da Silveira")
        field("Age",          int(student_row["age"]))
        field("Sex",          "Male" if student_row["sex"] == 1 else "Female")
        field("Address",      "Urban" if student_row["address"] == 1 else "Rural")
        field("Family size",  "More than 3" if student_row["famsize"] == 1 else "3 or fewer")
        field("Parents",      "Living together" if student_row["Pstatus"] == 1 else "Apart")
        field("Mother's job", _decode_job(student_row["Mjob"]))
        field("Father's job", _decode_job(student_row["Fjob"]))

    with col_b:
        st.markdown("##### Academics")
        field("Period 1 Grade", f"{int(student_row['G1'])} / 20")
        field("Period 2 Grade", f"{int(student_row['G2'])} / 20")
        field("Final Grade",    f"{int(student_row['G3'])} / 20")
        field("Study time",     study_label.get(int(student_row["studytime"]), "Unknown"))
        field("Absences",       int(student_row["absences"]))
        field("Past failures",  int(student_row["failures"]))

    with col_c:
        st.markdown("##### Support & Lifestyle")
        field("Extra support",   "Yes" if student_row["schoolsup"] == 1 else "No")
        field("Family support",  "Yes" if student_row["famsup"] == 1 else "No")
        field("Paid classes",    "Yes" if student_row["paid"] == 1 else "No")
        field("Activities",      "Yes" if student_row["activities"] == 1 else "No")
        field("Nursery school",  "Yes" if student_row["nursery"] == 1 else "No")
        field("Wants higher ed", "Yes" if student_row["higher"] == 1 else "No")
        field("Internet",        "Yes" if student_row["internet"] == 1 else "No")
        field("Romantic rel.",   "Yes" if student_row["romantic"] == 1 else "No")

    st.markdown("### Recommendations")
    for rec in recommendations:
        st.info(rec)

    st.divider()
    pdf_bytes = generate_pdf(
        student_row, recommendations,
        risk_level=student_row["risk_level"],
        predicted_grade=student_row["predicted_grade"],
        pass_fail=student_row["pass_fail"],
        category=student_row["category"],
    )
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=f"{student_row['name'].replace(' ', '_')}_report.pdf",
        mime="application/pdf",
    )


# ── Tab 4: Live Prediction ─────────────────────────────────────────────────
with tab4:
    st.subheader("Live Prediction")
    st.caption("Adjust the inputs below to predict a hypothetical student's outcome instantly.")

    lr, _, rf = load_models()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Academic**")
        g1         = st.slider("Period 1 Grade",    0, 20, 10)
        g2         = st.slider("Period 2 Grade",    0, 20, 10)
        failures   = st.slider("Past Failures",     0, 3,  0)
        absences   = st.slider("Absences",          0, 75, 5)
        studytime  = st.select_slider("Study Time", options=[1, 2, 3, 4],
                                      format_func=lambda x: {1:"<2h",2:"2-5h",3:"5-10h",4:">10h"}[x], value=2)

    with col2:
        st.markdown("**Lifestyle**")
        alcohol    = st.slider("Alcohol (avg)",     1.0, 5.0, 1.5, step=0.5)
        goout      = st.slider("Going Out (1-5)",   1, 5, 3)
        health     = st.slider("Health (1-5)",      1, 5, 3)
        freetime   = st.slider("Free Time (1-5)",   1, 5, 3)
        romantic   = st.selectbox("Romantic Rel.",  ["No", "Yes"])

    with col3:
        st.markdown("**Background**")
        higher     = st.selectbox("Wants Higher Ed", ["Yes", "No"])
        internet   = st.selectbox("Internet",        ["Yes", "No"])
        famsup     = st.selectbox("Family Support",  ["Yes", "No"])
        schoolsup  = st.selectbox("School Support",  ["No", "Yes"])
        parent_edu = st.slider("Parent Edu (avg)",   0.0, 4.0, 2.0, step=0.5)

    # Build a single-row DataFrame with all required features
    sample = pd.DataFrame([{
        "G1": g1, "G2": g2,
        "studytime": studytime, "failures": failures,
        "absences": absences, "traveltime": 1,
        "alcohol": alcohol, "goout": goout,
        "freetime": freetime, "romantic": 1 if romantic == "Yes" else 0,
        "health": health, "famrel": 3,
        "famsize": 1, "Pstatus": 1,
        "schoolsup": 1 if schoolsup == "Yes" else 0,
        "famsup": 1 if famsup == "Yes" else 0,
        "paid": 0, "activities": 0, "nursery": 1,
        "higher": 1 if higher == "Yes" else 0,
        "internet": 1 if internet == "Yes" else 0,
        "sex": 0, "address": 1, "age": 17, "school": 1,
        "parent_edu": parent_edu, "Medu": int(parent_edu),
        "Fedu": int(parent_edu), "Mjob": 2, "Fjob": 2,
    }])[lr.feature_names_in_]

    pred_grade    = round(lr.predict(sample)[0], 1)
    pred_pass     = 1 if pred_grade >= 10 else 0
    pred_category = rf.predict(sample)[0]
    pred_risk     = classify_risk(pred_grade, pred_pass, pred_category)

    st.divider()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Predicted Grade",  f"{pred_grade} / 20")
    r2.metric("Pass / Fail",      "Pass" if pred_pass == 1 else "Fail")
    r3.metric("Category",         pred_category)
    risk_color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}[pred_risk]
    r4.markdown(f"**Risk Level**")
    r4.markdown(f":{risk_color}[**{pred_risk}**]")
