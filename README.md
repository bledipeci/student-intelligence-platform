# student intelligence platform (sip)

an end-to-end machine learning system that predicts student grades, classifies risk levels, and gives personalised recommendations — built with python, sql, and scikit-learn.

---

## what it does

- stores real student data in a **sqlite database** with 4 tables and 1,048 students
- runs an **etl pipeline** that joins the tables and builds 30 features for the models
- trains **3 machine learning models** to predict grades, classify pass/fail, and categorise performance
- uses a **decision engine** to assign risk levels and generate recommendations for each student
- shows everything in a **streamlit dashboard** with live predictions and pdf report downloads

---

## models

| model | task | result |
|---|---|---|
| linear regression | predict final grade (0–20) | mae: 1.01 points |
| logistic regression | pass / fail classification | accuracy: 89.5% |
| random forest | low / medium / high category | cv accuracy: 85.9% ± 1.4% |

---

## dataset

[uci student performance dataset](https://www.kaggle.com/datasets/whenamancodes/student-performance) from kaggle — real data from portuguese secondary schools with grades, attendance, study habits, family background, and lifestyle across maths and portuguese subjects.

---

## project structure

```
├── database/
│   ├── schema.sql       # defines all 4 tables
│   └── db.py            # handles database connections
├── data/
│   └── seed.py          # downloads kaggle data and loads it into the database
├── src/
│   ├── pipeline.py      # etl: sql → pandas + feature engineering
│   ├── models.py        # trains and saves all three models
│   ├── engine.py        # risk classification + recommendations
│   └── report.py        # pdf report generation
├── app/
│   └── dashboard.py     # streamlit dashboard
├── models/              # saved model files (auto-generated on first run)
├── main.py              # run this to start everything
└── requirements.txt
```

---

## getting started

### 1. install dependencies

```bash
pip install -r requirements.txt
```

### 2. run

```bash
python main.py
```

this will download the dataset, seed the database, train the models, and open the dashboard at `http://localhost:8501`.

on every run after the first, seeding and training are skipped automatically.

---

## dashboard tabs

| tab | what it shows |
|---|---|
| overview | feature importance, grade distribution, risk breakdown, pass/fail by subject |
| student browser | searchable table of all 1,048 students with colour-coded risk levels |
| student report | full profile, predictions, recommendations, and pdf download per student |
| live prediction | adjust sliders and see predictions update in real time |

---

## tech stack

- **database** — sqlite
- **data processing** — pandas
- **machine learning** — scikit-learn
- **dashboard** — streamlit + plotly
- **pdf generation** — reportlab
- **data source** — kagglehub
