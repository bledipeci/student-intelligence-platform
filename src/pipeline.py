import pandas as pd
import sys
import os

# add the project root to the path so we can import from other folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import get_connection


def extract():
    # read each table from the database into a separate dataframe
    conn = get_connection()
    students = pd.read_sql("SELECT * FROM students", conn)
    grades   = pd.read_sql("SELECT * FROM grades",   conn)
    behavior = pd.read_sql("SELECT * FROM behavior", conn)
    support  = pd.read_sql("SELECT * FROM support",  conn)
    conn.close()
    return students, grades, behavior, support


def engineer_features(students, grades, behavior, support):
    # join all four tables into one flat dataframe using student_id as the key
    # we start from grades because every student has exactly one grades row
    df = grades.merge(students, left_on="student_id", right_on="id", suffixes=("", "_student"))
    df = df.merge(behavior, on="student_id")
    df = df.merge(support,  on="student_id")

    # convert all yes/no text columns to 1 and 0 so the model can use them
    yes_no_cols = ["schoolsup", "famsup", "paid", "activities", "nursery", "romantic", "internet", "higher"]
    for col in yes_no_cols:
        df[col] = (df[col] == "yes").astype(int)

    # convert the remaining text columns to numbers using binary encoding
    df["sex"]     = (df["sex"] == "M").astype(int)       # female=0, male=1
    df["address"] = (df["address"] == "U").astype(int)   # rural=0, urban=1
    df["famsize"] = (df["famsize"] == "GT3").astype(int) # small family=0, large family=1
    df["Pstatus"] = (df["Pstatus"] == "T").astype(int)   # parents apart=0, together=1
    df["school"]  = (df["school"] == "GP").astype(int)   # mousinho da silveira=0, gabriel pereira=1

    # encode job type as a number ranked by how much education the job typically requires
    # at_home=1 (lowest) up to teacher=5 (highest)
    job_rank = {"at_home": 1, "other": 2, "services": 3, "health": 4, "teacher": 5}
    df["Mjob"] = df["Mjob"].map(job_rank)
    df["Fjob"] = df["Fjob"].map(job_rank)

    # combine both parents' education into one average score
    df["parent_edu"] = (df["Medu"] + df["Fedu"]) / 2
    # combine weekday and weekend alcohol into one average score
    df["alcohol"]    = (df["Dalc"] + df["Walc"]) / 2

    # force every feature column to be a proper number
    # if anything couldn't be converted it becomes 0 instead of crashing
    for col in get_feature_columns():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def get_feature_columns():
    # this is the list of columns we feed into the machine learning models
    return [
        "G1", "G2",
        "studytime", "failures", "absences", "traveltime",
        "alcohol", "goout", "freetime", "romantic",
        "health", "famrel", "famsize", "Pstatus",
        "schoolsup", "famsup", "paid", "activities", "nursery",
        "higher", "internet", "sex", "address", "age", "school",
        "parent_edu", "Medu", "Fedu", "Mjob", "Fjob",
    ]


def run_pipeline():
    # run the full extract and transform process and return the feature table
    students, grades, behavior, support = extract()
    df = engineer_features(students, grades, behavior, support)
    return df


if __name__ == "__main__":
    df = run_pipeline()
    print(f"Dataset shape: {df.shape}")
    print(df[get_feature_columns() + ["G3"]].head(5).to_string(index=False))
