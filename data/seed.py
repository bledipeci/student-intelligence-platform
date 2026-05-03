import sys
import os
import random
import pandas as pd
import kagglehub
from faker import Faker

# add the project root to the path so we can import from other folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import get_connection, init_db

# faker generates random realistic names for each student
fake = Faker()
# setting a seed makes the random names the same every time we run this
random.seed(42)


def load_kaggle_data():
    # download the dataset from kaggle (uses cache after the first download)
    path = kagglehub.dataset_download("whenamancodes/student-performance")
    # both files are actually excel format despite the .csv extension
    maths = pd.read_excel(os.path.join(path, "Maths.csv"))
    maths["subject"] = "Maths"
    portuguese = pd.read_excel(os.path.join(path, "Portuguese.csv"))
    portuguese["subject"] = "Portuguese"
    # combine both subjects into one dataframe
    df = pd.concat([maths, portuguese], ignore_index=True)
    return df


def seed():
    # delete the old database if it exists so we always start clean
    db_path = "data/sip.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # create all the tables from schema.sql
    init_db()
    conn = get_connection()

    df = load_kaggle_data()

    # go through every student row in the dataset
    for _, row in df.iterrows():
        # insert the student's personal and family info
        # we generate a fake name since the real dataset doesn't include names
        cursor = conn.execute("""
            INSERT INTO students (name, school, sex, age, address, famsize, Pstatus,
                                  Medu, Fedu, Mjob, Fjob, guardian, internet, higher)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fake.name(),
            row.school, row.sex, row.age, row.address, row.famsize, row.Pstatus,
            row.Medu, row.Fedu, row.Mjob, row.Fjob, row.guardian, row.internet, row.higher
        ))
        # lastrowid gives us the auto-generated id of the student we just inserted
        student_id = cursor.lastrowid

        # insert the student's grades for this subject
        conn.execute("""
            INSERT INTO grades (student_id, subject, G1, G2, G3)
            VALUES (?, ?, ?, ?, ?)
        """, (student_id, row.subject, row.G1, row.G2, row.G3))

        # insert the student's behavior and lifestyle data
        conn.execute("""
            INSERT INTO behavior (student_id, studytime, traveltime, failures,
                                  absences, goout, Dalc, Walc, health, freetime, famrel)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id, row.studytime, row.traveltime, row.failures,
            row.absences, row.goout, row.Dalc, row.Walc, row.health, row.freetime, row.famrel
        ))

        # insert the student's support and extracurricular data
        conn.execute("""
            INSERT INTO support (student_id, schoolsup, famsup, paid, activities, nursery, romantic, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (student_id, row.schoolsup, row.famsup, row.paid, row.activities, row.nursery, row.romantic, row.reason))

    # save all inserts to the database at once
    conn.commit()
    conn.close()

    total = len(df)
    print(f"Seeded {total} student records ({total // 2} Maths + {total // 2} Portuguese).")


if __name__ == "__main__":
    seed()
