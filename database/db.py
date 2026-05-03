import sqlite3
import os

# build absolute paths so this works no matter where the app is launched from
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_ROOT, "data", "sip.db")
SCHEMA_PATH = os.path.join(_ROOT, "database", "schema.sql")


def get_connection():
    # open a connection to the sqlite database file
    conn = sqlite3.connect(DB_PATH)
    # this lets us access query results by column name like row["grade"] instead of row[0]
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    # read the sql file that defines all the tables
    with open(SCHEMA_PATH, "r") as f:
        schema = f.read()
    # run all the create table statements at once
    conn = get_connection()
    conn.executescript(schema)
    conn.commit()
    conn.close()
