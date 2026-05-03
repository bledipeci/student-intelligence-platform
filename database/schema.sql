CREATE TABLE IF NOT EXISTS students (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    school      TEXT,
    sex         TEXT,
    age         INTEGER,
    address     TEXT,
    famsize     TEXT,
    Pstatus     TEXT,
    Medu        INTEGER,  -- mother's education (0-4)
    Fedu        INTEGER,  -- father's education (0-4)
    Mjob        TEXT,
    Fjob        TEXT,
    guardian    TEXT,
    internet    TEXT,
    higher      TEXT      -- wants to pursue higher education
);

CREATE TABLE IF NOT EXISTS grades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER REFERENCES students(id),
    subject     TEXT,     -- 'Maths' or 'Portuguese'
    G1          INTEGER,  -- first period grade  (0-20)
    G2          INTEGER,  -- second period grade (0-20)
    G3          INTEGER   -- final grade         (0-20)
);

CREATE TABLE IF NOT EXISTS behavior (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER REFERENCES students(id),
    studytime   INTEGER,  -- weekly study hours (1=<2h, 4=>10h)
    traveltime  INTEGER,  -- home to school travel time (1=<15min, 4=>1h)
    failures    INTEGER,  -- number of past class failures
    absences    INTEGER,  -- number of school absences
    goout       INTEGER,  -- going out with friends (1-5)
    Dalc        INTEGER,  -- workday alcohol consumption (1-5)
    Walc        INTEGER,  -- weekend alcohol consumption (1-5)
    health      INTEGER,  -- current health status (1-5)
    freetime    INTEGER,  -- free time after school (1-5)
    famrel      INTEGER   -- quality of family relationships (1-5)
);

CREATE TABLE IF NOT EXISTS support (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER REFERENCES students(id),
    schoolsup   TEXT,     -- extra educational support
    famsup      TEXT,     -- family educational support
    paid        TEXT,     -- extra paid classes
    activities  TEXT,     -- extra-curricular activities
    nursery     TEXT,     -- attended nursery school
    romantic    TEXT,     -- in a romantic relationship
    reason      TEXT      -- reason for choosing this school
);
