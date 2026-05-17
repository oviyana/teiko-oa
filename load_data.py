import sqlite3
from pathlib import Path

import polars as pl

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "cell-count-3nf.db"
CSV_FILE = BASE_DIR / "cell-count.csv"
CELL_COLS = ["b_cell", "cd4_t_cell", "cd8_t_cell", "nk_cell", "monocyte"]

def setup_database():
    if DB_FILE.exists():
        DB_FILE.unlink()
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Projects
    cursor.execute("CREATE TABLE projects (project TEXT PRIMARY KEY)")

    # 2. Treatments Table with the Unique Composite Key
    cursor.execute('''
    CREATE TABLE treatments (
        treatment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        treatment TEXT,
        condition TEXT,
        UNIQUE(treatment, condition) -- This is the key line
    )''')

    # 3. Subjects
    cursor.execute('''
    CREATE TABLE subjects (
        subject TEXT PRIMARY KEY,
        project TEXT,
        age INTEGER,
        sex TEXT,
        FOREIGN KEY (project) REFERENCES projects (project))''')

    # 4. Subject Outcomes (Points to the Treatment ID)
    cursor.execute('''
    CREATE TABLE subject_outcomes (
        subject TEXT PRIMARY KEY,
        treatment_id INTEGER,
        response TEXT,
        FOREIGN KEY (subject) REFERENCES subjects (subject),
        FOREIGN KEY (treatment_id) REFERENCES treatments (treatment_id))''')

    # 5. Samples
    cursor.execute('''
    CREATE TABLE samples (
        sample TEXT PRIMARY KEY,
        subject TEXT,
        sample_type TEXT,
        time_from_treatment_start INTEGER,
        FOREIGN KEY (subject) REFERENCES subjects (subject))''')

    # 6. Cell populations
    cursor.execute("CREATE TABLE populations (population TEXT PRIMARY KEY)")

    # 7. Cell counts in long format so new populations can be added without changing schema
    cursor.execute('''
    CREATE TABLE cell_counts (
        sample TEXT,
        population TEXT,
        count INTEGER NOT NULL CHECK (count >= 0),
        PRIMARY KEY (sample, population),
        FOREIGN KEY (sample) REFERENCES samples (sample),
        FOREIGN KEY (population) REFERENCES populations (population))''')
    
    conn.commit()
    conn.close()

def load_data():
    df = pl.read_csv(CSV_FILE)
    uri = f"sqlite:///{DB_FILE.as_posix()}"

    # Insert Projects and Subjects first
    df.select("project").unique().write_database("projects", uri, if_table_exists="append", engine="adbc")
    df.select(["subject", "project", "age", "sex"]).unique().write_database("subjects", uri, if_table_exists="append", engine="adbc")

    # Insert unique Treatment/Condition pairs
    treatments_df = df.select([
        "treatment", 
        "condition"
    ]).unique()
    treatments_df.write_database("treatments", uri, if_table_exists="append", engine="adbc")

    # Join the original data to these IDs to populate the outcomes table
    db_treatments = pl.read_database_uri("SELECT * FROM treatments", uri, engine="adbc")
    
    outcomes_df = (
        df.select(["subject", "treatment", "condition", "response"])
        .unique()
        .join(db_treatments, 
              left_on=["treatment", "condition"], 
              right_on=["treatment", "condition"])
        .select(["subject", "treatment_id", "response"])
    )
    outcomes_df.write_database("subject_outcomes", uri, if_table_exists="append", engine="adbc")

    # Samples table stores sample metadata only.
    df.select([
        "sample", "subject", "sample_type", "time_from_treatment_start"
    ]).write_database("samples", uri, if_table_exists="append", engine="adbc")

    pl.DataFrame({"population": CELL_COLS}).write_database("populations", uri, if_table_exists="append", engine="adbc")

    (
        df.select(["sample", *CELL_COLS])
        .unpivot(index="sample", on=CELL_COLS, variable_name="population", value_name="count")
        .write_database("cell_counts", uri, if_table_exists="append", engine="adbc")
    )

if __name__ == "__main__":
    setup_database()
    load_data()
    print("Database initialized")
