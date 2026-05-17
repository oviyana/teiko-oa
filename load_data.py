import polars as pl
import sqlite3
import os

DB_FILE = 'cell-count-3nf.db'
CSV_FILE = 'cell-count.csv'

def setup_database():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        
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
        b_cell INTEGER, cd8_t_cell INTEGER, cd4_t_cell INTEGER, 
        nk_cell INTEGER, monocyte INTEGER,
        FOREIGN KEY (subject) REFERENCES subjects (subject))''')
    
    conn.commit()
    conn.close()

def load_data():
    df = pl.read_csv(CSV_FILE)
    uri = f"sqlite://{DB_FILE}"

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

    # Final Table: Samples
    df.select([
        "sample", "subject", "sample_type", "time_from_treatment_start",
        "b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"
    ]).write_database("samples", uri, if_table_exists="append", engine="adbc")

if __name__ == "__main__":
    setup_database()
    load_data()
    print("Database initialized")
