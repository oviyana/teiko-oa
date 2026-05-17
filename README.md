# teiko-oa

## Run Instructions (GitHub Codespaces)

1. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate

   pip install -r requirements.txt
   ```
2. Build the SQLite database from the CSV:
   ```bash
   python load_data.py
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```
    or use link: [https://teiko-oa-bsuuxrswyu6s3rb5lxda5y.streamlit.app)


## Relational Database Schema

The schema in `load_data.py` uses 7 tables:

- `projects(project PK)`
- `treatments(treatment_id PK, treatment, condition, UNIQUE(treatment, condition))`
- `subjects(subject PK, project FK, age, sex)`
- `subject_outcomes(subject PK/FK, treatment_id FK, response)`
- `samples(sample PK, subject FK, sample_type, time_from_treatment_start)`
- `populations(population PK)`
- `cell_counts(sample FK, population FK, count, PRIMARY KEY(sample, population))`

### Rationale and Scaling

The database is designed using Third Normal Form (3NF) principles to reduce data redundancy and enforce referential integrity. A key benefit of this design is validation. For example, a new subject cannot be ingested unless a valid project already exists in the system. Because attributes are isolated in specific tables, clinical or demographic updates (such as a change in treatment response) only need to be performed once. Cell counts are stored in long format, so adding a new immune population requires inserting a new row in `populations` and corresponding `cell_counts` records rather than changing the table schema.

While 3NF is ideal for data integrity, the level of normalization may need to be tuned based on the analytical workload. At very large scale, joining several large tables can impact performance. In such cases, selective denormalization can help. For example, if treatment effects are frequently analyzed across demographics, pre-joining `subjects` and `subject_outcomes` can reduce join overhead when attributes like age or sex are repeatedly required for downstream modeling.

## Code Structure Overview

- `load_data.py`
  - Creates the schema.
  - Loads and normalizes source data from `cell-count.csv` into SQLite using the `polars` package.
- `analysis.ipynb`
  - Initial notebook used to run the analysis, later used as the baseline for the dashboard in `app.py`.
  - The analysis reads only the required slices from `cell-count-3nf.db` via SQL (instead of loading all tables at once), which keeps memory usage lower and scales better as dataset size grows. Subsequent joins, reshaping, and filtering are then performed in `polars` for fast columnar transformations.
- `app.py`
  - Builds the dashboard directly from the analysis flow first developed in `analysis.ipynb`.
  - Keeps the dashboard data-driven by reading directly from SQLite at runtime, so database updates are reflected automatically without code changes.
  - Displays the required sample-population frequency table, responder/non-responder statistical comparison, baseline subset summaries, and the Part 5 average B-cell result.
  - Defaults Part 3 to baseline PBMC samples for response prediction and reports Welch tests, FDR/Bonferroni correction, effect sizes, confidence intervals, and Mann-Whitney sensitivity p-values.
  - For larger or computationally expensive analyses, a better approach is to precompute and store snapshot tables, then have the dashboard read those outputs for faster and more stable performance.
