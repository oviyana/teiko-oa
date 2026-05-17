#!/usr/bin/env python3
"""Interactive dashboard for Bob Loblaw's immune cell analysis assessment."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cell-count-3nf.db"
CELL_COLS = ["b_cell", "cd4_t_cell", "cd8_t_cell", "nk_cell", "monocyte"]


def to_display_text(value: str) -> str:
    return value.replace("_", " ").title()


def display_columns(df):
    out = df.copy()
    out.columns = [to_display_text(str(c)) for c in out.columns]
    return out


def bonferroni_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    return np.minimum(p * n, 1.0)


def highlight_significant_rows(row):
    flag = row.get("significant_bonferroni_lt_0_05", row.get("Significant Bonferroni Lt 0 05", False))
    if bool(flag):
        return ["background-color: #153f2c; color: #eafaf1"] * len(row)
    return [""] * len(row)


@st.cache_data
def load_dataframes(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        samples_df = pl.read_database(query="SELECT * FROM samples", connection=conn)

        treatments_df = pl.read_database(
            query="""
                SELECT treatment_id
                FROM treatments
                WHERE treatment = 'miraclib' AND condition = 'melanoma'
            """,
            connection=conn,
        )
        treatment_id = int(treatments_df["treatment_id"][0])

        outcomes_filtered = pl.read_database(
            query=f"""
                SELECT *
                FROM subject_outcomes
                WHERE treatment_id = {treatment_id}
            """,
            connection=conn,
        )

        subjects_df = pl.read_database(
            query="SELECT subject, project, sex FROM subjects",
            connection=conn,
        )

    return samples_df, outcomes_filtered, subjects_df


def main() -> None:
    st.set_page_config(page_title="Cell Count Analysis Dashboard", layout="wide")
    st.title("Cell Count Analysis Dashboard")

    if not DB_PATH.exists():
        st.error("No supported database found in repository root.")
        st.info("Expected file: `cell-count-3nf.db` in repository root.")
        st.stop()

    samples_df, outcomes_filtered, subjects_df = load_dataframes(DB_PATH)

    freq_df = (
        samples_df
        .with_columns(total_count=pl.sum_horizontal(CELL_COLS))
        .unpivot(
            index=["sample", "subject", "sample_type", "time_from_treatment_start", "total_count"],
            on=CELL_COLS,
            variable_name="population",
            value_name="count",
        )
        .with_columns(percentage=(pl.col("count") / pl.col("total_count")) * 100)
    )

    st.header("Part 2: Initial Analysis - Data Overview")
    st.markdown(
        "<div style='font-size:1.05rem; font-weight:600; margin-bottom:0.6rem;'>"
        "Each frequency is computed as <code>(count / total count) * 100</code> "
        "for every sample-population pair."
        "</div>",
        unsafe_allow_html=True,
    )

    required_table = freq_df.select(["sample", "total_count", "population", "count", "percentage"])

    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.subheader("Percentage Summary by Population")
        pct_summary = (
            required_table
            .group_by("population")
            .agg(
                pl.col("percentage").count().alias("count"),
                pl.col("percentage").mean().alias("mean"),
                pl.col("percentage").std().alias("std"),
                pl.col("percentage").min().alias("min"),
                pl.col("percentage").quantile(0.25).alias("25%"),
                pl.col("percentage").median().alias("50%"),
                pl.col("percentage").quantile(0.75).alias("75%"),
                pl.col("percentage").max().alias("max"),
            )
            .with_columns(pl.col("population").map_elements(to_display_text, return_dtype=pl.String))
            .sort("population")
        )
        st.dataframe(display_columns(pct_summary.to_pandas()), use_container_width=True)

    with stats_col2:
        st.subheader("Dataset Totals")
        st.metric("Unique samples", f"{required_table.select(pl.col('sample').n_unique()).item():,}")
        st.metric("Unique subjects", f"{samples_df.select(pl.col('subject').n_unique()).item():,}")

    st.header("Part 3: Statistical Analysis")
    st.markdown(
        "<div style='font-size:1.05rem; font-weight:600; margin-bottom:0.6rem;'>"
        "Assessment cohort: melanoma patients treated with miraclib, using PBMC samples."
        "</div>",
        unsafe_allow_html=True,
    )

    PBMC_merged_df = (
        freq_df
        .join(outcomes_filtered.select(["subject", "response"]), on="subject", how="inner")
        .filter((pl.col("sample_type") == "PBMC") & (pl.col("response").is_in(["yes", "no"])))
    )

    time_options = sorted(PBMC_merged_df["time_from_treatment_start"].unique().to_list())
    pop_options = [c for c in CELL_COLS if c in PBMC_merged_df["population"].unique().to_list()]

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        selected_timepoints = st.multiselect("Timepoint(s)", options=time_options, default=time_options)
    with control_col2:
        pop_display = {p: to_display_text(p) for p in pop_options}
        selected_populations = st.multiselect(
            "Cell populations",
            options=pop_options,
            default=pop_options,
            format_func=lambda x: pop_display.get(x, x),
        )

    analysis_df = PBMC_merged_df
    if selected_timepoints:
        analysis_df = analysis_df.filter(pl.col("time_from_treatment_start").is_in(selected_timepoints))
    analysis_df = analysis_df.filter(pl.col("population").is_in(selected_populations))

    if analysis_df.height == 0:
        st.warning("No rows found for selected Part 3 filters.")
    else:
        plot_df = analysis_df.with_columns(
            pl.col("population").map_elements(to_display_text, return_dtype=pl.String).alias("population_label")
        ).to_pandas()

        fig = px.box(
            plot_df,
            x="population_label",
            y="percentage",
            color="response",
            category_orders={"population_label": [to_display_text(p) for p in selected_populations], "response": ["no", "yes"]},
            color_discrete_map={"no": "#8ac7ff", "yes": "#008dff"},
            labels={"population_label": "Population", "percentage": "Percentage", "response": "Response"},
            title="Immune Cell Frequencies: Responders vs Non-Responders",
            points="outliers",
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#030a17",
            plot_bgcolor="#030a17",
            legend_title_text="Response",
            xaxis_title="Population",
            yaxis_title="Percentage",
        )
        st.plotly_chart(fig, use_container_width=True)

        results = []
        for cell in selected_populations:
            responders = analysis_df.filter((pl.col("population") == cell) & (pl.col("response") == "yes"))["percentage"]
            non_responders = analysis_df.filter((pl.col("population") == cell) & (pl.col("response") == "no"))["percentage"]

            if responders.len() == 0 or non_responders.len() == 0:
                t_stat, p_val = np.nan, np.nan
            else:
                t_stat, p_val = stats.ttest_ind(
                    responders.to_numpy(),
                    non_responders.to_numpy(),
                    equal_var=False,
                    nan_policy="omit",
                )

            mean_resp = responders.mean()
            mean_non_resp = non_responders.mean()

            results.append(
                {
                    "population": to_display_text(cell),
                    "n_responders": int(responders.len()),
                    "n_non_responders": int(non_responders.len()),
                    "mean_responder_pct": round(float(mean_resp), 2) if mean_resp is not None else np.nan,
                    "mean_non_responder_pct": round(float(mean_non_resp), 2) if mean_non_resp is not None else np.nan,
                    "difference": round(float(mean_resp - mean_non_resp), 2) if mean_resp is not None and mean_non_resp is not None else np.nan,
                    "t_stat": round(float(t_stat), 4) if not np.isnan(t_stat) else np.nan,
                    "p_value": float(p_val) if not np.isnan(p_val) else np.nan,
                }
            )

        ttest_df = pl.DataFrame(results).to_pandas()
        ttest_df["bonferroni_p_value"] = bonferroni_adjust(ttest_df["p_value"].fillna(1.0).to_numpy())
        ttest_df["significant_p_lt_0_05"] = ttest_df["p_value"] < 0.05
        ttest_df["significant_bonferroni_lt_0_05"] = ttest_df["bonferroni_p_value"] < 0.05
        ttest_df = ttest_df.sort_values("p_value", na_position="last")

        st.subheader("Welch's T-test Results")
        ttest_display = display_columns(ttest_df)
        styled_ttest = ttest_display.style.apply(highlight_significant_rows, axis=1)
        st.dataframe(styled_ttest, use_container_width=True)

        st.markdown(
            "<div style='font-size:1.02rem; margin-top:0.4rem;'>"
            "Statistical analysis: Welch's two-sample t-test on percentage frequencies "
            "(responders vs non-responders), with Bonferroni correction across selected populations. "
            "Rows highlighted in green are Bonferroni-significant at 0.05."
            "</div>",
            unsafe_allow_html=True,
        )

    st.header("Part 4: Data Subset Analysis")
    st.markdown(
        "<div style='font-size:1.05rem; font-weight:600; margin-bottom:0.6rem;'>"
        "Subset criteria: melanoma patients treated with miraclib, PBMC samples, "
        "and baseline only (time from treatment start = 0)."
        "</div>",
        unsafe_allow_html=True,
    )

    merged_samples_df = (
        samples_df
        .filter((pl.col("sample_type") == "PBMC") & (pl.col("time_from_treatment_start") == 0))
        .join(outcomes_filtered.select(["subject", "response"]), on="subject", how="inner")
        .join(subjects_df, on="subject", how="inner")
        .filter(pl.col("response").is_in(["yes", "no"]))
        .select(["sample", "subject", "project", "response", "sex"])
        .unique()
    )

    samples_by_project = merged_samples_df.group_by("project").agg(pl.col("sample").n_unique().alias("sample_count")).sort("project")
    if "prj2" not in samples_by_project["project"].to_list():
        sample_count_dtype = samples_by_project.schema["sample_count"]
        prj2_row = pl.DataFrame({"project": ["prj2"], "sample_count": [0]}).with_columns(
            pl.col("sample_count").cast(sample_count_dtype)
        )
        samples_by_project = pl.concat([samples_by_project, prj2_row]).sort("project")

    subjects_by_response = (
        merged_samples_df
        .select(["subject", "response"])
        .unique()
        .group_by("response")
        .agg(pl.col("subject").n_unique().alias("subject_count"))
        .sort("response")
    )

    subjects_by_sex = (
        merged_samples_df
        .select(["subject", "sex"])
        .unique()
        .group_by("sex")
        .agg(pl.col("subject").n_unique().alias("subject_count"))
        .sort("sex")
    )

    p4_col1, p4_col2, p4_col3 = st.columns(3)
    with p4_col1:
        st.subheader("Samples per Project")
        st.dataframe(display_columns(samples_by_project.to_pandas()), use_container_width=True)
    with p4_col2:
        st.subheader("Subjects by Response")
        st.dataframe(display_columns(subjects_by_response.to_pandas()), use_container_width=True)
    with p4_col3:
        st.subheader("Subjects by Sex")
        st.dataframe(display_columns(subjects_by_sex.to_pandas()), use_container_width=True)


if __name__ == "__main__":
    main()
