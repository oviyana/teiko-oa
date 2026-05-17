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


def csv_bytes(df: pl.DataFrame) -> bytes:
    return df.write_csv().encode("utf-8")


def bonferroni_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    return np.minimum(p * n, 1.0)


def fdr_bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return adjusted

    valid_p = p[valid]
    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    ranks = np.arange(1, len(ranked_p) + 1)
    ranked_adjusted = ranked_p * len(ranked_p) / ranks
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.minimum(ranked_adjusted, 1.0)

    valid_positions = np.where(valid)[0]
    adjusted[valid_positions[order]] = ranked_adjusted
    return adjusted


def cohen_d(group_a, group_b):
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_var = (((len(a) - 1) * np.var(a, ddof=1)) + ((len(b) - 1) * np.var(b, ddof=1))) / (len(a) + len(b) - 2)
    if pooled_var <= 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / np.sqrt(pooled_var)


def welch_mean_diff_ci(group_a, group_b, confidence=0.95):
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    se = np.sqrt((var_a / len(a)) + (var_b / len(b)))
    diff = np.mean(a) - np.mean(b)
    if se == 0:
        return diff, diff

    numerator = ((var_a / len(a)) + (var_b / len(b))) ** 2
    denominator = ((var_a / len(a)) ** 2 / (len(a) - 1)) + ((var_b / len(b)) ** 2 / (len(b) - 1))
    dof = numerator / denominator
    alpha = 1 - confidence
    margin = stats.t.ppf(1 - alpha / 2, dof) * se
    return diff - margin, diff + margin


def highlight_significant_rows(row):
    flag = row.get(
        "significant_fdr_lt_0_05",
        row.get("Significant Fdr Lt 0 05", row.get("significant_bonferroni_lt_0_05", False)),
    )
    if bool(flag):
        return ["background-color: #153f2c; color: #eafaf1"] * len(row)
    return [""] * len(row)


@st.cache_data
def load_dataframes(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        samples_df = pl.read_database(query="SELECT * FROM samples", connection=conn)

        cell_counts_df = pl.read_database(query="SELECT sample, population, count FROM cell_counts", connection=conn)

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

        projects_df = pl.read_database(
            query="SELECT project FROM projects ORDER BY project",
            connection=conn,
        )

        part5_df = pl.read_database(
            query="""
                SELECT ROUND(AVG(cc.count), 2) AS avg_b_cell
                FROM samples samp
                JOIN cell_counts cc ON samp.sample = cc.sample
                JOIN subjects sub ON samp.subject = sub.subject
                JOIN subject_outcomes out ON sub.subject = out.subject
                JOIN treatments t ON out.treatment_id = t.treatment_id
                WHERE t.condition = 'melanoma'
                    AND cc.population = 'b_cell'
                    AND sub.sex = 'M'
                    AND out.response = 'yes'
                    AND samp.time_from_treatment_start = 0
            """,
            connection=conn,
        )

    return samples_df, cell_counts_df, outcomes_filtered, subjects_df, projects_df, float(part5_df["avg_b_cell"][0])


def main() -> None:
    st.set_page_config(page_title="Cell Count Analysis Dashboard", layout="wide")
    st.title("Cell Count Analysis Dashboard")

    if not DB_PATH.exists():
        st.error("No supported database found in repository root.")
        st.info("Expected file: `cell-count-3nf.db` in repository root.")
        st.stop()

    samples_df, cell_counts_df, outcomes_filtered, subjects_df, projects_df, part5_avg_b_cell = load_dataframes(DB_PATH)

    total_counts_df = cell_counts_df.group_by("sample").agg(pl.col("count").sum().alias("total_count"))
    freq_df = (
        samples_df
        .join(total_counts_df, on="sample", how="inner")
        .join(cell_counts_df, on="sample", how="inner")
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

    pct_summary = (
        required_table
        .group_by("population")
        .agg(
            pl.col("percentage").count().alias("rows"),
            pl.col("percentage").mean().round(2).alias("mean_pct"),
            pl.col("percentage").median().round(2).alias("median_pct"),
            pl.col("percentage").quantile(0.25).round(2).alias("q1_pct"),
            pl.col("percentage").quantile(0.75).round(2).alias("q3_pct"),
        )
        .with_columns(pl.col("population").str.replace_all("_", " ", literal=True).str.to_titlecase())
        .sort("population")
    )

    overview_cols = st.columns(4)
    overview_cols[0].metric("Samples", f"{required_table.select(pl.col('sample').n_unique()).item():,}")
    overview_cols[1].metric("Subjects", f"{samples_df.select(pl.col('subject').n_unique()).item():,}")
    overview_cols[2].metric("Frequency rows", f"{required_table.height:,}")
    overview_cols[3].metric(
        "Mean total cells",
        f"{required_table.select(pl.col('total_count').mean()).item():,.0f}",
    )

    summary_col, lookup_col = st.columns([2, 1])
    with summary_col:
        st.subheader("Population Frequency Summary")
        st.dataframe(display_columns(pct_summary.to_pandas()), width="stretch", hide_index=True)
    with lookup_col:
        st.subheader("Sample Lookup")
        sample_options = samples_df.select("sample").sort("sample")["sample"].to_list()
        selected_sample = st.selectbox("Sample", sample_options, label_visibility="collapsed")
        sample_table = (
            required_table
            .filter(pl.col("sample") == selected_sample)
            .with_columns(pl.col("percentage").round(2))
            .sort("population")
        )
        st.dataframe(display_columns(sample_table.to_pandas()), width="stretch", hide_index=True)

    detail_col, download_col = st.columns([2, 1])
    with detail_col:
        with st.expander("Preview required row-level frequency table"):
            preview_table = required_table.with_columns(pl.col("percentage").round(2)).head(25)
            st.dataframe(display_columns(preview_table.to_pandas()), width="stretch", hide_index=True)
    with download_col:
        st.download_button(
            "Download full frequency table",
            data=csv_bytes(required_table),
            file_name="sample_population_frequencies.csv",
            mime="text/csv",
            width="stretch",
        )

    st.header("Part 3: Statistical Analysis")
    st.markdown(
        "<div style='font-size:1.05rem; font-weight:600; margin-bottom:0.6rem;'>"
        "Assessment cohort: melanoma patients treated with miraclib, using PBMC samples. "
        "Baseline is selected by default for response-prediction analysis; if multiple timepoints "
        "are selected, frequencies are averaged to one value per subject before testing."
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

    analysis_mode = st.radio(
        "Analysis set",
        ["Baseline only (prediction)", "All timepoints (exploratory)", "Custom timepoints"],
        horizontal=True,
    )

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        default_timepoints = [0] if 0 in time_options else time_options
        if analysis_mode == "Baseline only (prediction)":
            selected_timepoints = default_timepoints
            st.caption("Using baseline samples only.")
        elif analysis_mode == "All timepoints (exploratory)":
            selected_timepoints = time_options
            st.caption("Averaging selected timepoints to one value per subject.")
        else:
            selected_timepoints = st.multiselect("Timepoint(s)", options=time_options, default=default_timepoints)
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
        analysis_subject_df = (
            analysis_df
            .group_by(["subject", "response", "population"])
            .agg(pl.col("percentage").mean().alias("percentage"))
        )

        analysis_metrics = st.columns(3)
        analysis_metrics[0].metric("Subjects tested", f"{analysis_subject_df.select(pl.col('subject').n_unique()).item():,}")
        analysis_metrics[1].metric("Responder subjects", f"{analysis_subject_df.filter(pl.col('response') == 'yes').select(pl.col('subject').n_unique()).item():,}")
        analysis_metrics[2].metric("Non-responder subjects", f"{analysis_subject_df.filter(pl.col('response') == 'no').select(pl.col('subject').n_unique()).item():,}")

        plot_df = analysis_subject_df.with_columns(
            pl.col("population").str.replace_all("_", " ", literal=True).str.to_titlecase().alias("population_label")
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
        st.plotly_chart(fig, width="stretch")

        results = []
        for cell in selected_populations:
            responders = analysis_subject_df.filter((pl.col("population") == cell) & (pl.col("response") == "yes"))["percentage"]
            non_responders = analysis_subject_df.filter((pl.col("population") == cell) & (pl.col("response") == "no"))["percentage"]
            responder_values = responders.to_numpy()
            non_responder_values = non_responders.to_numpy()

            if responders.len() == 0 or non_responders.len() == 0:
                t_stat, p_val, mann_whitney_p = np.nan, np.nan, np.nan
                ci_low, ci_high = np.nan, np.nan
                effect_size = np.nan
            else:
                t_stat, p_val = stats.ttest_ind(
                    responder_values,
                    non_responder_values,
                    equal_var=False,
                    nan_policy="omit",
                )
                _, mann_whitney_p = stats.mannwhitneyu(responder_values, non_responder_values, alternative="two-sided")
                ci_low, ci_high = welch_mean_diff_ci(responder_values, non_responder_values)
                effect_size = cohen_d(responder_values, non_responder_values)

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
                    "ci_95_low": round(float(ci_low), 2) if not np.isnan(ci_low) else np.nan,
                    "ci_95_high": round(float(ci_high), 2) if not np.isnan(ci_high) else np.nan,
                    "cohens_d": round(float(effect_size), 3) if not np.isnan(effect_size) else np.nan,
                    "t_stat": round(float(t_stat), 4) if not np.isnan(t_stat) else np.nan,
                    "p_value": float(p_val) if not np.isnan(p_val) else np.nan,
                    "mann_whitney_p_value": float(mann_whitney_p) if not np.isnan(mann_whitney_p) else np.nan,
                }
            )

        ttest_df = pl.DataFrame(results).to_pandas()
        ttest_df["fdr_q_value"] = fdr_bh_adjust(ttest_df["p_value"].to_numpy())
        ttest_df["bonferroni_p_value"] = bonferroni_adjust(ttest_df["p_value"].fillna(1.0).to_numpy())
        ttest_df["significant_p_lt_0_05"] = ttest_df["p_value"] < 0.05
        ttest_df["significant_fdr_lt_0_05"] = ttest_df["fdr_q_value"] < 0.05
        ttest_df["significant_bonferroni_lt_0_05"] = ttest_df["bonferroni_p_value"] < 0.05
        ttest_df = ttest_df.sort_values("fdr_q_value", na_position="last")

        significant_populations = ttest_df.loc[ttest_df["significant_fdr_lt_0_05"], "population"].tolist()
        if significant_populations:
            st.success("FDR-significant populations: " + ", ".join(significant_populations))
        else:
            st.info("No selected populations are significant after FDR correction at 0.05.")

        st.subheader("Subject-Level Statistical Results")
        ttest_display = display_columns(ttest_df)
        styled_ttest = ttest_display.style.apply(highlight_significant_rows, axis=1)
        st.dataframe(styled_ttest, width="stretch")

        st.markdown(
            "<div style='font-size:1.02rem; margin-top:0.4rem;'>"
            "Statistical analysis: one subject-level percentage per population, Welch's two-sample t-test "
            "for responder vs non-responder means, Benjamini-Hochberg FDR correction, Bonferroni correction, "
            "95% Welch confidence intervals for the mean difference, Cohen's d effect size, and Mann-Whitney U "
            "as a non-parametric sensitivity check. Rows highlighted in green are FDR-significant at 0.05."
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

    samples_by_project = (
        projects_df
        .join(
            merged_samples_df.group_by("project").agg(pl.col("sample").n_unique().alias("sample_count")),
            on="project",
            how="left",
        )
        .with_columns(pl.col("sample_count").fill_null(0).cast(pl.Int64))
        .sort("project")
    )

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

    subset_metrics = st.columns(4)
    subset_metrics[0].metric("Baseline PBMC samples", f"{merged_samples_df.select(pl.col('sample').n_unique()).item():,}")
    subset_metrics[1].metric("Subjects", f"{merged_samples_df.select(pl.col('subject').n_unique()).item():,}")
    subset_metrics[2].metric("Responders", f"{subjects_by_response.filter(pl.col('response') == 'yes')['subject_count'][0]:,}")
    subset_metrics[3].metric("Non-responders", f"{subjects_by_response.filter(pl.col('response') == 'no')['subject_count'][0]:,}")

    p4_col1, p4_col2, p4_col3 = st.columns(3)
    with p4_col1:
        st.subheader("Samples per Project")
        st.dataframe(display_columns(samples_by_project.to_pandas()), width="stretch")
    with p4_col2:
        st.subheader("Subjects by Response")
        st.dataframe(display_columns(subjects_by_response.to_pandas()), width="stretch")
    with p4_col3:
        st.subheader("Subjects by Sex")
        st.dataframe(display_columns(subjects_by_sex.to_pandas()), width="stretch")

    sample_preview_col, sample_download_col = st.columns([2, 1])
    with sample_preview_col:
        st.subheader("Identified Sample Preview")
        project_choices = ["All"] + samples_by_project["project"].to_list()
        selected_project = st.selectbox("Project filter", project_choices, label_visibility="collapsed")
        sample_preview = merged_samples_df.sort(["project", "subject", "sample"])
        if selected_project != "All":
            sample_preview = sample_preview.filter(pl.col("project") == selected_project)
        st.dataframe(display_columns(sample_preview.head(30).to_pandas()), width="stretch", hide_index=True)
    with sample_download_col:
        st.download_button(
            "Download full baseline sample list",
            data=csv_bytes(merged_samples_df.sort(["project", "subject", "sample"])),
            file_name="melanoma_miraclib_baseline_pbmc_samples.csv",
            mime="text/csv",
            width="stretch",
        )

    st.header("Part 5: Average B Cells")
    st.markdown(
        "<div style='font-size:1.05rem; font-weight:600; margin-bottom:0.6rem;'>"
        "Criteria: melanoma, male, responder, and time from treatment start = 0."
        "</div>",
        unsafe_allow_html=True,
    )
    st.metric("Average B cells", f"{part5_avg_b_cell:.2f}")


if __name__ == "__main__":
    main()
