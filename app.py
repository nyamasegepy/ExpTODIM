from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import matplotlib.pyplot as plt

from todim_family_engine import (
    CASE_FILES,
    METHOD_ORDER,
    NORMALIZATION_ORDER,
    TODIMData,
    compare_methods,
    load_builtin_case,
    method_note,
    normalize_weights,
    result_tables,
    run_todim_family,
    sensitivity_analysis,
    summarize_sensitivity,
)


st.set_page_config(
    page_title="TODIM Family Robustness Dashboard",
    page_icon="📊",
    layout="wide",
)

METHOD_LABELS = {
    "classical_todim": "Classical TODIM",
    "generalized_inverse": "Generalized TODIM - inverse-weight loss",
    "generalized_monotone": "Generalized Monotone / Power TODIM",
    "log_todim": "Logarithmic TODIM",
    "exptodim": "ExpTODIM",
}
NORMALIZATION_LABELS = {
    "max": "Max normalization",
    "max_min": "Max-min normalization",
    "sum": "Sum normalization",
    "vector": "Vector normalization",
}
SENSE_TO_INT = {"Benefit": 1, "Cost": 0}
INT_TO_SENSE = {1: "Benefit", 0: "Cost"}
DATA_DIR = PROJECT_DIR / "data"


def read_uploaded_csv(uploaded_file) -> TODIMData:
    frame = pd.read_csv(uploaded_file)
    if frame.shape[0] < 2 or frame.shape[1] < 3:
        raise ValueError("CSV must contain at least two alternatives and two criteria")
    alternatives = frame.iloc[:, 0].astype(str).tolist()
    criteria = [str(column) for column in frame.columns[1:]]
    matrix = frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    return TODIMData(
        matrix=matrix,
        alternatives=alternatives,
        criteria=criteria,
        weights=np.ones(len(criteria), dtype=float),
        senses=np.ones(len(criteria), dtype=int),
    )


def parse_grid(value: str, name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} grid contains a non-numeric value") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} grid must contain positive values")
    return values


def data_signature(data: TODIMData) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(data.matrix).tobytes())
    digest.update(np.ascontiguousarray(data.weights).tobytes())
    digest.update(np.ascontiguousarray(data.senses).tobytes())
    digest.update("|".join(data.alternatives + data.criteria).encode("utf-8"))
    return digest.hexdigest()[:12]


def download_csv(frame: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def ranking_bar(scores: pd.DataFrame):
    ordered = scores.sort_values("Rank")
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    colors = ["#177E89" if rank == 1 else "#6C8EAD" for rank in ordered["Rank"]]
    ax.bar(ordered["Alternative"].astype(str), ordered["Normalized score"], color=colors)
    ax.set_ylabel("Normalized global score")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def sensitivity_heatmap(
    results: pd.DataFrame,
    method: str,
    normalization: str,
    value_column: str,
    beta: float | None = None,
):
    subset = results[
        (results["method"] == method) & (results["normalization"] == normalization)
    ].copy()
    if method == "classical_todim":
        series = subset.groupby("theta", sort=True)[value_column].mean()
        pivot = pd.DataFrame([series.to_numpy()], index=[""], columns=series.index)
        x_label, y_label = "theta", ""
    elif method in {"log_todim", "exptodim"}:
        pivot = subset.pivot_table(
            index="rho", columns="lambda", values=value_column, aggfunc="mean"
        )
        x_label, y_label = "lambda", "rho"
    else:
        if beta is not None:
            subset = subset[np.isclose(subset["beta"], beta)]
        pivot = subset.pivot_table(
            index="alpha", columns="lambda", values=value_column, aggfunc="mean"
        )
        x_label, y_label = "lambda", "alpha"

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    image = ax.imshow(pivot.values, aspect="auto", origin="lower", vmin=-1, vmax=1, cmap="RdYlGn")
    format_tick = lambda value: f"{value:g}" if isinstance(value, (int, float, np.number)) else str(value)
    ax.set_xticks(range(len(pivot.columns)), [format_tick(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [format_tick(value) for value in pivot.index])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{value_column.replace('_', ' ').title()} vs. ExpTODIM baseline", pad=14)
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iat[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return fig


st.title("TODIM Family Robustness Dashboard")
st.caption("Five crisp dominance functions with scenario-based ranking robustness analysis")

with st.sidebar:
    st.header("Decision data")
    source = st.radio("Source", ("Built-in case", "Upload CSV"), horizontal=True)
    case_name = "Uploaded matrix"
    if source == "Built-in case":
        case_options = ("Travel-destination case", *CASE_FILES.keys())
        case_name = st.selectbox("Case", case_options)
        base_data = load_builtin_case(case_name, DATA_DIR)
    else:
        uploaded = st.file_uploader("Decision matrix", type=["csv"])
        if uploaded is None:
            st.info("Select a CSV decision matrix.")
            st.stop()
        try:
            base_data = read_uploaded_csv(uploaded)
        except Exception as exc:
            st.error(f"Invalid CSV: {exc}")
            st.stop()

    st.header("Current configuration")
    method_keys = list(METHOD_ORDER)
    method = st.selectbox(
        "Dominance function",
        method_keys,
        index=method_keys.index("exptodim"),
        format_func=METHOD_LABELS.get,
    )
    normalization = st.selectbox(
        "Normalization",
        list(NORMALIZATION_ORDER),
        index=list(NORMALIZATION_ORDER).index("sum"),
        format_func=NORMALIZATION_LABELS.get,
    )

    theta, lambda_, rho, alpha, beta = 1.0, 2.25, 3.0, 0.88, 0.88
    if method == "classical_todim":
        theta = st.slider("Loss attenuation theta", 0.25, 10.0, 1.0, 0.25)
    elif method in {"log_todim", "exptodim"}:
        lambda_ = st.slider("Loss aversion lambda", 0.50, 6.0, 2.25, 0.25)
        rho_label = "Logarithmic sensitivity rho" if method == "log_todim" else "Exponential sensitivity rho"
        rho = st.slider(rho_label, 0.50, 10.0, 3.0, 0.50)
    else:
        lambda_ = st.slider("Loss aversion lambda", 0.50, 6.0, 2.25, 0.25)
        alpha = st.slider("Gain curvature alpha", 0.10, 2.0, 0.88, 0.01)
        beta = st.slider("Loss curvature beta", 0.10, 2.0, 0.88, 0.01)

settings = pd.DataFrame(
    {
        "Criterion": base_data.criteria,
        "Weight": base_data.weights,
        "Sense": [INT_TO_SENSE[int(value)] for value in base_data.senses],
    }
)

try:
    edited_settings = st.data_editor(
        settings,
        hide_index=True,
        width="stretch",
        disabled=["Criterion"],
        column_config={
            "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, format="%.6f"),
            "Sense": st.column_config.SelectboxColumn("Sense", options=["Benefit", "Cost"], required=True),
        },
        key=f"criterion_settings_{case_name}",
    )
    data = TODIMData(
        matrix=base_data.matrix,
        alternatives=base_data.alternatives,
        criteria=base_data.criteria,
        weights=normalize_weights(edited_settings["Weight"].to_numpy(dtype=float)),
        senses=np.array([SENSE_TO_INT[value] for value in edited_settings["Sense"]], dtype=int),
    )
except Exception as exc:
    st.error(f"Invalid criterion settings: {exc}")
    st.stop()

try:
    current_result = run_todim_family(
        data, method, normalization, theta, lambda_, rho, alpha, beta
    )
    current_tables = result_tables(data, current_result)
except Exception as exc:
    st.error(f"Current scenario failed: {exc}")
    st.stop()


single_tab, comparison_tab, sensitivity_tab, data_tab, notes_tab = st.tabs(
    ["Current ranking", "Method comparison", "Robustness analysis", "Decision data", "Definitions"]
)

with single_tab:
    top = data.alternatives[int(current_result["ranking"][0])]
    ranking_text = " > ".join(data.alternatives[int(index)] for index in current_result["ranking"])
    metric_columns = st.columns([1.4, 1.1, 1.0, 2.1])
    metric_columns[0].metric("Method", METHOD_LABELS[method])
    metric_columns[1].metric("Normalization", NORMALIZATION_LABELS[normalization])
    metric_columns[2].metric("Top alternative", top)
    metric_columns[3].metric("Ranking", ranking_text)
    st.info(method_note(method))

    left, right = st.columns([1.15, 1.0])
    with left:
        st.subheader("Scores and ranking")
        st.dataframe(current_tables["scores"], width="stretch", hide_index=True)
        download_csv(current_tables["scores"], "current_scores.csv", "Download scores")
    with right:
        st.pyplot(ranking_bar(current_tables["scores"]), width="stretch")

    with st.expander("Normalized matrix"):
        st.dataframe(current_tables["normalized"].round(6), width="stretch")
    with st.expander("Dominance matrix"):
        st.caption("Positive: row alternative dominates column alternative. Negative: row alternative is dominated.")
        st.dataframe(current_tables["dominance"].round(6), width="stretch")

with comparison_tab:
    st.subheader("Five-method comparison")
    reference_method = st.selectbox(
        "Reference method",
        method_keys,
        index=method_keys.index("exptodim"),
        format_func=METHOD_LABELS.get,
        key="comparison_reference",
    )
    comparison = compare_methods(
        data,
        normalization=normalization,
        reference_method=reference_method,
        theta=theta,
        lambda_=lambda_,
        rho=rho,
        alpha=alpha,
        beta=beta,
    )
    comparison.insert(1, "method_label", comparison["method"].map(METHOD_LABELS))
    display_comparison = comparison[
        ["method_label", "top_alternative", "ranking", "spearman_vs_reference", "kendall_tau_vs_reference"]
    ].rename(
        columns={
            "method_label": "Method",
            "top_alternative": "Top alternative",
            "ranking": "Ranking",
            "spearman_vs_reference": "Spearman vs. reference",
            "kendall_tau_vs_reference": "Kendall tau vs. reference",
        }
    )
    st.dataframe(display_comparison, width="stretch", hide_index=True)
    download_csv(comparison, "method_comparison.csv", "Download comparison")

with sensitivity_tab:
    st.subheader("Scenario-based robustness analysis")
    with st.form("sensitivity_grid_form"):
        methods = st.multiselect(
            "Methods",
            method_keys,
            default=method_keys,
            format_func=METHOD_LABELS.get,
        )
        normalizations = st.multiselect(
            "Normalizations",
            list(NORMALIZATION_ORDER),
            default=list(NORMALIZATION_ORDER),
            format_func=NORMALIZATION_LABELS.get,
        )
        grid_columns = st.columns(3)
        with grid_columns[0]:
            theta_grid = st.text_input("theta grid", "0.5, 1.0, 2.0, 5.0, 10.0")
            lambda_grid = st.text_input("lambda grid", "1.0, 1.5, 2.25, 3.0, 5.0")
        with grid_columns[1]:
            rho_grid = st.text_input("rho grid", "1.0, 2.0, 3.0, 5.0, 10.0")
            alpha_grid = st.text_input("alpha grid", "0.5, 0.88, 1.0")
        with grid_columns[2]:
            beta_grid = st.text_input("beta grid", "0.5, 0.88, 1.0")
            baseline_normalization = st.selectbox(
                "Baseline normalization",
                list(NORMALIZATION_ORDER),
                index=list(NORMALIZATION_ORDER).index("sum"),
                format_func=NORMALIZATION_LABELS.get,
            )
        run_grid = st.form_submit_button("Run robustness grid", type="primary")

    if run_grid:
        try:
            sensitivity = sensitivity_analysis(
                data,
                methods=tuple(methods),
                lambdas=parse_grid(lambda_grid, "lambda"),
                rhos=parse_grid(rho_grid, "rho"),
                thetas=parse_grid(theta_grid, "theta"),
                alphas=parse_grid(alpha_grid, "alpha"),
                betas=parse_grid(beta_grid, "beta"),
                normalizations=tuple(normalizations),
                baseline_method="exptodim",
                baseline_normalization=baseline_normalization,
                baseline_lambda=2.25,
                baseline_rho=3.0,
                baseline_theta=1.0,
                baseline_alpha=0.88,
                baseline_beta=0.88,
            )
            st.session_state["sensitivity_results"] = sensitivity
            st.session_state["sensitivity_context"] = {
                "case": case_name,
                "signature": data_signature(data),
                "baseline": f"ExpTODIM / {NORMALIZATION_LABELS[baseline_normalization]} / lambda=2.25 / rho=3.0",
            }
        except Exception as exc:
            st.error(f"Sensitivity grid failed: {exc}")

    sensitivity = st.session_state.get("sensitivity_results")
    context = st.session_state.get("sensitivity_context", {})
    if sensitivity is None:
        st.info("No robustness grid has been computed in this session.")
    elif context.get("signature") != data_signature(data):
        st.warning("Criterion data or settings changed after the stored grid was computed. Run the grid again.")
    else:
        summary = summarize_sensitivity(sensitivity)
        summary.insert(1, "method_label", summary["method"].map(METHOD_LABELS))
        metric_columns = st.columns(4)
        metric_columns[0].metric("Scenarios", f"{len(sensitivity):,}")
        metric_columns[1].metric("Methods", sensitivity["method"].nunique())
        metric_columns[2].metric("Top-one changes", int((~sensitivity["top_stable"]).sum()))
        metric_columns[3].metric("Changed rankings", int(sensitivity["changed_ranking"].sum()))
        st.caption(f"Reference: {context.get('baseline', 'ExpTODIM baseline')} | Case: {context.get('case', case_name)}")

        display_summary = summary[
            [
                "method_label",
                "scenarios",
                "top_one_stability",
                "full_ranking_stability",
                "mean_spearman",
                "min_spearman",
                "mean_kendall",
                "changed_rankings",
            ]
        ].rename(
            columns={
                "method_label": "Method",
                "scenarios": "Scenarios",
                "top_one_stability": "TOS",
                "full_ranking_stability": "FRS",
                "mean_spearman": "Mean Spearman",
                "min_spearman": "Min Spearman",
                "mean_kendall": "Mean Kendall",
                "changed_rankings": "Changed rankings",
            }
        )
        st.dataframe(
            display_summary.style.format(
                {
                    "TOS": "{:.3f}",
                    "FRS": "{:.3f}",
                    "Mean Spearman": "{:.3f}",
                    "Min Spearman": "{:.3f}",
                    "Mean Kendall": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        normalization_summary = (
            sensitivity.groupby(["method", "normalization"], sort=False)
            .agg(
                scenarios=("method", "size"),
                TOS=("top_stable", "mean"),
                FRS=("full_ranking_stable", "mean"),
                mean_spearman=("spearman", "mean"),
                mean_kendall=("kendall_tau", "mean"),
            )
            .reset_index()
        )
        normalization_summary["Method"] = normalization_summary["method"].map(METHOD_LABELS)
        normalization_summary["Normalization"] = normalization_summary["normalization"].map(NORMALIZATION_LABELS)
        with st.expander("Results by normalization"):
            st.dataframe(
                normalization_summary[
                    ["Method", "Normalization", "scenarios", "TOS", "FRS", "mean_spearman", "mean_kendall"]
                ],
                width="stretch",
                hide_index=True,
            )

        heatmap_columns = st.columns(4)
        with heatmap_columns[0]:
            heatmap_method = st.selectbox(
                "Heatmap method",
                sensitivity["method"].drop_duplicates().tolist(),
                format_func=METHOD_LABELS.get,
            )
        with heatmap_columns[1]:
            heatmap_normalization = st.selectbox(
                "Heatmap normalization",
                sensitivity["normalization"].drop_duplicates().tolist(),
                format_func=NORMALIZATION_LABELS.get,
            )
        with heatmap_columns[2]:
            heatmap_indicator = st.selectbox(
                "Indicator", ["spearman", "kendall_tau", "top_stable", "full_ranking_stable"]
            )
        selected_beta = None
        with heatmap_columns[3]:
            if heatmap_method in {"generalized_inverse", "generalized_monotone"}:
                beta_options = sorted(sensitivity.loc[sensitivity["method"] == heatmap_method, "beta"].unique())
                selected_beta = st.selectbox("beta slice", beta_options)
            else:
                st.empty()
        st.pyplot(
            sensitivity_heatmap(
                sensitivity,
                heatmap_method,
                heatmap_normalization,
                heatmap_indicator,
                selected_beta,
            ),
            width="stretch",
        )

        with st.expander("Scenario-level results"):
            st.dataframe(sensitivity, width="stretch", hide_index=True)
        download_csv(sensitivity, "todim_family_sensitivity_results.csv", "Download scenarios")
        download_csv(summary, "todim_family_robustness_summary.csv", "Download summary")

with data_tab:
    st.subheader(case_name)
    matrix_frame = pd.DataFrame(data.matrix, index=data.alternatives, columns=data.criteria)
    st.dataframe(matrix_frame, width="stretch")
    criterion_frame = pd.DataFrame(
        {
            "Criterion": data.criteria,
            "Weight": data.weights,
            "Sense": [INT_TO_SENSE[int(value)] for value in data.senses],
        }
    )
    st.dataframe(criterion_frame, width="stretch", hide_index=True)
    export_frame = matrix_frame.reset_index(names="Alternative")
    download_csv(export_frame, "decision_matrix.csv", "Download decision matrix")

with notes_tab:
    st.subheader("Robustness indicators")
    indicator_frame = pd.DataFrame(
        [
            ("TOS", "Share of scenarios preserving the baseline top alternative"),
            ("FRS", "Share of scenarios preserving the complete baseline ranking"),
            ("Spearman", "Agreement in alternative rank positions relative to the baseline"),
            ("Kendall tau", "Agreement in pairwise orderings relative to the baseline"),
            ("Changed rankings", "Number of scenarios whose complete ranking differs from the baseline"),
        ],
        columns=["Indicator", "Interpretation"],
    )
    st.dataframe(indicator_frame, width="stretch", hide_index=True)
    st.caption("Robustness is conditional on the selected deterministic scenario grid; grid levels are not probability distributions.")

    st.subheader("Dominance functions")
    st.markdown(
        r"""
**Classical TODIM**

Gain: $\sqrt{w_j |\delta|}$; loss: $-\theta^{-1}\sqrt{|\delta|/w_j}$.

**Generalized TODIM - inverse-weight loss**

Gain: $(w_j|\delta|)^\alpha$; loss: $-\lambda(|\delta|/w_j)^\alpha$.

**Generalized Monotone / Power TODIM**

Gain: $w_j|\delta|^\alpha$; loss: $-\lambda w_j|\delta|^\beta$.

**Logarithmic TODIM**

Gain: $w_j\log(1+10\rho|\delta|)$; loss: $-\lambda w_j\log(1+10\rho|\delta|)$.

**ExpTODIM**

Gain: $w_j(1-10^{-\rho|\delta|})$; loss: $-\lambda w_j(1-10^{-\rho|\delta|})$.
"""
    )
