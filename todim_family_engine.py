from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence
import itertools

import numpy as np
import pandas as pd


NormalizationMethod = Literal["max", "max_min", "sum", "vector"]
TODIMMethod = Literal[
    "classical_todim",
    "generalized_inverse",
    "generalized_monotone",
    "log_todim",
    "exptodim",
]

METHOD_ORDER: tuple[TODIMMethod, ...] = (
    "classical_todim",
    "generalized_inverse",
    "generalized_monotone",
    "log_todim",
    "exptodim",
)
NORMALIZATION_ORDER: tuple[NormalizationMethod, ...] = (
    "max",
    "max_min",
    "sum",
    "vector",
)

CASE_FILES = {
    "Balanced trade-offs": "balanced_tradeoffs",
    "Loss-aversion sensitive": "loss_aversion_sensitive",
    "Close-score rho test": "rho_sensitive_close_scores",
    "Weight-sensitive case": "weights_sensitive",
    "Benefit/cost sense test": "senses_flip_test",
}


@dataclass(frozen=True)
class TODIMData:
    matrix: np.ndarray
    alternatives: list[str]
    criteria: list[str]
    weights: np.ndarray
    senses: np.ndarray  # 1 = benefit, 0 = cost

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        senses = np.asarray(self.senses, dtype=int)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
            raise ValueError("the decision matrix must contain at least two alternatives and two criteria")
        if matrix.shape != (len(self.alternatives), len(self.criteria)):
            raise ValueError("matrix dimensions must match alternative and criterion labels")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("matrix values must be finite")
        if weights.shape != (matrix.shape[1],) or senses.shape != (matrix.shape[1],):
            raise ValueError("weights and senses must match the number of criteria")
        if len(set(self.alternatives)) != len(self.alternatives):
            raise ValueError("alternative names must be unique")
        if len(set(self.criteria)) != len(self.criteria):
            raise ValueError("criterion names must be unique")
        if not np.all(np.isin(senses, (0, 1))):
            raise ValueError("criterion senses must be 0 (cost) or 1 (benefit)")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "weights", normalize_weights(weights))
        object.__setattr__(self, "senses", senses)


def built_in_example() -> TODIMData:
    matrix = np.array(
        [
            [5.0, 2.5, 4.0, 2840.0, 5.0, 3.0, 9.0, 8.0],
            [3.5, 12.0, 6.0, 3700.0, 9.0, 7.0, 3.0, 6.0],
            [2.5, 4.0, 5.0, 2683.0, 4.0, 5.0, 7.0, 7.5],
            [3.0, 13.0, 7.0, 4150.0, 6.0, 9.0, 6.0, 7.0],
            [4.0, 18.0, 9.0, 4500.0, 3.0, 8.0, 5.0, 4.0],
        ],
        dtype=float,
    )
    return TODIMData(
        matrix=matrix,
        alternatives=["A1", "A2", "A3", "A4", "A5"],
        criteria=[
            "Hotel Rating",
            "Time Traveling",
            "Days",
            "Cost",
            "Shopping",
            "Cultural Attractions",
            "Nature",
            "Safety",
        ],
        weights=np.array(
            [
                0.214732143,
                0.015625000,
                0.152232143,
                0.339732143,
                0.110565476,
                0.079315476,
                0.033482143,
                0.054315476,
            ]
        ),
        senses=np.array([1, 0, 1, 0, 1, 1, 1, 1]),
    )


def load_builtin_case(name: str, data_dir: Optional[Path] = None) -> TODIMData:
    if name == "Travel-destination case":
        return built_in_example()
    if name not in CASE_FILES:
        raise ValueError(f"unknown built-in case: {name}")
    root = data_dir or Path(__file__).resolve().parent / "data"
    stem = CASE_FILES[name]
    matrix_df = pd.read_csv(root / f"{stem}.csv")
    settings_df = pd.read_csv(root / f"{stem}_weights_senses.csv")
    criteria = matrix_df.columns[1:].tolist()
    settings_df = settings_df.set_index("Criterion").loc[criteria]
    return TODIMData(
        matrix=matrix_df.iloc[:, 1:].to_numpy(dtype=float),
        alternatives=matrix_df.iloc[:, 0].astype(str).tolist(),
        criteria=criteria,
        weights=settings_df["Weight"].to_numpy(dtype=float),
        senses=settings_df["Sense"].to_numpy(dtype=int),
    )


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("weights must be a finite one-dimensional array")
    if np.any(values < 0):
        raise ValueError("weights must be non-negative")
    total = values.sum()
    if total <= 0:
        raise ValueError("sum of weights must be positive")
    return values / total


def normalize_matrix(
    matrix: np.ndarray,
    senses: np.ndarray,
    method: NormalizationMethod = "sum",
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    senses = np.asarray(senses, dtype=int)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if senses.shape != (values.shape[1],):
        raise ValueError("senses length must equal number of criteria")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix values must be finite")

    normalized = np.empty_like(values, dtype=float)
    benefit = senses == 1
    cost = senses == 0

    if method == "max":
        maximum = values.max(axis=0)
        minimum = values.min(axis=0)
        if np.any(maximum[benefit] == 0):
            raise ZeroDivisionError("benefit criterion maximum cannot be zero")
        if np.any(values[:, cost] <= 0) or np.any(minimum[cost] <= 0):
            raise ZeroDivisionError("max normalization requires positive cost values")
        normalized[:, benefit] = values[:, benefit] / maximum[benefit]
        normalized[:, cost] = minimum[cost] / values[:, cost]
    elif method == "max_min":
        maximum = values.max(axis=0)
        minimum = values.min(axis=0)
        denominator = maximum - minimum
        constant = denominator == 0
        normalized[:, constant] = 1.0
        varying_benefit = benefit & ~constant
        varying_cost = cost & ~constant
        normalized[:, varying_benefit] = (
            values[:, varying_benefit] - minimum[varying_benefit]
        ) / denominator[varying_benefit]
        normalized[:, varying_cost] = (
            maximum[varying_cost] - values[:, varying_cost]
        ) / denominator[varying_cost]
    elif method == "sum":
        if np.any(benefit):
            denominator = values[:, benefit].sum(axis=0)
            if np.any(denominator == 0):
                raise ZeroDivisionError("benefit criterion sum cannot be zero")
            normalized[:, benefit] = values[:, benefit] / denominator
        if np.any(cost):
            if np.any(values[:, cost] <= 0):
                raise ZeroDivisionError("sum normalization requires positive cost values")
            inverse = 1.0 / values[:, cost]
            normalized[:, cost] = inverse / inverse.sum(axis=0)
    elif method == "vector":
        if np.any(benefit):
            denominator = np.linalg.norm(values[:, benefit], axis=0)
            if np.any(denominator == 0):
                raise ZeroDivisionError("benefit criterion norm cannot be zero")
            normalized[:, benefit] = values[:, benefit] / denominator
        if np.any(cost):
            if np.any(values[:, cost] <= 0):
                raise ZeroDivisionError("vector normalization requires positive cost values")
            inverse = 1.0 / values[:, cost]
            normalized[:, cost] = inverse / np.linalg.norm(inverse, axis=0)
    else:
        raise ValueError(f"unknown normalization method: {method}")
    return normalized


def _pow_nonnegative(values: np.ndarray, exponent: float) -> np.ndarray:
    return np.power(np.maximum(np.asarray(values, dtype=float), 0.0), exponent)


def compute_todim_family_dominance(
    normalized_matrix: np.ndarray,
    weights: np.ndarray,
    method: TODIMMethod = "exptodim",
    theta: float = 1.0,
    lambda_: float = 2.25,
    rho: float = 3.0,
    alpha: float = 0.88,
    beta: float = 0.88,
) -> np.ndarray:
    matrix = np.asarray(normalized_matrix, dtype=float)
    weights = normalize_weights(weights)
    if np.any(weights <= 0):
        raise ValueError("all criterion weights must be positive for TODIM dominance")
    if min(theta, lambda_, rho, alpha, beta) <= 0:
        raise ValueError("TODIM parameters must be positive")
    if method not in METHOD_ORDER:
        raise ValueError(f"unknown TODIM-family method: {method}")

    dominance = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float)
    ln10 = np.log(10.0)
    for criterion_index in range(matrix.shape[1]):
        difference = matrix[:, [criterion_index]] - matrix[:, [criterion_index]].T
        magnitude = np.abs(difference)
        gain = difference > 0
        loss = difference < 0
        weight = float(weights[criterion_index])
        contribution = np.zeros_like(difference)

        if method == "classical_todim":
            contribution[gain] = np.sqrt(weight * magnitude[gain])
            contribution[loss] = -(1.0 / theta) * np.sqrt(magnitude[loss] / weight)
        elif method == "generalized_inverse":
            contribution[gain] = _pow_nonnegative(weight * magnitude[gain], alpha)
            contribution[loss] = -lambda_ * _pow_nonnegative(magnitude[loss] / weight, alpha)
        elif method == "generalized_monotone":
            contribution[gain] = weight * _pow_nonnegative(magnitude[gain], alpha)
            contribution[loss] = -lambda_ * weight * _pow_nonnegative(magnitude[loss], beta)
        elif method == "log_todim":
            transformed = np.log1p(10.0 * rho * magnitude)
            contribution[gain] = weight * transformed[gain]
            contribution[loss] = -lambda_ * weight * transformed[loss]
        elif method == "exptodim":
            transformed = 1.0 - np.exp(-rho * magnitude * ln10)
            contribution[gain] = weight * transformed[gain]
            contribution[loss] = -lambda_ * weight * transformed[loss]
        dominance += contribution
    return dominance


def compute_scores(dominance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(dominance, dtype=float).sum(axis=1)
    minimum, maximum = raw.min(), raw.max()
    normalized = np.ones_like(raw) if maximum == minimum else (raw - minimum) / (maximum - minimum)
    return raw, normalized


def rank_alternatives(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")


def ranking_positions(ranking: np.ndarray, n_alternatives: Optional[int] = None) -> np.ndarray:
    ranking = np.asarray(ranking, dtype=int)
    count = len(ranking) if n_alternatives is None else n_alternatives
    positions = np.empty(count, dtype=int)
    for position, index in enumerate(ranking, start=1):
        positions[index] = position
    return positions


def spearman_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    denominator = np.sqrt((centered_left**2).sum() * (centered_right**2).sum())
    return 1.0 if denominator == 0 else float((centered_left * centered_right).sum() / denominator)


def kendall_tau(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    concordant = discordant = 0
    for first, second in itertools.combinations(range(len(left)), 2):
        product = np.sign(left[first] - left[second]) * np.sign(right[first] - right[second])
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
    total = concordant + discordant
    return 1.0 if total == 0 else float((concordant - discordant) / total)


def run_todim_family(
    data: TODIMData,
    method: TODIMMethod = "exptodim",
    normalization: NormalizationMethod = "sum",
    theta: float = 1.0,
    lambda_: float = 2.25,
    rho: float = 3.0,
    alpha: float = 0.88,
    beta: float = 0.88,
) -> dict:
    normalized_matrix = normalize_matrix(data.matrix, data.senses, normalization)
    dominance_matrix = compute_todim_family_dominance(
        normalized_matrix, data.weights, method, theta, lambda_, rho, alpha, beta
    )
    raw_scores, normalized_scores = compute_scores(dominance_matrix)
    ranking = rank_alternatives(normalized_scores)
    positions = ranking_positions(ranking, len(data.alternatives))
    return {
        "normalized_matrix": normalized_matrix,
        "dominance_matrix": dominance_matrix,
        "raw_scores": raw_scores,
        "normalized_scores": normalized_scores,
        "ranking": ranking,
        "positions": positions,
    }


def result_tables(data: TODIMData, result: dict) -> dict[str, pd.DataFrame]:
    return {
        "normalized": pd.DataFrame(result["normalized_matrix"], index=data.alternatives, columns=data.criteria),
        "dominance": pd.DataFrame(result["dominance_matrix"], index=data.alternatives, columns=data.alternatives),
        "scores": pd.DataFrame(
            {
                "Alternative": data.alternatives,
                "Raw score": result["raw_scores"],
                "Normalized score": result["normalized_scores"],
                "Rank": result["positions"],
            }
        ).sort_values("Rank"),
    }


def compare_methods(
    data: TODIMData,
    methods: Sequence[TODIMMethod] = METHOD_ORDER,
    normalization: NormalizationMethod = "sum",
    reference_method: TODIMMethod = "exptodim",
    theta: float = 1.0,
    lambda_: float = 2.25,
    rho: float = 3.0,
    alpha: float = 0.88,
    beta: float = 0.88,
) -> pd.DataFrame:
    reference = run_todim_family(
        data, reference_method, normalization, theta, lambda_, rho, alpha, beta
    )
    rows = []
    for method in methods:
        result = run_todim_family(data, method, normalization, theta, lambda_, rho, alpha, beta)
        rows.append(
            {
                "method": method,
                "top_alternative": data.alternatives[int(result["ranking"][0])],
                "ranking": " > ".join(data.alternatives[int(index)] for index in result["ranking"]),
                "spearman_vs_reference": spearman_rank_correlation(
                    reference["positions"], result["positions"]
                ),
                "kendall_tau_vs_reference": kendall_tau(
                    reference["positions"], result["positions"]
                ),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_analysis(
    data: TODIMData,
    methods: Sequence[TODIMMethod] = METHOD_ORDER,
    lambdas: Sequence[float] = (1.0, 1.5, 2.25, 3.0, 5.0),
    rhos: Sequence[float] = (1.0, 2.0, 3.0, 5.0, 10.0),
    thetas: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0),
    alphas: Sequence[float] = (0.5, 0.88, 1.0),
    betas: Sequence[float] = (0.5, 0.88, 1.0),
    normalizations: Sequence[NormalizationMethod] = NORMALIZATION_ORDER,
    baseline_method: TODIMMethod = "exptodim",
    baseline_normalization: NormalizationMethod = "sum",
    baseline_lambda: float = 2.25,
    baseline_rho: float = 3.0,
    baseline_theta: float = 1.0,
    baseline_alpha: float = 0.88,
    baseline_beta: float = 0.88,
) -> pd.DataFrame:
    baseline = run_todim_family(
        data,
        baseline_method,
        baseline_normalization,
        baseline_theta,
        baseline_lambda,
        baseline_rho,
        baseline_alpha,
        baseline_beta,
    )
    baseline_ranking = baseline["ranking"]
    baseline_positions = baseline["positions"]
    baseline_top = data.alternatives[int(baseline_ranking[0])]
    rows = []

    for method in methods:
        for normalization in normalizations:
            if method == "classical_todim":
                combinations = [
                    (theta, baseline_lambda, baseline_rho, baseline_alpha, baseline_beta)
                    for theta in thetas
                ]
            elif method in {"log_todim", "exptodim"}:
                combinations = [
                    (baseline_theta, lambda_, rho, baseline_alpha, baseline_beta)
                    for rho in rhos
                    for lambda_ in lambdas
                ]
            else:
                combinations = [
                    (baseline_theta, lambda_, baseline_rho, alpha, beta)
                    for lambda_ in lambdas
                    for alpha in alphas
                    for beta in betas
                ]

            for theta, lambda_, rho, alpha, beta in combinations:
                result = run_todim_family(
                    data, method, normalization, theta, lambda_, rho, alpha, beta
                )
                ranking = result["ranking"]
                positions = result["positions"]
                top = data.alternatives[int(ranking[0])]
                full_stable = bool(np.array_equal(ranking, baseline_ranking))
                row = {
                    "method": method,
                    "normalization": normalization,
                    "theta": float(theta),
                    "rho": float(rho),
                    "lambda": float(lambda_),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "top_alternative": top,
                    "top_stable": bool(top == baseline_top),
                    "full_ranking_stable": full_stable,
                    "changed_ranking": int(not full_stable),
                    "spearman": spearman_rank_correlation(baseline_positions, positions),
                    "kendall_tau": kendall_tau(baseline_positions, positions),
                    "ranking": " > ".join(data.alternatives[int(index)] for index in ranking),
                }
                for index, alternative in enumerate(data.alternatives):
                    row[f"score_{alternative}"] = float(result["normalized_scores"][index])
                    row[f"rank_{alternative}"] = int(positions[index])
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    summary = (
        results.groupby("method", sort=False)
        .agg(
            scenarios=("method", "size"),
            top_one_stability=("top_stable", "mean"),
            full_ranking_stability=("full_ranking_stable", "mean"),
            mean_spearman=("spearman", "mean"),
            min_spearman=("spearman", "min"),
            mean_kendall=("kendall_tau", "mean"),
            changed_rankings=("changed_ranking", "sum"),
        )
        .reset_index()
    )
    return summary


def method_note(method: TODIMMethod) -> str:
    notes = {
        "classical_todim": "Inverse weight appears in the loss term; theta attenuates losses.",
        "generalized_inverse": "Inverse-weight losses are retained; lambda and alpha control loss aversion and curvature.",
        "generalized_monotone": "Generalized Monotone/Power TODIM uses separate gain and loss curvature without inverse-weight losses.",
        "log_todim": "Logarithmic TODIM dampens large normalized differences through lambda and rho.",
        "exptodim": "ExpTODIM uses exponential saturation controlled by rho and asymmetric losses controlled by lambda.",
    }
    return notes[method]
