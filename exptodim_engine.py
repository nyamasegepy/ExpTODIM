from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal, Optional
import itertools
import numpy as np
import pandas as pd

NormalizationMethod = Literal['max','max_min','sum','vector']

@dataclass(frozen=True)
class ExpTODIMData:
    matrix: np.ndarray
    alternatives: list[str]
    criteria: list[str]
    weights: np.ndarray
    senses: np.ndarray  # 1 benefit, 0 cost

def built_in_example() -> ExpTODIMData:
    matrix = np.array([
        [5.0, 2.5, 4.0, 2840.0, 5.0, 3.0, 9.0, 8.0],
        [3.5, 12.0, 6.0, 3700.0, 9.0, 7.0, 3.0, 6.0],
        [2.5, 4.0, 5.0, 2683.0, 4.0, 5.0, 7.0, 7.5],
        [3.0, 13.0, 7.0, 4150.0, 6.0, 9.0, 6.0, 7.0],
        [4.0, 18.0, 9.0, 4500.0, 3.0, 8.0, 5.0, 4.0],
    ], dtype=float)
    alternatives = ['A1','A2','A3','A4','A5']
    criteria = ['Hotel Rating','Time Traveling','Days','Cost','Shopping','Cultural Attractions','Nature','Safety']
    weights = np.array([0.214732143,0.015625000,0.152232143,0.339732143,0.110565476,0.079315476,0.033482143,0.054315476], dtype=float)
    weights = weights / weights.sum()
    senses = np.array([1,0,1,0,1,1,1,1], dtype=int)
    return ExpTODIMData(matrix, alternatives, criteria, weights, senses)

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('weights must be a 1D non-negative vector with positive sum')
    return w / w.sum()

def roc_weights(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError('n must be positive')
    w = np.array([(1/n) * sum(1/j for j in range(i, n+1)) for i in range(1, n+1)], dtype=float)
    return w / w.sum()

def normalize_matrix(matrix: np.ndarray, senses: np.ndarray, method: NormalizationMethod='sum') -> np.ndarray:
    X = np.asarray(matrix, dtype=float)
    s = np.asarray(senses, dtype=int)
    if X.ndim != 2:
        raise ValueError('matrix must be 2D')
    if s.shape != (X.shape[1],):
        raise ValueError('senses must have one entry per criterion')
    M = np.empty_like(X, dtype=float)
    b = s == 1
    c = s == 0
    if method == 'max':
        maxj = X.max(axis=0)
        if np.any(maxj == 0):
            raise ZeroDivisionError('max normalization cannot divide by zero')
        M[:, b] = X[:, b] / maxj[b]
        M[:, c] = 1.0 - X[:, c] / maxj[c]
    elif method == 'max_min':
        maxj = X.max(axis=0); minj = X.min(axis=0); denom = maxj - minj
        const = denom == 0
        M[:, const] = 1.0
        nb = b & ~const; nc = c & ~const
        M[:, nb] = (X[:, nb] - minj[nb]) / denom[nb]
        M[:, nc] = (maxj[nc] - X[:, nc]) / denom[nc]
    elif method == 'sum':
        if np.any(c) and np.any(X[:, c] == 0):
            raise ZeroDivisionError('cost criteria cannot contain zero under sum normalization')
        if np.any(b):
            denom = X[:, b].sum(axis=0)
            if np.any(denom == 0):
                raise ZeroDivisionError('benefit criterion sum cannot be zero')
            M[:, b] = X[:, b] / denom
        if np.any(c):
            inv = 1.0 / X[:, c]
            M[:, c] = inv / inv.sum(axis=0)
    elif method == 'vector':
        if np.any(c) and np.any(X[:, c] == 0):
            raise ZeroDivisionError('cost criteria cannot contain zero under vector normalization')
        if np.any(b):
            denom = np.linalg.norm(X[:, b], axis=0)
            if np.any(denom == 0):
                raise ZeroDivisionError('benefit criterion norm cannot be zero')
            M[:, b] = X[:, b] / denom
        if np.any(c):
            inv = 1.0 / X[:, c]
            M[:, c] = inv / np.linalg.norm(inv, axis=0)
    else:
        raise ValueError('method must be max, max_min, sum, or vector')
    return M

def compute_dominance(normalized_matrix: np.ndarray, weights: np.ndarray, rho: float=3.0, lambda_: float=2.25) -> np.ndarray:
    M = np.asarray(normalized_matrix, dtype=float)
    w = normalize_weights(weights)
    if rho <= 0 or lambda_ <= 0:
        raise ValueError('rho and lambda_ must be positive')
    if w.shape != (M.shape[1],):
        raise ValueError('weights length must equal number of criteria')
    Phi = np.zeros((M.shape[0], M.shape[0]), dtype=float)
    ln10 = np.log(10.0)
    for j in range(M.shape[1]):
        col = M[:, j]
        delta = col[:, None] - col[None, :]
        base = 1.0 - np.exp(-rho * np.abs(delta) * ln10)
        contrib = np.zeros_like(delta)
        contrib[delta > 0] = w[j] * base[delta > 0]
        contrib[delta < 0] = -lambda_ * w[j] * base[delta < 0]
        Phi += contrib
    return Phi

def compute_scores(Phi: np.ndarray, normalize_output: bool=True) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(Phi, dtype=float).sum(axis=1)
    if not normalize_output:
        return raw, raw.copy()
    mn, mx = raw.min(), raw.max()
    norm = np.ones_like(raw) if mx == mn else (raw - mn) / (mx - mn)
    return raw, norm

def rank_alternatives(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=float), kind='mergesort')

def ranking_positions(ranking: np.ndarray, n_alternatives: Optional[int]=None) -> np.ndarray:
    r = np.asarray(ranking, dtype=int)
    n = len(r) if n_alternatives is None else n_alternatives
    pos = np.empty(n, dtype=int)
    for p, idx in enumerate(r, start=1):
        pos[idx] = p
    return pos

def spearman_rank_correlation(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    a = np.asarray(pos_a, dtype=float); b = np.asarray(pos_b, dtype=float)
    ac = a - a.mean(); bc = b - b.mean()
    denom = np.sqrt(np.sum(ac**2) * np.sum(bc**2))
    return 1.0 if denom == 0 else float(np.sum(ac * bc) / denom)

def kendall_tau(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    a = np.asarray(pos_a, dtype=int); b = np.asarray(pos_b, dtype=int)
    conc = disc = 0
    for i, k in itertools.combinations(range(len(a)), 2):
        prod = np.sign(a[i]-a[k]) * np.sign(b[i]-b[k])
        if prod > 0: conc += 1
        elif prod < 0: disc += 1
    total = conc + disc
    return 1.0 if total == 0 else float((conc - disc) / total)

def run_exptodim(data: ExpTODIMData, normalization: NormalizationMethod='sum', rho: float=3.0, lambda_: float=2.25) -> dict:
    M = normalize_matrix(data.matrix, data.senses, normalization)
    Phi = compute_dominance(M, data.weights, rho=rho, lambda_=lambda_)
    raw, norm = compute_scores(Phi, normalize_output=True)
    ranking = rank_alternatives(norm)
    positions = ranking_positions(ranking, len(data.alternatives))
    return {'normalized_matrix': M, 'dominance_matrix': Phi, 'raw_scores': raw, 'normalized_scores': norm, 'ranking': ranking, 'positions': positions}

def sensitivity_analysis(data: ExpTODIMData, lambdas=(1.0,1.5,2.25,3.0,5.0), rhos=(1.0,2.0,3.0,5.0,10.0), normalizations=('max','max_min','sum','vector'), baseline_normalization='sum', baseline_lambda=2.25, baseline_rho=3.0) -> pd.DataFrame:
    base = run_exptodim(data, baseline_normalization, baseline_rho, baseline_lambda)
    base_ranking = base['ranking']; base_pos = base['positions']; base_top = data.alternatives[int(base_ranking[0])]
    rows = []
    for norm in normalizations:
        for rho in rhos:
            for lam in lambdas:
                res = run_exptodim(data, norm, float(rho), float(lam))
                ranking = res['ranking']; pos = res['positions']; top = data.alternatives[int(ranking[0])]
                row = {'normalization': norm, 'rho': float(rho), 'lambda': float(lam), 'top_alternative': top, 'top_stable': top == base_top, 'full_ranking_stable': bool(np.array_equal(ranking, base_ranking)), 'spearman': spearman_rank_correlation(base_pos, pos), 'kendall_tau': kendall_tau(base_pos, pos), 'ranking': ' > '.join(data.alternatives[int(i)] for i in ranking)}
                for idx, alt in enumerate(data.alternatives):
                    row[f'score_{alt}'] = float(res['normalized_scores'][idx])
                    row[f'rank_{alt}'] = int(pos[idx])
                rows.append(row)
    return pd.DataFrame(rows)

def result_tables(data: ExpTODIMData, result: dict) -> dict[str, pd.DataFrame]:
    return {
        'normalized': pd.DataFrame(result['normalized_matrix'], index=data.alternatives, columns=data.criteria),
        'dominance': pd.DataFrame(result['dominance_matrix'], index=data.alternatives, columns=data.alternatives),
        'scores': pd.DataFrame({'Alternative': data.alternatives, 'Raw score V(i)': result['raw_scores'], "Normalized score V'(i)": result['normalized_scores'], 'Rank': result['positions']}).sort_values('Rank')
    }
