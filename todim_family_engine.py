from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal, Optional
import itertools
import numpy as np
import pandas as pd

NormalizationMethod = Literal['max','max_min','sum','vector']
TODIMMethod = Literal['classical_todim','generalized_inverse','generalized_monotone','power_todim','log_todim','exptodim']

@dataclass(frozen=True)
class TODIMData:
    matrix: np.ndarray
    alternatives: list[str]
    criteria: list[str]
    weights: np.ndarray
    senses: np.ndarray  # 1 benefit, 0 cost

def built_in_example() -> TODIMData:
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
    return TODIMData(matrix, alternatives, criteria, weights, senses)

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError('weights must be one-dimensional')
    if np.any(weights < 0):
        raise ValueError('weights must be non-negative')
    s = weights.sum()
    if s <= 0:
        raise ValueError('sum of weights must be positive')
    return weights / s

def normalize_matrix(matrix: np.ndarray, senses: np.ndarray, method: NormalizationMethod='sum') -> np.ndarray:
    X = np.asarray(matrix, dtype=float)
    senses = np.asarray(senses, dtype=int)
    if X.ndim != 2:
        raise ValueError('matrix must be two-dimensional')
    if senses.shape != (X.shape[1],):
        raise ValueError('senses length must equal number of criteria')
    M = np.empty_like(X, dtype=float)
    benefit = senses == 1
    cost = senses == 0
    if method == 'max':
        maxj = X.max(axis=0)
        minj = X.min(axis=0)
        if np.any(benefit) and np.any(maxj[benefit] == 0):
            raise ZeroDivisionError('benefit criterion max cannot be zero')
        if np.any(cost) and np.any(X[:, cost] == 0):
            raise ZeroDivisionError('cost criteria cannot contain zero under inverse max normalization')
        M[:, benefit] = X[:, benefit] / maxj[benefit]
        M[:, cost] = minj[cost] / X[:, cost]
    elif method == 'max_min':
        maxj = X.max(axis=0); minj = X.min(axis=0); denom = maxj - minj
        const = denom == 0; M[:, const] = 1.0
        b = benefit & ~const; c = cost & ~const
        M[:, b] = (X[:, b] - minj[b]) / denom[b]
        M[:, c] = (maxj[c] - X[:, c]) / denom[c]
    elif method == 'sum':
        if np.any(benefit):
            denom = X[:, benefit].sum(axis=0)
            if np.any(denom == 0): raise ZeroDivisionError('benefit column sum cannot be zero')
            M[:, benefit] = X[:, benefit] / denom
        if np.any(cost):
            if np.any(X[:, cost] == 0): raise ZeroDivisionError('cost criteria cannot contain zero')
            inv = 1.0 / X[:, cost]
            M[:, cost] = inv / inv.sum(axis=0)
    elif method == 'vector':
        if np.any(benefit):
            denom = np.linalg.norm(X[:, benefit], axis=0)
            if np.any(denom == 0): raise ZeroDivisionError('benefit norm cannot be zero')
            M[:, benefit] = X[:, benefit] / denom
        if np.any(cost):
            if np.any(X[:, cost] == 0): raise ZeroDivisionError('cost criteria cannot contain zero')
            inv = 1.0 / X[:, cost]
            M[:, cost] = inv / np.linalg.norm(inv, axis=0)
    else:
        raise ValueError('unknown normalization method')
    return M

def _pow_nonneg(x, p):
    return np.power(np.maximum(np.asarray(x, dtype=float), 0.0), p)

def compute_todim_family_dominance(M: np.ndarray, weights: np.ndarray, method: TODIMMethod='exptodim', theta: float=1.0, lambda_: float=2.25, rho: float=3.0, alpha: float=0.88, beta: float=0.88) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    weights = normalize_weights(np.asarray(weights, dtype=float))
    if np.any(weights <= 0): raise ValueError('all weights must be positive')
    if theta <= 0 or lambda_ <= 0 or rho <= 0 or alpha <= 0 or beta <= 0:
        raise ValueError('parameters must be positive')
    n_alt, n_crit = M.shape
    Phi = np.zeros((n_alt, n_alt), dtype=float)
    ln10 = np.log(10.0)
    for k in range(n_crit):
        d = M[:, [k]] - M[:, [k]].T
        a = np.abs(d)
        gain = d > 0
        loss = d < 0
        wk = float(weights[k])
        C = np.zeros_like(d)
        if method == 'classical_todim':
            C[gain] = np.sqrt(wk * a[gain])
            C[loss] = -(1.0/theta) * np.sqrt(a[loss] / wk)
        elif method == 'generalized_inverse':
            C[gain] = _pow_nonneg(wk * a[gain], alpha)
            C[loss] = -lambda_ * _pow_nonneg(a[loss] / wk, alpha)
        elif method in {'generalized_monotone', 'power_todim'}:
            C[gain] = wk * _pow_nonneg(a[gain], alpha)
            C[loss] = -lambda_ * wk * _pow_nonneg(a[loss], beta)
        elif method == 'log_todim':
            base = np.log1p(10.0 * rho * a)
            C[gain] = wk * base[gain]
            C[loss] = -lambda_ * wk * base[loss]
        elif method == 'exptodim':
            base = 1.0 - np.exp(-rho * a * ln10)
            C[gain] = wk * base[gain]
            C[loss] = -lambda_ * wk * base[loss]
        else:
            raise ValueError('unknown TODIM-family method')
        Phi += C
    return Phi

def compute_scores(Phi: np.ndarray, normalize_output=True):
    raw = np.asarray(Phi, dtype=float).sum(axis=1)
    if not normalize_output:
        return raw, raw.copy()
    mn, mx = raw.min(), raw.max()
    norm = np.ones_like(raw) if mx == mn else (raw - mn)/(mx - mn)
    return raw, norm

def rank_alternatives(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=float), kind='mergesort')

def ranking_positions(ranking: np.ndarray, n_alternatives: Optional[int]=None) -> np.ndarray:
    ranking = np.asarray(ranking, dtype=int)
    if n_alternatives is None: n_alternatives = len(ranking)
    pos = np.empty(n_alternatives, dtype=int)
    for p, idx in enumerate(ranking, start=1): pos[idx] = p
    return pos

def spearman_rank_correlation(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ac = a - a.mean(); bc = b - b.mean()
    den = np.sqrt((ac*ac).sum() * (bc*bc).sum())
    return 1.0 if den == 0 else float((ac*bc).sum()/den)

def kendall_tau(a, b):
    a = np.asarray(a, dtype=int); b = np.asarray(b, dtype=int)
    c = d = 0
    for i, k in itertools.combinations(range(len(a)), 2):
        prod = np.sign(a[i]-a[k]) * np.sign(b[i]-b[k])
        if prod > 0: c += 1
        elif prod < 0: d += 1
    return 1.0 if c+d == 0 else float((c-d)/(c+d))

def run_todim_family(data: TODIMData, method: TODIMMethod='exptodim', normalization: NormalizationMethod='sum', theta=1.0, lambda_=2.25, rho=3.0, alpha=0.88, beta=0.88) -> dict:
    M = normalize_matrix(data.matrix, data.senses, normalization)
    Phi = compute_todim_family_dominance(M, data.weights, method, theta, lambda_, rho, alpha, beta)
    raw, norm = compute_scores(Phi, True)
    ranking = rank_alternatives(norm)
    pos = ranking_positions(ranking, len(data.alternatives))
    return {'normalized_matrix': M, 'dominance_matrix': Phi, 'raw_scores': raw, 'normalized_scores': norm, 'ranking': ranking, 'positions': pos}

def result_tables(data: TODIMData, result: dict) -> dict[str, pd.DataFrame]:
    return {
        'normalized': pd.DataFrame(result['normalized_matrix'], index=data.alternatives, columns=data.criteria),
        'dominance': pd.DataFrame(result['dominance_matrix'], index=data.alternatives, columns=data.alternatives),
        'scores': pd.DataFrame({'Alternative': data.alternatives, 'Raw score': result['raw_scores'], 'Normalized score': result['normalized_scores'], 'Rank': result['positions']}).sort_values('Rank')
    }

def compare_methods(data: TODIMData, methods=('classical_todim','generalized_inverse','generalized_monotone','power_todim','log_todim','exptodim'), normalization='sum', reference_method='exptodim', theta=1.0, lambda_=2.25, rho=3.0, alpha=0.88, beta=0.88) -> pd.DataFrame:
    ref = run_todim_family(data, reference_method, normalization, theta, lambda_, rho, alpha, beta)
    rows = []
    for method in methods:
        r = run_todim_family(data, method, normalization, theta, lambda_, rho, alpha, beta)
        rows.append({
            'method': method,
            'top_alternative': data.alternatives[int(r['ranking'][0])],
            'ranking': ' > '.join(data.alternatives[int(i)] for i in r['ranking']),
            'spearman_vs_reference': spearman_rank_correlation(ref['positions'], r['positions']),
            'kendall_tau_vs_reference': kendall_tau(ref['positions'], r['positions'])
        })
    return pd.DataFrame(rows)

def sensitivity_analysis(data: TODIMData, methods=('exptodim',), lambdas=(1.0,1.5,2.25,3.0,5.0), rhos=(1.0,2.0,3.0,5.0,10.0), thetas=(0.5,1.0,2.0,5.0,10.0), alphas=(0.5,0.88,1.0), betas=(0.5,0.88,1.0), normalizations=('max','max_min','sum','vector'), baseline_method='exptodim', baseline_normalization='sum', baseline_lambda=2.25, baseline_rho=3.0, baseline_theta=1.0, baseline_alpha=0.88, baseline_beta=0.88) -> pd.DataFrame:
    base = run_todim_family(data, baseline_method, baseline_normalization, baseline_theta, baseline_lambda, baseline_rho, baseline_alpha, baseline_beta)
    base_rank = base['ranking']; base_pos = base['positions']; base_top = data.alternatives[int(base_rank[0])]
    rows = []
    for method in methods:
        for norm in normalizations:
            if method == 'classical_todim':
                iterator = [(t, baseline_lambda, baseline_rho, baseline_alpha, baseline_beta) for t in thetas]
            elif method in {'exptodim','log_todim'}:
                iterator = [(baseline_theta, l, r, baseline_alpha, baseline_beta) for r in rhos for l in lambdas]
            else:
                iterator = [(baseline_theta, l, baseline_rho, a, b) for l in lambdas for a in alphas for b in betas]
            for t, l, r, a, b in iterator:
                res = run_todim_family(data, method, norm, t, l, r, a, b)
                ranking = res['ranking']; pos = res['positions']; top = data.alternatives[int(ranking[0])]
                row = {
                    'method': method, 'normalization': norm, 'theta': float(t), 'rho': float(r), 'lambda': float(l), 'alpha': float(a), 'beta': float(b),
                    'top_alternative': top, 'top_stable': bool(top == base_top), 'full_ranking_stable': bool(np.array_equal(ranking, base_rank)),
                    'spearman': spearman_rank_correlation(base_pos, pos), 'kendall_tau': kendall_tau(base_pos, pos),
                    'ranking': ' > '.join(data.alternatives[int(i)] for i in ranking)
                }
                for idx, alt in enumerate(data.alternatives):
                    row[f'score_{alt}'] = float(res['normalized_scores'][idx]); row[f'rank_{alt}'] = int(pos[idx])
                rows.append(row)
    return pd.DataFrame(rows)

def method_warning(method: TODIMMethod) -> str:
    if method in {'classical_todim','generalized_inverse'}:
        return 'This formulation includes inverse dependence on criterion weight in the loss term; literature reports possible weight-related paradoxes.'
    if method in {'generalized_monotone','power_todim'}:
        return 'Power TODIM / monotone-prospect uses independent alpha and beta curvature parameters and avoids inverse-weight losses.'
    if method == 'log_todim':
        return 'LogTODIM uses logarithmic dampening to reduce the dominance of extreme criterion differences/outliers.'
    if method == 'exptodim':
        return 'ExpTODIM uses exponential saturation and loss aversion without a reference criterion. Test robustness over rho, lambda and normalization.'
    return ''
