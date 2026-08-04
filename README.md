# TODIM Family Robustness Dashboard

Streamlit dashboard and Python engine for scenario-based sensitivity analysis of five distinct crisp TODIM-family dominance functions:

1. Classical TODIM
2. Generalized TODIM with inverse-weight losses
3. Generalized Monotone / Power TODIM
4. Logarithmic TODIM
5. ExpTODIM

Generalized Monotone and Power TODIM are represented by one entry because the implemented crisp-data dominance function is mathematically identical.

## Reference configuration

The manuscript reference is ExpTODIM with sum normalization, loss aversion `lambda=2.25`, and exponential sensitivity `rho=3.0`. The standard family grid contains 580 method-specific deterministic scenarios:

| Method | Parameters varied | Scenarios |
|---|---|---:|
| Classical TODIM | 5 theta levels x 4 normalizations | 20 |
| Generalized inverse | 5 lambda x 3 alpha x 3 beta x 4 normalizations | 180 |
| Generalized Monotone / Power | 5 lambda x 3 alpha x 3 beta x 4 normalizations | 180 |
| Logarithmic TODIM | 5 lambda x 5 rho x 4 normalizations | 100 |
| ExpTODIM | 5 lambda x 5 rho x 4 normalizations | 100 |

The dashboard reports top-one stability, full-ranking stability, Spearman correlation, Kendall tau, minimum Spearman, and changed-ranking counts. Robustness is conditional on the selected grid; grid levels are not probability distributions.

## Included cases

- Travel-destination decision matrix
- Balanced trade-offs
- Loss-aversion sensitive trade-offs
- Close-score rho test
- Weight-sensitive case
- Benefit/cost sense test
- Optional CSV upload

Synthetic matrices and their weight/sense files are stored in `data/`.

## Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Reproduce manuscript outputs

```powershell
python generate_default_outputs.py
python -m unittest discover -s tests -v
```

The generator creates:

- `method_comparison_default.csv`
- `sensitivity_results_todim_family_default.csv`
- `robustness_summary_default.csv`

## Files

- `todim_family_engine.py`: validated computational engine and robustness indicators.
- `app.py`: Streamlit dashboard.
- `generate_default_outputs.py`: reproducibility output generator.
- `tests/test_engine.py`: numerical regression tests.
- `data/`: synthetic stress matrices and criterion settings.

Public package: <https://github.com/nyamasegepy/ExpTODIM>

Deployed dashboard: <https://exptodim-ehbjcaajhaahuuj3y3jzjm.streamlit.app/>
