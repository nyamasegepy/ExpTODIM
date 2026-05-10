# TODIM Family Robustness Dashboard

This package implements a computational dashboard for the TODIM family:

1. Classical TODIM
2. Generalized TODIM — inverse-weight loss
3. Generalized TODIM — monotone/prospect
4. Power TODIM
5. Logarithmic TODIM
6. ExpTODIM

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `todim_family_engine.py`: computational engine.
- `app.py`: Streamlit dashboard.
- `requirements.txt`: dependencies.
- `method_comparison_default.csv`: default comparison across methods.
- `sensitivity_results_todim_family_default.csv`: default sensitivity grid.

## Notes

The classical and inverse-weight generalized formulations reproduce the inverse-weight loss behavior discussed in the literature. The monotone/prospect and Power TODIM formulations avoid inverse-weight losses and use gain/loss curvature parameters. Logarithmic TODIM uses logarithmic dampening controlled by rho and lambda. ExpTODIM uses an exponential saturation function controlled by rho and lambda.

The revised package uses inverse-ratio max normalization for cost criteria, i.e. min(x_j)/x_ij, to avoid assuming that zero is always the ideal cost value.
