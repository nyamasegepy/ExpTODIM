# ExpTODIM Robustness Dashboard

Interactive dashboard for the paper idea:

**Exploring Ranking Robustness in ExpTODIM: Effects of Loss Aversion, Exponential Sensitivity and Normalization**

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Contents

- `exptodim_engine.py`: ExpTODIM computational engine.
- `app.py`: Streamlit dashboard.
- `requirements.txt`: Python dependencies.

## Built-in example

5 alternatives and 8 criteria:
Hotel Rating, Time Traveling, Days, Cost, Shopping, Cultural Attractions, Nature, Safety.

Baseline:
- normalization: sum
- loss aversion: lambda = 2.25
- exponential sensitivity: rho = 3

The baseline ranking is: A1 > A2 > A3 > A4 > A5.
