from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from exptodim_engine import ExpTODIMData, built_in_example, normalize_weights, run_exptodim, sensitivity_analysis, result_tables

st.set_page_config(page_title='ExpTODIM Robustness Dashboard', layout='wide')
LABELS = {'max':'Max normalization','max_min':'Max-min normalization','sum':'Sum normalization','vector':'Vector normalization'}

def read_uploaded_csv(uploaded_file) -> ExpTODIMData:
    df = pd.read_csv(uploaded_file)
    alternatives = df.iloc[:,0].astype(str).tolist()
    criteria = list(df.columns[1:])
    matrix = df.iloc[:,1:].to_numpy(dtype=float)
    weights = np.ones(len(criteria), dtype=float) / len(criteria)
    senses = np.ones(len(criteria), dtype=int)
    return ExpTODIMData(matrix, alternatives, criteria, weights, senses)

def plot_scores(scores_df):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(scores_df['Alternative'], scores_df["Normalized score V'(i)"])
    ax.set_xlabel('Alternative'); ax.set_ylabel("Normalized score V'(i)"); ax.set_title('ExpTODIM ranking scores')
    ax.set_ylim(0, 1.1); ax.grid(axis='y', alpha=0.3)
    return fig

def plot_heatmap(df, normalization, value_col, title):
    sub = df[df['normalization'] == normalization]
    pivot = sub.pivot_table(index='rho', columns='lambda', values=value_col, aggfunc='mean')
    fig, ax = plt.subplots(figsize=(7,4.5))
    im = ax.imshow(pivot.values, aspect='auto', origin='lower')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel('Loss aversion λ'); ax.set_ylabel('Exponential sensitivity ρ'); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def plot_top_map(df, normalization, alternatives):
    sub = df[df['normalization'] == normalization].copy()
    code = {alt:i+1 for i, alt in enumerate(alternatives)}
    sub['top_code'] = sub['top_alternative'].map(code)
    pivot = sub.pivot_table(index='rho', columns='lambda', values='top_code', aggfunc='first')
    fig, ax = plt.subplots(figsize=(7,4.5))
    im = ax.imshow(pivot.values, aspect='auto', origin='lower')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel('Loss aversion λ'); ax.set_ylabel('Exponential sensitivity ρ'); ax.set_title('Top-ranked alternative map')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_ticks(list(code.values())); cbar.set_ticklabels(list(code.keys()))
    return fig

st.title('Exploring Ranking Robustness in ExpTODIM')
st.caption('Effects of loss aversion (λ), exponential sensitivity (ρ), and normalization')

with st.sidebar:
    st.header('1. Data')
    source = st.radio('Data source', ['Built-in 5×8 example','Upload CSV'])
    if source == 'Built-in 5×8 example':
        data = built_in_example()
    else:
        up = st.file_uploader('Upload CSV: first column = alternative, remaining columns = criteria', type=['csv'])
        if up is None: st.stop()
        data = read_uploaded_csv(up)
    st.header('2. Scenario controls')
    normalization = st.selectbox('Normalization', list(LABELS.keys()), index=2, format_func=lambda x: LABELS[x])
    lambda_ = st.slider('Loss aversion λ', 0.5, 6.0, 2.25, 0.25)
    rho = st.slider('Exponential sensitivity ρ', 0.5, 10.0, 3.0, 0.5)
    st.header('3. Baseline')
    baseline_norm = st.selectbox('Baseline normalization', list(LABELS.keys()), index=2, format_func=lambda x: LABELS[x])
    baseline_lambda = st.number_input('Baseline λ', value=2.25, min_value=0.1, max_value=20.0, step=0.25)
    baseline_rho = st.number_input('Baseline ρ', value=3.0, min_value=0.1, max_value=20.0, step=0.5)
    st.header('4. Sensitivity grid')
    lambda_grid_text = st.text_input('λ values', '1.0, 1.5, 2.25, 3.0, 5.0')
    rho_grid_text = st.text_input('ρ values', '1.0, 2.0, 3.0, 5.0, 10.0')
    selected_norms = st.multiselect('Normalizations', list(LABELS.keys()), default=list(LABELS.keys()), format_func=lambda x: LABELS[x])
    st.header('5. Weights and senses')
    weights_text = st.text_area('Weights', ', '.join(f'{x:.9f}' for x in data.weights), height=80)
    senses_text = st.text_area('Senses (1=benefit, 0=cost)', ', '.join(str(int(x)) for x in data.senses), height=70)

try:
    w = np.array([float(x.strip()) for x in weights_text.split(',') if x.strip()], dtype=float)
    s = np.array([int(x.strip()) for x in senses_text.split(',') if x.strip()], dtype=int)
    if len(w) != len(data.criteria) or len(s) != len(data.criteria):
        st.error('Weights and senses must match number of criteria.'); st.stop()
    data = ExpTODIMData(data.matrix, data.alternatives, data.criteria, normalize_weights(w), s)
    result = run_exptodim(data, normalization, rho, lambda_)
    tables = result_tables(data, result)
except Exception as e:
    st.error(f'Error: {e}'); st.stop()

tab1, tab2, tab3, tab4 = st.tabs(['Current scenario','Sensitivity analysis','Data','Method notes'])
with tab1:
    st.subheader('Current scenario')
    c1,c2,c3 = st.columns(3); c1.metric('Normalization', LABELS[normalization]); c2.metric('λ', f'{lambda_:.2f}'); c3.metric('ρ', f'{rho:.2f}')
    st.success('Ranking: ' + ' > '.join(data.alternatives[int(i)] for i in result['ranking']))
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.dataframe(tables['scores'], use_container_width=True)
        st.download_button('Download current scores', tables['scores'].to_csv(index=False).encode('utf-8'), 'current_scores.csv', 'text/csv')
    with col2:
        st.pyplot(plot_scores(tables['scores']))
    with st.expander('Normalized matrix M'):
        st.dataframe(tables['normalized'].round(6), use_container_width=True)
    with st.expander('Dominance matrix Φ'):
        st.dataframe(tables['dominance'].round(6), use_container_width=True)
with tab2:
    st.subheader('Sensitivity analysis')
    try:
        lambdas = tuple(float(x.strip()) for x in lambda_grid_text.split(',') if x.strip())
        rhos = tuple(float(x.strip()) for x in rho_grid_text.split(',') if x.strip())
        sens = sensitivity_analysis(data, lambdas=lambdas, rhos=rhos, normalizations=selected_norms, baseline_normalization=baseline_norm, baseline_lambda=baseline_lambda, baseline_rho=baseline_rho)
    except Exception as e:
        st.error(f'Sensitivity analysis failed: {e}'); st.stop()
    m1,m2,m3,m4 = st.columns(4); m1.metric('Scenarios', len(sens)); m2.metric('Top-1 stability', f'{100*sens.top_stable.mean():.1f}%'); m3.metric('Full ranking stability', f'{100*sens.full_ranking_stable.mean():.1f}%'); m4.metric('Mean Spearman', f'{sens.spearman.mean():.3f}')
    st.dataframe(sens, use_container_width=True)
    st.download_button('Download sensitivity results', sens.to_csv(index=False).encode('utf-8'), 'sensitivity_results.csv', 'text/csv')
    heat_norm = st.selectbox('Normalization for heatmaps', selected_norms, format_func=lambda x: LABELS[x])
    h1,h2 = st.columns(2)
    with h1: st.pyplot(plot_heatmap(sens, heat_norm, 'spearman', f'Spearman correlation ({LABELS[heat_norm]})'))
    with h2: st.pyplot(plot_heatmap(sens, heat_norm, 'kendall_tau', f'Kendall tau ({LABELS[heat_norm]})'))
    h3,h4 = st.columns(2)
    with h3: st.pyplot(plot_heatmap(sens, heat_norm, 'top_stable', f'Top-1 stability ({LABELS[heat_norm]})'))
    with h4: st.pyplot(plot_top_map(sens, heat_norm, data.alternatives))
with tab3:
    st.dataframe(pd.DataFrame(data.matrix, index=data.alternatives, columns=data.criteria), use_container_width=True)
    st.dataframe(pd.DataFrame({'Criterion': data.criteria, 'Weight': data.weights, 'Sense': ['Benefit' if x == 1 else 'Cost' for x in data.senses]}), use_container_width=True)
with tab4:
    st.markdown(r'''The dashboard implements:

$$\delta_{ijk}=m_{ij}-m_{kj}$$

$$\Delta_{ijk}=\begin{cases}0,&\delta_{ijk}=0\\w_j(1-10^{-\rho|\delta_{ijk}|}),&\delta_{ijk}>0\\-\lambda w_j(1-10^{-\rho|\delta_{ijk}|}),&\delta_{ijk}<0\end{cases}$$

$$\Phi(i,k)=\sum_j\Delta_{ijk}, \qquad V(i)=\sum_k\Phi(i,k).$$

The robustness analysis varies $\lambda$, $\rho$ and the normalization method.''')
