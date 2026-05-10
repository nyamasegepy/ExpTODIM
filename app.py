from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from todim_family_engine import TODIMData, built_in_example, normalize_weights, run_todim_family, result_tables, compare_methods, sensitivity_analysis, method_warning

st.set_page_config(page_title='TODIM Family Robustness Dashboard', layout='wide')
METHOD_LABELS = {
    'classical_todim':'Classical TODIM',
    'generalized_inverse':'Generalized TODIM — inverse-weight loss',
    'generalized_monotone':'Generalized TODIM — monotone/prospect',
    'power_todim':'Power TODIM',
    'log_todim':'Logarithmic TODIM',
    'exptodim':'ExpTODIM'
}
NORM_LABELS = {'max':'Max normalization','max_min':'Max-min normalization','sum':'Sum normalization','vector':'Vector normalization'}

def read_uploaded_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if df.shape[1] < 3: raise ValueError('CSV must contain one alternative column and at least two criteria columns.')
    alternatives = df.iloc[:,0].astype(str).tolist()
    criteria = list(df.columns[1:])
    matrix = df.iloc[:,1:].to_numpy(dtype=float)
    weights = np.ones(len(criteria))/len(criteria)
    senses = np.ones(len(criteria), dtype=int)
    return TODIMData(matrix, alternatives, criteria, weights, senses)

def plot_bar(scores_df):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(scores_df['Alternative'].astype(str), scores_df['Normalized score'].astype(float))
    ax.set_xlabel('Alternative'); ax.set_ylabel('Normalized global score'); ax.set_title('TODIM-family ranking scores')
    ax.set_ylim(0, 1.1); ax.grid(axis='y', alpha=.3)
    return fig

def csv_download(df, filename, label):
    st.download_button(label, df.to_csv(index=False).encode('utf-8'), file_name=filename, mime='text/csv')

def heatmap(df, method, normalization, value_col, title):
    sub = df[(df['method']==method) & (df['normalization']==normalization)].copy()
    fig, ax = plt.subplots(figsize=(7,4.5))
    if sub.empty:
        ax.text(.5,.5,'No data',ha='center'); ax.axis('off'); return fig
    if method == 'classical_todim':
        pivot = sub.pivot_table(index='normalization', columns='theta', values=value_col, aggfunc='mean')
        xlabel='theta'; ylabel='normalization'
    elif method in {'exptodim','log_todim'}:
        pivot = sub.pivot_table(index='rho', columns='lambda', values=value_col, aggfunc='mean')
        xlabel='lambda'; ylabel='rho'
    else:
        pivot = sub.pivot_table(index='alpha', columns='lambda', values=value_col, aggfunc='mean')
        xlabel='lambda'; ylabel='alpha'
    im = ax.imshow(pivot.values, aspect='auto', origin='lower')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    return fig

st.title('TODIM Family Robustness Dashboard')
st.caption('Classical TODIM, Generalized TODIM, monotone/prospect TODIM, and ExpTODIM')

with st.sidebar:
    st.header('1. Data')
    source = st.radio('Data source', ['Built-in 5×8 example','Upload CSV'], index=0)
    if source == 'Built-in 5×8 example':
        data = built_in_example()
    else:
        uploaded = st.file_uploader('Upload CSV', type=['csv'])
        if uploaded is None:
            st.info('Upload a CSV or use built-in example.'); st.stop()
        data = read_uploaded_csv(uploaded)
    st.header('2. Method')
    method = st.selectbox('TODIM-family method', list(METHOD_LABELS), index=list(METHOD_LABELS).index('exptodim'), format_func=lambda x: METHOD_LABELS[x])
    normalization = st.selectbox('Normalization', list(NORM_LABELS), index=2, format_func=lambda x: NORM_LABELS[x])
    st.header('3. Parameters')
    theta = 1.0; lambda_ = 2.25; rho = 3.0; alpha = .88; beta = .88
    if method == 'classical_todim': theta = st.slider('Loss attenuation θ', .25, 10.0, 1.0, .25)
    elif method == 'exptodim':
        lambda_ = st.slider('Loss aversion λ', .5, 6.0, 2.25, .25)
        rho = st.slider('Exponential sensitivity ρ', .5, 10.0, 3.0, .5)
    else:
        lambda_ = st.slider('Loss aversion λ', .5, 6.0, 2.25, .25)
        alpha = st.slider('Gain curvature α', .1, 2.0, .88, .01)
        beta = st.slider('Loss curvature β', .1, 2.0, .88, .01)
    st.header('4. Edit weights/senses')
    weights_text = st.text_area('Weights', ', '.join(f'{x:.9f}' for x in data.weights), height=80)
    senses_text = st.text_area('Senses: 1=benefit, 0=cost', ', '.join(str(int(x)) for x in data.senses), height=70)

try:
    weights = np.array([float(x.strip()) for x in weights_text.split(',') if x.strip()], dtype=float)
    senses = np.array([int(x.strip()) for x in senses_text.split(',') if x.strip()], dtype=int)
    if len(weights) != len(data.criteria) or len(senses) != len(data.criteria): raise ValueError('Wrong length')
    data = TODIMData(data.matrix, data.alternatives, data.criteria, normalize_weights(weights), senses)
except Exception as e:
    st.error(f'Invalid weights/senses: {e}'); st.stop()

try:
    result = run_todim_family(data, method, normalization, theta, lambda_, rho, alpha, beta)
    tables = result_tables(data, result)
except Exception as e:
    st.error(f'Scenario failed: {e}'); st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Single method','Compare methods','Sensitivity analysis','Data','Method notes'])
with tab1:
    st.subheader('Single-method result')
    if method in {'classical_todim','generalized_inverse'}: st.warning(method_warning(method))
    else: st.info(method_warning(method))
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Method', METHOD_LABELS[method]); c2.metric('Normalization', NORM_LABELS[normalization])
    c3.metric('Top alternative', data.alternatives[int(result['ranking'][0])])
    c4.metric('Ranking', ' > '.join(data.alternatives[int(i)] for i in result['ranking']))
    left, right = st.columns([1.1,1.0])
    with left:
        st.write('### Scores and ranking'); st.dataframe(tables['scores'], use_container_width=True)
        csv_download(tables['scores'], 'current_scores.csv', 'Download current scores')
    with right: st.pyplot(plot_bar(tables['scores']))
    with st.expander('Normalized matrix'): st.dataframe(tables['normalized'].round(6), use_container_width=True)
    with st.expander('Dominance matrix'): st.dataframe(tables['dominance'].round(6), use_container_width=True)
with tab2:
    st.subheader('Method comparison under same data and normalization')
    ref = st.selectbox('Reference method', list(METHOD_LABELS), index=list(METHOD_LABELS).index('exptodim'), format_func=lambda x: METHOD_LABELS[x])
    comp = compare_methods(data, normalization=normalization, reference_method=ref, theta=theta, lambda_=lambda_, rho=rho, alpha=alpha, beta=beta)
    comp['method_label'] = comp['method'].map(METHOD_LABELS)
    st.dataframe(comp[['method_label','top_alternative','ranking','spearman_vs_reference','kendall_tau_vs_reference']], use_container_width=True)
    csv_download(comp, 'method_comparison.csv', 'Download method comparison')
with tab3:
    st.subheader('Sensitivity analysis')
    methods = st.multiselect('Methods', list(METHOD_LABELS), default=['classical_todim','generalized_inverse','power_todim','log_todim','exptodim'], format_func=lambda x: METHOD_LABELS[x])
    norms = st.multiselect('Normalizations', list(NORM_LABELS), default=list(NORM_LABELS), format_func=lambda x: NORM_LABELS[x])
    a,b,c = st.columns(3)
    with a:
        theta_grid = st.text_input('θ grid','0.5, 1.0, 2.0, 5.0, 10.0')
        lambda_grid = st.text_input('λ grid','1.0, 1.5, 2.25, 3.0, 5.0')
    with b:
        rho_grid = st.text_input('ρ grid','1.0, 2.0, 3.0, 5.0, 10.0')
        alpha_grid = st.text_input('α grid','0.5, 0.88, 1.0')
    with c:
        beta_grid = st.text_input('β grid','0.5, 0.88, 1.0')
        baseline = st.selectbox('Baseline method', list(METHOD_LABELS), index=list(METHOD_LABELS).index('exptodim'), format_func=lambda x: METHOD_LABELS[x])
    if st.button('Run sensitivity grid'):
        sens = sensitivity_analysis(data, methods=tuple(methods), lambdas=tuple(float(x.strip()) for x in lambda_grid.split(',') if x.strip()), rhos=tuple(float(x.strip()) for x in rho_grid.split(',') if x.strip()), thetas=tuple(float(x.strip()) for x in theta_grid.split(',') if x.strip()), alphas=tuple(float(x.strip()) for x in alpha_grid.split(',') if x.strip()), betas=tuple(float(x.strip()) for x in beta_grid.split(',') if x.strip()), normalizations=tuple(norms), baseline_method=baseline)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric('Scenarios', len(sens)); m2.metric('Top-1 stability', f'{100*sens.top_stable.mean():.1f}%'); m3.metric('Full ranking stability', f'{100*sens.full_ranking_stable.mean():.1f}%'); m4.metric('Mean Spearman', f'{sens.spearman.mean():.3f}')
        st.dataframe(sens, use_container_width=True); csv_download(sens, 'todim_family_sensitivity_results.csv', 'Download sensitivity results')
        hm = st.selectbox('Heatmap method', methods, format_func=lambda x: METHOD_LABELS[x])
        hn = st.selectbox('Heatmap normalization', norms, format_func=lambda x: NORM_LABELS[x])
        x1,x2 = st.columns(2)
        with x1: st.pyplot(heatmap(sens, hm, hn, 'spearman', 'Spearman with baseline'))
        with x2: st.pyplot(heatmap(sens, hm, hn, 'kendall_tau', 'Kendall tau with baseline'))
    else: st.info("Click 'Run sensitivity grid' to generate robustness outputs.")
with tab4:
    st.subheader('Decision data')
    st.dataframe(pd.DataFrame(data.matrix, index=data.alternatives, columns=data.criteria), use_container_width=True)
    st.dataframe(pd.DataFrame({'Criterion':data.criteria,'Weight':data.weights,'Sense':['Benefit' if s==1 else 'Cost' for s in data.senses]}), use_container_width=True)
with tab5:
    st.subheader('Method notes')
    st.markdown(r'''
### Classical TODIM
\[\Phi_k(A_i,A_j)=\sqrt{w_k(z_{ik}-z_{jk})}\] for gains and \[-\frac{1}{\theta}\sqrt{\frac{z_{jk}-z_{ik}}{w_k}}\] for losses.

### Generalized TODIM — inverse-weight loss
\[\Phi_k(A_i,A_j)=[w_k(z_{ik}-z_{jk})]^\alpha\] for gains and \[-\lambda\left[\frac{z_{jk}-z_{ik}}{w_k}\right]^\alpha\] for losses.

### Generalized TODIM — monotone/prospect
\[\Phi_k(A_i,A_j)=w_k(z_{ik}-z_{jk})^\alpha\] for gains and \[-\lambda w_k(z_{jk}-z_{ik})^\beta\] for losses.

### Power TODIM

Power TODIM uses the monotone prospect formulation:

\[
\Phi_k(A_i,A_j)=
\begin{cases}
w_k(z_{ik}-z_{jk})^\alpha, & z_{ik}>z_{jk},\\
0, & z_{ik}=z_{jk},\\
-\lambda w_k(z_{jk}-z_{ik})^\beta, & z_{ik}<z_{jk}.
\end{cases}
\]

### Logarithmic TODIM

\[
\Phi_k(A_i,A_j)=
\begin{cases}
w_k\log(1+10\rho|z_{ik}-z_{jk}|), & z_{ik}>z_{jk},\\
0, & z_{ik}=z_{jk},\\
-\lambda w_k\log(1+10\rho|z_{ik}-z_{jk}|), & z_{ik}<z_{jk}.
\end{cases}
\]

### ExpTODIM
\[\Phi_k(A_i,A_j)=w_k(1-10^{-\rho |z_{ik}-z_{jk}|})\] for gains and \[-\lambda w_k(1-10^{-\rho |z_{ik}-z_{jk}|})\] for losses.
''')
