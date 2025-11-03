# roi_montecarlo_app.py
# Monte Carlo ROI for Rasa Legal
# Formula: ROI = (People per year × $ per person per year × Years) ÷ Cost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------- Helpers --------------
def lognormal_params_from_mean_sd(mean, sd):
    """Return mu, sigma for the underlying normal of a lognormal with given mean and sd."""
    if mean <= 0:
        raise ValueError("Mean must be > 0 for lognormal.")
    if sd <= 0:
        sigma = 1e-9
        mu = np.log(mean)
        return mu, sigma
    sigma2 = np.log(1 + (sd**2)/(mean**2))
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5*sigma2
    return mu, sigma

def draw_lognormal(mean, sd, n, z=None):
    mu, sigma = lognormal_params_from_mean_sd(mean, sd)
    if z is None:
        z = np.random.normal(size=n)
    return np.exp(mu + sigma*z)

def draw_trunc_normal(mean, sd, n, z=None):
    if z is None:
        z = np.random.normal(size=n)
    return np.maximum(0.0, mean + sd*z)

def correlated_normals(n, rho):
    z1 = np.random.normal(size=n)
    z2 = np.random.normal(size=n)
    z2c = rho*z1 + np.sqrt(max(0.0, 1 - rho**2))*z2
    return z1, z2c

def summarize(series):
    s = np.asarray(series)
    s = s[~np.isnan(s)]
    return {
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "p25": float(np.percentile(s, 25)),
        "p75": float(np.percentile(s, 75)),
        "p05": float(np.percentile(s, 5)),
        "p95": float(np.percentile(s, 95)),
    }

def rank_corr(x, y):
    xr = pd.Series(x).rank(method="average")
    yr = pd.Series(y).rank(method="average")
    return float(np.corrcoef(xr, yr)[0,1])

# -------------- UI --------------
st.set_page_config(page_title="Monte Carlo ROI (Rasa Legal)", layout="centered")
st.title("Monte Carlo ROI – Rasa Legal")
st.caption("Set SDs for key variables. See ROI interquartile range (p25 to p75).")

with st.sidebar:
    st.header("Simulation")
    n_draws = st.number_input("Number of draws", min_value=1000, max_value=500000, value=20000, step=1000)
    dist = st.selectbox("Distribution for P and D", ["Lognormal (recommended)", "Truncated Normal (non-negative)"])
    rho = st.slider("Correlation between P and D", min_value=-0.9, max_value=0.9, value=0.0, step=0.05)
    sd_scale = st.slider("SD scale multiplier", min_value=0.25, max_value=3.0, value=1.0, step=0.05)
    show_sweep = st.checkbox("Show how IQR changes vs SD scale", value=False)
    if show_sweep:
        sweep_min = st.number_input("Sweep: min SD scale", 0.25, 10.0, 0.5, 0.05)
        sweep_max = st.number_input("Sweep: max SD scale", 0.25, 10.0, 2.0, 0.05)
        sweep_steps = st.number_input("Sweep steps", 3, 25, 7, 1)

st.subheader("Inputs")
col1, col2 = st.columns(2)
with col1:
    P_mean = st.number_input("People reached per year mean (P_mean)", min_value=0.0, value=10000.0, step=100.0, format="%.4f")
    P_sd   = st.number_input("People reached per year SD (P_sd)", min_value=0.0, value=2000.0, step=50.0, format="%.4f")
    Years  = st.number_input("Years", min_value=0.0, value=5.0, step=0.5, format="%.2f")
with col2:
    D_mean = st.number_input("Depth of impact $/person/year mean (D_mean)", min_value=0.0, value=50.0, step=1.0, format="%.4f")
    D_sd   = st.number_input("Depth of impact $/person/year SD (D_sd)", min_value=0.0, value=20.0, step=1.0, format="%.4f")
    Cost   = st.number_input("Cost of the investment ($)", min_value=0.0, value=1000000.0, step=1000.0, format="%.2f")

run = st.button("Run simulation")

def run_sim(P_mean, P_sd, D_mean, D_sd, Years, Cost, n_draws, rho, sd_scale, dist):
    z1, z2c = correlated_normals(n_draws, rho)
    P_sd_eff = P_sd * sd_scale
    D_sd_eff = D_sd * sd_scale

    if dist.startswith("Lognormal"):
        P = draw_lognormal(P_mean, P_sd_eff, n_draws, z=z1)
        D = draw_lognormal(D_mean, D_sd_eff, n_draws, z=z2c)
    else:
        P = draw_trunc_normal(P_mean, P_sd_eff, n_draws, z=z1)
        D = draw_trunc_normal(D_mean, D_sd_eff, n_draws, z=z2c)

    denom = Cost if Cost > 0 else np.nan
    ROI = (P * D * Years) / denom

    roi_stats = summarize(ROI)
    p_stats = summarize(P)
    d_stats = summarize(D)

    sens_P = rank_corr(P, ROI)
    sens_D = rank_corr(D, ROI)

    return ROI, P, D, roi_stats, p_stats, d_stats, sens_P, sens_D

if run:
    ROI, P, D, roi_stats, p_stats, d_stats, sens_P, sens_D = run_sim(
        P_mean, P_sd, D_mean, D_sd, Years, Cost, n_draws, rho, sd_scale, dist
    )

    st.markdown("### ROI summary")
    df_sum = pd.DataFrame({
        "metric": ["mean","median","p25","p75","p05","p95"],
        "ROI": [roi_stats[k] for k in ["mean","median","p25","p75","p05","p95"]]
    })
    st.dataframe(df_sum, use_container_width=True)
    st.markdown(f"**Interquartile range (IQR):** {roi_stats['p25']:.4f} to {roi_stats['p75']:.4f}")

    st.markdown("### Sensitivity (Spearman rank correlation with ROI)")
    st.write(f"People reached (P) vs ROI: {sens_P:.3f}")
    st.write(f"Depth $/person/year (D) vs ROI: {sens_D:.3f}")

    fig = plt.figure(figsize=(7,4))
    plt.hist(ROI[~np.isnan(ROI)], bins=60)
    plt.xlabel("ROI")
    plt.ylabel("Frequency")
    plt.title("Distribution of ROI")
    st.pyplot(fig)

    with st.expander("Input draw summaries"):
        st.write("P draws:", p_stats)
        st.write("D draws:", d_stats)

    if show_sweep:
        scales = np.linspace(sweep_min, sweep_max, int(sweep_steps))
        p25s, p75s = [], []
        for s in scales:
            ROI_s, *_ = run_sim(P_mean, P_sd, D_mean, D_sd, Years, Cost, max(2000, n_draws//5), rho, s, dist)
            s_clean = ROI_s[~np.isnan(ROI_s)]
            p25s.append(np.percentile(s_clean, 25))
            p75s.append(np.percentile(s_clean, 75))
        fig2 = plt.figure(figsize=(7,4))
        plt.plot(scales, p25s, marker='o', label='p25')
        plt.plot(scales, p75s, marker='o', label='p75')
        plt.xlabel("SD scale")
        plt.ylabel("ROI")
        plt.title("IQR vs SD scale")
        plt.legend()
        st.pyplot(fig2)
