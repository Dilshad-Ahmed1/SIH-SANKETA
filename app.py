"""
SANKET-Lite demo UI.
Run with: streamlit run app.py
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import signal as sp_signal

import signal_gen
import estimator
import verify

st.set_page_config(page_title="SANKET-Lite", layout="wide")
st.title("SANKET-Lite — RF Signal Fingerprinting (MVP)")
st.caption(
    "Model-free parameter estimation + Analysis-by-Synthesis self-verification. "
    "No training data, no hardware — synthetic IQ in, evidence card out."
)

with st.sidebar:
    st.header("Signal source")
    modulation = st.selectbox("True modulation (ground truth)", signal_gen.MODULATIONS)
    symbol_rate = st.slider("Symbol rate (Hz)", 10_000, 200_000, 50_000, step=5_000)
    freq_offset = st.slider("Carrier / freq offset (Hz)", -400_000, 400_000, 30_000, step=10_000)
    snr_db = st.slider("SNR (dB)", 0, 30, 20)
    seed = st.number_input("Random seed", value=1, step=1)
    run = st.button("Generate & Analyze", type="primary", use_container_width=True)

if not run and "last_result" not in st.session_state:
    st.info("Set parameters in the sidebar and click **Generate & Analyze**.")
    st.stop()

if run:
    iq, gt = signal_gen.generate(
        modulation=modulation, symbol_rate=symbol_rate, freq_offset=freq_offset,
        snr_db=snr_db, seed=int(seed),
    )
    fs = gt["fs"]
    est = estimator.full_estimate(iq, fs)
    ver = verify.analysis_by_synthesis(iq, fs, est, duration=0.02)
    st.session_state["last_result"] = dict(iq=iq, gt=gt, fs=fs, est=est, ver=ver)

r = st.session_state["last_result"]
iq, gt, fs, est, ver = r["iq"], r["gt"], r["fs"], r["est"], r["ver"]
cls = est["classification"]

# ---- top-line evidence card ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted modulation", cls["predicted_modulation"], f"true: {gt['modulation']}")
c2.metric("Confidence", f"{cls['confidence']*100:.0f}%")
c3.metric("Symbol rate est.", f"{est['symbol_rate_est_hz']:.0f} Hz" if est["symbol_rate_est_hz"] else "—",
          f"true: {gt['symbol_rate']:.0f} Hz" if gt["symbol_rate"] else "n/a")
c4.metric("Reconstruction fidelity", f"{ver['fidelity_score']:.0f}%" if ver["fidelity_score"] is not None else "—")

if cls["predicted_modulation"] == "UNKNOWN":
    st.warning("Classifier declined to commit to a label — top candidates were too close to call. "
               "This is intentional: an honest UNKNOWN beats a confident wrong answer.")
elif ver["fidelity_score"] is not None and ver["fidelity_score"] < 60:
    st.warning(f"Low reconstruction fidelity ({ver['fidelity_score']:.0f}%) — {ver['verdict']}")
else:
    st.success(ver.get("verdict", ""))

st.divider()

# ---- plots ----
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Spectrogram (observed signal)")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    f, t, Sxx = sp_signal.spectrogram(iq, fs=fs, nperseg=256, return_onesided=False)
    order = np.argsort(f)
    ax.pcolormesh(t * 1000, f[order] / 1000, 10 * np.log10(Sxx[order] + 1e-12), shading="auto")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (kHz)")
    st.pyplot(fig, use_container_width=True)

    st.subheader("Power spectrum: observed vs re-synthesized")
    fig2, ax2 = plt.subplots(figsize=(5, 3.2))
    fa, pa = sp_signal.welch(iq, fs=fs, nperseg=1024, return_onesided=False)
    oa = np.argsort(fa)
    ax2.semilogy(fa[oa] / 1000, pa[oa], label="observed")
    if ver.get("iq_reconstructed") is not None:
        fb, pb = sp_signal.welch(ver["iq_reconstructed"], fs=fs, nperseg=1024, return_onesided=False)
        ob = np.argsort(fb)
        ax2.semilogy(fb[ob] / 1000, pb[ob], label="re-synthesized", alpha=0.7)
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("PSD")
    ax2.legend()
    st.pyplot(fig2, use_container_width=True)

with col_right:
    st.subheader("Constellation / IQ scatter (first 2000 samples)")
    fig3, ax3 = plt.subplots(figsize=(5, 3.2))
    seg = iq[:2000]
    ax3.scatter(seg.real, seg.imag, s=2, alpha=0.4)
    ax3.set_xlabel("I")
    ax3.set_ylabel("Q")
    ax3.axis("equal")
    st.pyplot(fig3, use_container_width=True)

    st.subheader("Evidence card (raw estimator output)")
    st.json({
        "spectral": {
            "center_freq_offset_hz": round(est["center_freq_offset_hz"], 1),
            "occupied_bw_hz": round(est["occupied_bw_hz"], 1),
            "snr_db_est": round(est["snr_db_est"], 1),
        },
        "classification": {
            "predicted_modulation": cls["predicted_modulation"],
            "confidence": cls["confidence"],
            "features": {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in cls["features"].items()},
        },
        "symbol_rate_est_hz": est["symbol_rate_est_hz"],
        "verification": {
            "fidelity_score": ver.get("fidelity_score"),
            "spectral_similarity": ver.get("spectral_similarity"),
            "envelope_similarity": ver.get("envelope_similarity"),
        },
    })


