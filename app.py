"""SANKET-Lite — generate, export, independently ingest and analyse IQ."""
from __future__ import annotations

import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import signal as sp_signal

import estimator
import signal_gen
import signal_io
import verify

st.set_page_config(page_title="SANKET-Lite", page_icon="📡", layout="wide")
st.title("SANKET-Lite | RF Signal Evidence Workbench")
st.caption("Generate a controlled RF signal, export it, then independently ingest and extract forensic parameters.")


def analyse(iq, fs, source):
    est = estimator.full_estimate(iq, fs)
    ver = verify.analysis_by_synthesis(iq, fs, est, duration=len(iq) / fs)
    return {"iq": iq, "fs": fs, "source": source, "estimate": est, "verification": ver}


def anatomy(mod):
    return {
        "AM": "Amplitude follows the message; frequency remains around the carrier offset.",
        "FM": "Instantaneous frequency swings continuously while amplitude stays nearly constant.",
        "BPSK": "Each symbol flips phase between two states; transitions create spectral sidebands.",
        "QPSK": "Symbols occupy four phase states; clusters in the constellation reveal them.",
        "2FSK": "Each binary symbol selects one of two frequency states, visible as frequency bands.",
    }.get(mod, "Hover waveform, spectrum, and IQ samples to inspect local signal evidence.")


def interactive_views(iq, fs, label, truth=None):
    n = min(len(iq), 4_000)
    time_ms = np.arange(n) / fs * 1_000
    wave = go.Figure()
    wave.add_trace(go.Scatter(x=time_ms, y=iq.real[:n], mode="lines", name="I (in-phase)",
        hovertemplate="Time: %{x:.3f} ms<br>I amplitude: %{y:.4f}<extra>I channel</extra>"))
    wave.add_trace(go.Scatter(x=time_ms, y=iq.imag[:n], mode="lines", name="Q (quadrature)",
        hovertemplate="Time: %{x:.3f} ms<br>Q amplitude: %{y:.4f}<extra>Q channel</extra>"))
    wave.update_layout(title=f"{label}: I/Q waveform (first {n} samples)", xaxis_title="Time (ms)", yaxis_title="Amplitude", height=290, margin=dict(l=20, r=20, t=45, b=20), legend=dict(orientation="h"))
    st.plotly_chart(wave, use_container_width=True)

    f, t, sxx = sp_signal.spectrogram(iq, fs=fs, nperseg=min(256, len(iq)), return_onesided=False)
    order, stride = np.argsort(f), max(1, len(t) // 250)
    spec = go.Figure(go.Heatmap(x=t[::stride] * 1_000, y=f[order] / 1_000, z=10 * np.log10(sxx[order, ::stride] + 1e-12), colorscale="Turbo", colorbar_title="dB", hovertemplate="Time: %{x:.3f} ms<br>Frequency: %{y:.2f} kHz<br>Power: %{z:.1f} dB<extra>Spectral region</extra>"))
    spec.update_layout(title=f"{label}: interactive spectrogram", xaxis_title="Time (ms)", yaxis_title="Frequency (kHz)", height=360, margin=dict(l=20, r=20, t=45, b=20))
    st.plotly_chart(spec, use_container_width=True)

    count = min(len(iq), 2_000)
    samples = go.Figure(go.Scattergl(x=iq.real[:count], y=iq.imag[:count], mode="markers", marker=dict(size=4, opacity=.55), text=[f"Sample {i}<br>Time {i/fs*1e3:.3f} ms" for i in range(count)], hovertemplate="%{text}<br>I: %{x:.4f}<br>Q: %{y:.4f}<extra>IQ sample</extra>"))
    samples.update_layout(title=f"{label}: constellation / IQ scatter", xaxis_title="I", yaxis_title="Q", height=310, margin=dict(l=20, r=20, t=45, b=20))
    samples.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(samples, use_container_width=True)
    if truth:
        st.info("**Signal anatomy:** " + anatomy(truth.get("modulation")))


def render_analysis(result):
    iq, fs, est, ver = result["iq"], result["fs"], result["estimate"], result["verification"]
    cls = est["classification"]
    st.subheader("Extraction evidence")
    a, b, c, d = st.columns(4)
    a.metric("Predicted modulation", cls["predicted_modulation"])
    b.metric("Confidence", f"{cls['confidence'] * 100:.0f}%")
    c.metric("Symbol rate", f"{est['symbol_rate_est_hz']:.0f} Hz" if est["symbol_rate_est_hz"] else "n/a")
    d.metric("Reconstruction fidelity", f"{ver['fidelity_score']:.1f}%" if ver.get("fidelity_score") is not None else "n/a")
    st.caption(f"Source: {result['source']} · sample rate: {fs:,.0f} Hz · samples: {len(iq):,}")
    if ver.get("fidelity_score") is not None and ver["fidelity_score"] < 60:
        st.warning(ver.get("verdict", "Low-fidelity result"))
    elif cls["predicted_modulation"] == "UNKNOWN":
        st.warning(ver.get("note", "Classifier declined to commit to a modulation."))
    else:
        st.success(ver.get("verdict", "Parameter extraction complete."))
    left, right = st.columns(2)
    with left:
        interactive_views(iq, fs, "Observed signal")
    with right:
        f, p = sp_signal.welch(iq, fs=fs, nperseg=min(1024, len(iq)), return_onesided=False)
        order = np.argsort(f)
        spectrum = go.Figure(go.Scatter(x=f[order]/1_000, y=10*np.log10(p[order]+1e-15), mode="lines", name="Observed"))
        if ver.get("iq_reconstructed") is not None:
            f2, p2 = sp_signal.welch(ver["iq_reconstructed"], fs=fs, nperseg=min(1024, len(iq)), return_onesided=False)
            order2 = np.argsort(f2)
            spectrum.add_trace(go.Scatter(x=f2[order2]/1_000, y=10*np.log10(p2[order2]+1e-15), mode="lines", name="Re-synthesized"))
        spectrum.update_layout(title="Observed vs re-synthesized spectrum", xaxis_title="Frequency (kHz)", yaxis_title="PSD (dB)", height=310, margin=dict(l=20,r=20,t=45,b=20))
        st.plotly_chart(spectrum, use_container_width=True)
        st.subheader("Machine-readable evidence")
        st.json({"spectral": {k: round(est[k], 2) for k in ("center_freq_offset_hz", "occupied_bw_hz", "snr_db_est")}, "classification": {"modulation": cls["predicted_modulation"], "confidence": cls["confidence"], "features": cls["features"]}, "symbol_rate_est_hz": est["symbol_rate_est_hz"], "verification": {k: ver.get(k) for k in ("fidelity_score", "spectral_similarity", "envelope_similarity", "verdict")}})
    sigmf = signal_io.analysis_sigmf_report(iq, fs, est, ver, result["source"])
    text = signal_io.text_report(est, ver, fs, result["source"])
    x, y = st.columns(2)
    x.download_button("Download SigMF evidence report (.sigmf-meta)", sigmf, "sanket_analysis.sigmf-meta", "application/json", use_container_width=True)
    y.download_button("Download analyst summary (.txt)", text, "sanket_analysis_report.txt", "text/plain", use_container_width=True)


generator_tab, extractor_tab = st.tabs(["1 · Generate & export", "2 · Upload & extract"])
with generator_tab:
    controls, preview = st.columns([1, 2])
    with controls:
        st.subheader("Synthetic signal parameters")
        with st.form("generator_form"):
            modulation = st.selectbox("Modulation", signal_gen.MODULATIONS, index=3)
            symbol_rate = st.slider("Symbol rate (Hz)", 10_000, 200_000, 50_000, step=5_000)
            freq_offset = st.slider("Carrier / frequency offset (Hz)", -400_000, 400_000, 30_000, step=10_000)
            snr_db = st.slider("SNR (dB)", 0, 30, 20)
            seed = st.number_input("Random seed", value=1, step=1)
            generate = st.form_submit_button("Generate signal", type="primary", use_container_width=True)
        if generate or "generated" not in st.session_state:
            iq, truth = signal_gen.generate(modulation, symbol_rate=symbol_rate, freq_offset=freq_offset, snr_db=snr_db, seed=int(seed))
            st.session_state.generated = {"iq": iq, "truth": truth}
        iq, truth = st.session_state.generated["iq"], st.session_state.generated["truth"]
        st.success(f"Generated {truth['modulation']} at {truth['fs']:,.0f} samples/s.")
        st.download_button("Download raw IQ (.iq, cf32_le)", signal_io.iq_bytes(iq), "sanket_signal.iq", "application/octet-stream", use_container_width=True)
        st.download_button("Download stereo I/Q WAV (.wav)", signal_io.wav_bytes(iq, truth["fs"]), "sanket_signal.wav", "audio/wav", use_container_width=True)
        st.download_button("Download SigMF capture (.zip)", signal_io.sigmf_bundle(iq, truth["fs"], truth), "sanket_capture_sigmf.zip", "application/zip", use_container_width=True)
        st.caption("SigMF bundle includes `sanket_capture.sigmf-data` and its matching `.sigmf-meta` metadata.")
    with preview:
        st.subheader("Generated-signal inspection")
        interactive_views(iq, truth["fs"], "Generated signal", truth)
        st.json(truth)

with extractor_tab:
    st.subheader("Independent capture extraction")
    st.write("Upload an exported WAV, raw `.iq` (interleaved `cf32_le`), or SigMF capture. This workflow does not reuse the generated signal in memory.")
    source_type = st.radio("Input format", ["WAV (stereo I/Q)", "Raw IQ (.iq, cf32_le)", "SigMF bundle (.zip)", "SigMF pair (.sigmf-data + .sigmf-meta)"], horizontal=True)
    uploaded = meta_upload = None
    raw_fs = None
    if source_type == "SigMF pair (.sigmf-data + .sigmf-meta)":
        one, two = st.columns(2)
        # Streamlit's browser-side extension allow-list is unreliable for
        # compound SigMF suffixes such as `.sigmf-data`.  Accept the two files
        # here and validate their names before decoding instead.
        uploaded = one.file_uploader("SigMF data (.sigmf-data)")
        meta_upload = two.file_uploader("SigMF metadata (.sigmf-meta)")
    else:
        ext = {"WAV (stereo I/Q)": ["wav"], "Raw IQ (.iq, cf32_le)": ["iq"], "SigMF bundle (.zip)": ["zip"]}[source_type]
        uploaded = st.file_uploader("Capture file", type=ext)
        if source_type == "Raw IQ (.iq, cf32_le)":
            raw_fs = st.number_input("Sample rate for raw IQ (Hz)", min_value=1_000, value=1_000_000, step=1_000, help="Raw IQ has no header; supply its capture sample rate.")
    ready = uploaded is not None and (source_type != "SigMF pair (.sigmf-data + .sigmf-meta)" or meta_upload is not None)
    if st.button("Extract parameters from uploaded capture", type="primary", disabled=not ready):
        try:
            if source_type == "SigMF pair (.sigmf-data + .sigmf-meta)":
                if not uploaded.name.lower().endswith(".sigmf-data"):
                    raise ValueError("SigMF data file must have the .sigmf-data extension.")
                if not meta_upload.name.lower().endswith(".sigmf-meta"):
                    raise ValueError("SigMF metadata file must have the .sigmf-meta extension.")
            if source_type == "WAV (stereo I/Q)": iq, fs, info = signal_io.read_wav(uploaded.getvalue())
            elif source_type == "Raw IQ (.iq, cf32_le)": iq, fs, info = signal_io.read_raw_iq(uploaded.getvalue(), raw_fs)
            elif source_type == "SigMF bundle (.zip)": iq, fs, info = signal_io.read_sigmf_zip(uploaded.getvalue())
            else: iq, fs, info = signal_io.read_sigmf(uploaded.getvalue(), meta_upload.getvalue())
            st.session_state.extracted, st.session_state.extract_info = analyse(iq, fs, uploaded.name), info
        except (ValueError, json.JSONDecodeError, OSError) as error:
            st.error(f"Could not read this capture: {error}")
    if "extracted" in st.session_state:
        st.caption(f"Detected input format: {st.session_state.extract_info['format']}")
        render_analysis(st.session_state.extracted)

st.divider()
st.caption("Demo scope: synthetic baseband IQ and SigMF `cf32_le` captures. Results are explainable DSP estimates, not a claim of live SDR capability.")
