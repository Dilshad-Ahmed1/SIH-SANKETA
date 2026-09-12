"""
SANKET-Lite: Analysis-by-Synthesis self-verification.

Core idea kept from the full design: regenerate a signal from the
estimator's own parameter estimates, then score how close the
re-synthesised signal is to the original. This gives a correctness
check with NO ground truth required at inference time -- exactly the
property that matters operationally (you never have ground truth on
a real intercept).
"""
import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import gaussian_filter1d
import signal_gen


def _spectral_similarity(iq_a, iq_b, fs):
    """Cosine similarity between the two signals' power spectra, after
    light Gaussian smoothing. Raw PSD bins can be only 1-2 bins wide for
    narrowband/tone-like signals (FSK), so a small (sub-bin-width)
    frequency estimate error can shift a spike entirely off its match
    and collapse cosine similarity even when the signals are visually
    near-identical. Smoothing trades a little precision for robustness
    to exactly that kind of estimation noise -- the fidelity score should
    track "is this the right signal shape", not "did we hit the exact bin"."""
    n = min(len(iq_a), len(iq_b))
    fa, pa = sp_signal.welch(iq_a[:n], fs=fs, nperseg=1024, return_onesided=False)
    fb, pb = sp_signal.welch(iq_b[:n], fs=fs, nperseg=1024, return_onesided=False)
    pa = gaussian_filter1d(pa, sigma=2)
    pb = gaussian_filter1d(pb, sigma=2)
    pa, pb = pa / (np.linalg.norm(pa) + 1e-12), pb / (np.linalg.norm(pb) + 1e-12)
    cos_sim = float(np.dot(pa, pb))
    return max(0.0, min(1.0, cos_sim))


def _envelope_similarity(iq_a, iq_b):
    ea, eb = np.abs(iq_a), np.abs(iq_b)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n], eb[:n]
    # compare envelope variance ratio (shape of modulation, not exact phase)
    va, vb = np.var(ea) / (np.mean(ea) ** 2 + 1e-12), np.var(eb) / (np.mean(eb) ** 2 + 1e-12)
    ratio = min(va, vb) / (max(va, vb) + 1e-12)
    return float(ratio)


def analysis_by_synthesis(iq_original, fs, estimate: dict, duration: float):
    """
    Re-synthesise a signal using the estimator's own outputs and score
    the reconstruction. Returns a 0-100 Reconstruction Fidelity Score
    plus the two signals for side-by-side plotting.
    """
    mod = estimate["classification"]["predicted_modulation"]

    if mod == "UNKNOWN":
        return {
            "fidelity_score": None,
            "note": "Signal flagged UNKNOWN by classifier — synthesis skipped. "
                    "Verification only runs against a committed hypothesis.",
            "iq_reconstructed": None,
        }

    symbol_rate = estimate.get("symbol_rate_est_hz") or 50_000.0
    freq_offset = estimate.get("center_freq_offset_hz", 0.0)

    iq_recon, _ = signal_gen.generate(
        modulation=mod,
        fs=fs,
        duration=duration,
        symbol_rate=symbol_rate,
        freq_offset=freq_offset,
        snr_db=30.0,  # synthesise clean; noise isn't part of "the model"
    )

    spec_sim = _spectral_similarity(iq_original, iq_recon, fs)
    env_sim = _envelope_similarity(iq_original, iq_recon)
    # Envelope-variance ratio is only a meaningful discriminator for
    # info-bearing envelopes (AM); for constant-envelope modulations both
    # values are near-zero and their ratio is dominated by noise, so its
    # weight is small. Spectral match carries most of the score.
    fidelity = 100.0 * (0.85 * spec_sim + 0.15 * env_sim)

    verdict = "PASS — parameters consistent with observed signal"
    if fidelity < 60:
        verdict = "LOW FIDELITY — demote to analyst review"

    return {
        "fidelity_score": round(float(fidelity), 1),
        "spectral_similarity": round(spec_sim, 3),
        "envelope_similarity": round(env_sim, 3),
        "verdict": verdict,
        "iq_reconstructed": iq_recon,
    }
