# SANKET-Lite

A one-day, buildable slice of the full SANKET RF-forensics vision: model-free
parameter estimation + Analysis-by-Synthesis self-verification, running on
synthetic IQ (no SDR hardware needed for the demo).

## Setup

```bash
pip install numpy scipy matplotlib streamlit
streamlit run app.py
```

## What it does

1. **`signal_gen.py`** — generates synthetic baseband IQ for AM / FM / BPSK /
   QPSK / 2FSK with known ground truth (symbol rate, freq offset, SNR).
2. **`estimator.py`** — pure classical DSP, zero training data:
   - spectral estimation (center freq, occupied bandwidth, SNR) via Welch's method
   - symbol-rate estimation via squaring/quartic-law cyclostationary nonlinearity
   - rule-based modulation classification using envelope statistics,
     instantaneous-frequency kurtosis/bimodality, and PSK-order nonlinearity strength
   - an honest `UNKNOWN` output when the top two candidates are too close to call
3. **`verify.py`** — **Analysis-by-Synthesis**: re-synthesizes a signal from the
   estimator's own outputs and scores how well it matches the original
   (smoothed-spectrum cosine similarity + envelope-shape agreement). This is a
   correctness check that needs no ground truth at inference time.
4. **`app.py`** — Streamlit evidence-card UI: spectrogram, constellation,
   observed-vs-reconstructed spectrum overlay, and the raw JSON evidence card.

## Measured performance (synthetic sweep, symbol_rate=50kHz, freq_offset=30kHz)

| SNR | Classification accuracy |
|---|---|
| 25 dB | 100% |
| 15 dB | 80% |
| 8 dB  | ~20% (degrades honestly — see note below) |
| 3 dB  | ~20% |

**Talking point, not a bug to hide:** low-SNR degradation is real physics —
envelope/frequency statistics get swamped by noise before the classical
features can separate. The full SANKET design's answer to this (a learned
detector + conformal calibration) is explicitly out of scope for a one-day
build; say so on the slide. Judges respond better to an honest limitations
slide than to a claimed 99% that falls apart under a live low-SNR demo.

## What's deliberately cut from the full SANKET design (say this on your slide)

- Blind raw-IQ / headerless format inference (kept to a known baseband model)
- CNN/learned signal detector (using classical energy-threshold logic instead)
- Open-set calibration / conformal prediction (using a simple margin-based
  UNKNOWN gate instead — same idea, no training)
- LLM/RAG report generation
- Real SDR hardware capture (synthetic IQ only)
- Cryptographic evidence signing / SigMF export

## Suggested demo flow (3 minutes)

1. Show a clean BPSK signal at 25 dB — correct classification, high fidelity score.
2. Drop SNR to 8 dB live — watch it degrade to `UNKNOWN` or the wrong label,
   and explain *why* the DSP breaks down (this is your credibility moment).
3. Show the Analysis-by-Synthesis overlay plot — "the tool checks its own work."
4. One slide: "here's what we'd add with more than a day" (the cut list above).

## Files

- `signal_gen.py` — synthetic signal generator
- `estimator.py` — spectral + cyclostationary + classification
- `verify.py` — Analysis-by-Synthesis fidelity scoring
- `app.py` — Streamlit demo UI
