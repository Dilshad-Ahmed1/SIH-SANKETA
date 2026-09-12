# SANKET-Lite: Essential Features to Implement Before SIH 2026

This document lists the minimum features that are still missing from the current implementation and are essential for a convincing SIH 2026 demo and pitch.

## Executive summary

The current project is a strong synthetic DSP MVP, but it is not yet a complete RF-forensics solution. It proves the idea well, but it does not yet cover the real-world problems judges will likely test:

- unknown file formats and raw IQ inputs
- multi-signal captures
- real capture health checks
- honest uncertainty / UNKNOWN handling
- evidence-grade reporting
- deployment-style usability

For SIH 2026, we should not chase every advanced feature in the grand design. We must focus on the features that make the project look credible, defensible, and technically complete enough for judges.

---

## Priority model

### P0 = Must-have before demo
These are essential.

### P1 = Very valuable, should also be included if time allows
These improve credibility and make the project feel mature.

### P2 = Future scope / not needed for SIH 2026
These are nice to have later, not required for the first judged build.

---

## P0: Must-have features missing from the current project

### 1) Blind format inference for raw capture files
Status: Missing

Why it matters:
- Real RF files are usually not clean, labelled IQ arrays
- Judges expect the tool to ingest raw captures and explain what it found
- This is one of the strongest “real-world” differentiators in the project vision

Must include:
- support for `.iq`, `.bin`, `.dat`, `.wav` inputs
- detection of sample type: int8, uint8, int16, float32
- detection of interleaving / channel order
- sample-rate estimation or plausibility checks
- metadata sniffing for WAV / SigMF
- ranked parse hypotheses with reasons

Minimum output:
- “Likely format: int16 interleaved IQ, sample rate ~2.048 MS/s, confidence: high”

---

### 2) Multi-signal detection and separation
Status: Missing

Why it matters:
- A real capture rarely contains just one clean emitter
- Demonstrating multiple signals in one file makes the project feel like a real RF forensics engine
- This is critical for matching the project vision

Must include:
- spectrogram-based detection
- time-frequency region detection using energy threshold or CFAR
- bounding-box style separation of signals
- per-signal cropping / extraction
- per-signal processing pipeline

Minimum output:
- a capture with multiple emitters gets segmented into multiple signal objects
- each signal is analysed independently

---

### 3) Capture health / quality diagnostics
Status: Missing

Why it matters:
- A tool that never checks whether the capture is valid looks weak
- Real readers expect a signal-quality gate before trusting parameter estimates
- This improves credibility and makes the system look forensic, not just statistical

Must include:
- DC offset estimation and correction
- clipping / saturation detection
- noisy or distorted capture warning
- IQ imbalance detection
- spur detection
- overall quality score that can suppress bad results

Minimum output:
- “Capture health: acceptable / degraded / unreliable”

---

### 4) Honest uncertainty handling and UNKNOWN logic
Status: Present but underdeveloped

Why it matters:
- Real judges like tools that know when they are unsure
- A system that always gives a label looks brittle
- “I don’t know” is a major differentiator in RF analysis

Must include:
- improved gate for ambiguous classifications
- calibration of confidence scores
- out-of-distribution detection for unseen signals
- “UNKNOWN” not just as a fallback, but as a real operational output

Minimum output:
- if the evidence is weak: output `UNKNOWN` with nearest candidates and low confidence

---

### 5) Broader but still realistic modulation coverage
Status: Missing / limited

Why it matters:
- The current project covers only a few classes
- For a SIH demo, you need at least enough diversity to show seriousness, not a toy dataset

Minimum realistic set for SIH 2026:
- AM
- FM
- BPSK
- QPSK
- 2FSK
- 4FSK or GMSK
- maybe 8PSK or QAM for a stronger demo

Must include:
- better feature separation between modulations
- robust output for each class
- documentation of failure modes at low SNR

---

### 6) Better parameter extraction than the current toy implementation
Status: Partial

Why it matters:
- The project is supposed to be about signal parameter extraction, not only modulation labels
- The judges want measurable outputs

Must include:
- center frequency estimate
- occupied bandwidth
- SNR estimate
- symbol rate estimate
- modulation type
- uncertainty estimate for each parameter

Minimum output example:
- center_freq = 30.1 kHz offset
- bw = 98 kHz
- snr = 18.3 dB
- symbol_rate = 50.2 kHz
- modulation = QPSK

---

### 7) Stronger Analysis-by-Synthesis verification
Status: Present, but too basic

Why it matters:
- This is the best technical differentiator in the current project
- It should be strengthened and made more convincing for judges

Must include:
- reconstruction from estimated parameters
- comparison using multiple criteria, not only spectrum + envelope
- a clearly explained fidelity score
- low-fidelity cases flagged for analyst review

Minimum output:
- “Fidelity score: 94.2% — parameters consistent with observed signal”
- or “Low fidelity: demote to analyst review”

---

### 8) Output package suitable for analyst use
Status: Missing

Why it matters:
- A project with only a UI looks like a prototype
- For SIH, we need evidence-like outputs and machine-readable fields

Must include:
- JSON evidence file with parameters
- result export in a structured format
- downloadable report summary
- raw evidence card for each signal

Minimum output:
- `result.json` with modulation, bandwidth, symbol rate, SNR, fidelity, confidence, unknown status

---

## P1: Very valuable features to add if time permits

### 9) Real SDR / real capture support
Status: Missing

Why it matters:
- A live capture demo is a huge credibility boost
- It answers the common objection: “does this work only on synthetic data?”

Must include:
- RTL-SDR / HackRF / Pluto input support
- WAV-based IQ support
- real capture workflow

---

### 10) SigMF / metadata support
Status: Missing

Why it matters:
- It makes the tool look standards-aligned
- Helps with reproducibility and forensic traceability

Must include:
- read SigMF metadata if present
- write SigMF-compatible annotations/results

---

### 11) CLI / batch mode
Status: Missing

Why it matters:
- Demoing one file in a UI is okay, but a toolchain must also work in batch or CLI mode
- This shows product maturity

Minimum output:
- `python app.py ...` or `sanket analyze file.iq --json`

---

### 12) Offline report generation
Status: Missing

Why it matters:
- A signed or structured report makes the project feel more like a forensic system than a research demo

Minimum output:
- short PDF or text report with the signal summary and evidence

---

## P2: Future-scope features, not essential for the first SIH build

- local LLM / narrative generator
- spectrum-allocation / India frequency-check layer
- full conformal prediction framework
- deep learning detector
- full SigMF ecosystem integration
- real-time streaming ingestion
- edge hardware deployment
- radar / pulsed-signal specialized modules

These are strategic and interesting, but they should not distract from the minimum viable judge-ready system.

---

## Recommended minimum essential stack for SIH 2026

If we want the project to be impressive and credible with limited time, the essential minimum should be:

1. Raw file ingestion for `.iq` / `.wav` / common binary formats
2. Capture health analysis
3. Multi-signal detection and segmentation
4. Single-signal parameter estimation for at least 5–6 modulation classes
5. Honest UNKNOWN handling with confidence and uncertainty
6. Stronger Analysis-by-Synthesis score
7. Structured evidence export in JSON
8. A polished demo UI with clear plots and metrics
9. CLI or batch-run mode
10. A polished pitch explaining that this is a real RF forensic tool, not just a modulation classifier

This is the smallest set that makes the system look serious enough for SIH 2026.

---

## Final recommendation

Do not try to implement the full grand architecture now.

The right strategy is:
- keep the current DSP/mathematical foundation
- add the missing real-world pipeline features above
- make the demo look like an evidence-first RF triage engine
- be honest about limitations, but show a complete story end-to-end

In short: the project must feel like a forensic tool, not a lab script.

That means the most important gaps are:
- raw-input handling
- multi-signal detection
- health checks
- uncertainty
- structured evidence and output

These are the essential missing features we must implement before SIH 2026.
