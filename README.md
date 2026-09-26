# Label-Free Uncertainty Diagnostics and Repair for Non-Cooperative TDOA Emitter Geolocation

Code, frozen models, and the artifacts behind every number reported in the paper.

A receiver network estimates **where an emitter is** from its emissions alone. The
emitter is non-cooperative, so deployment provides **no position labels** — which is
exactly what split conformal prediction needs in order to stay valid. This repository
contains the pipeline that exposes what happens when that precondition breaks, and the
label-free diagnostics that detect and partially repair it.

---

## 1. Concepts you need

The paper is four pages; this section is the background it has no room for.

**TDOA and GCC-PHAT.** Position information lives in the *relative alignment* of two
receiver signals. For each of the six receiver pairs we compute a generalized
cross-correlation with phase transform, keep lags in `±128`, and stack them into
`X ∈ R^{6×257}`. This front end is **fixed, not learned**: the network never sees raw
IQ. Keeping it fixed is what lets the time-delay Cramér–Rao lower bound remain an
analytical reference for the aleatoric covariance.

**Evidential head (NIW).** A 128-dimensional trunk feature `φ` enters a *single linear
layer* that emits a mean `μ ∈ R²`, the Cholesky factor `L` of a 2×2 scale matrix, and a
degrees-of-freedom parameter `ν = 8 + 5·tanh(·) ∈ (3,13)`. From these,

```
Σ_a = ν·LLᵀ / (ν−3)        (aleatoric)
Σ_e =   LLᵀ / (ν−3)        (epistemic)
```

Because the head is one linear layer, the path from `φ` to the predicted ellipse is an
identity, not an approximation — that is what makes the mechanism analysis in Sec. 4.3
exact rather than a post-hoc story.

**Split conformal prediction.** The nonconformity score is Mahalanobis,

```
M(X,y) = ((y−μ)ᵀ (Σ_a + Σ_e)⁻¹ (y−μ))^{1/2}
```

a scalar, so ordinary split CP applies. With `n` calibration scores and `Q` their
empirical quantile at level `⌈(n+1)(1−α)⌉ / n` (NumPy `higher` interpolation), the
region is `C_α(X) = { y : M(X,y) ≤ Q }`. It covers with probability at least `1−α`
**whenever calibration and test data are exchangeable**.

**The frozen constant.** `Q` is calibrated once on the synthetic distribution `P₀` and
then frozen. Every shifted evaluation reuses that same constant. This is the deployment
situation, not a robustness stress test: a non-cooperative emitter cannot be asked to
play back from a known position, so there is nothing to recalibrate on.

**The rule-transfer test (Sec. 4).** In distribution, the head's reported epistemic
scale is a stable function of three quality statistics of the correlation curve:

```
log σ_epi = 23.26 + 3.02·log featRMS − 1.89·log(pk/far) − 0.99·log hw
```

(`featRMS` = RMS of the windowed curve, `pk/far` = peak-to-far-sidelobe ratio,
`hw` = peak half-width; cross-validated R² = 0.54–0.60). We fit this rule in
distribution and apply it, unchanged, to shifted data. The **violation** — what the head
reports divided by what its own rule implies — needs no labels, yet it ranks conditions
the way label-dependent coverage does.

**Three label-free flags.** Each is turned into an abstention rule through a conformal
p-value, and each also serves as a *normalizer* for the same failure family:

| flag | what it reads | frozen κ |
|---|---|---|
| rule residual | signed violation of the rule above | 0.6 |
| φ-norm | upper tail of `log‖φ‖` | 1.5 |
| energy | a regression of log error on the same three curve statistics | 0.4 |

**Normalization.** `M_κ = M / exp(κ·z)`, with `z` the assigned signal standardized on
one half of the calibration split and the quantile taken on the other. Negative `z` is
set to zero for the rule residual and the φ-norm; the energy flag keeps its sign, which
is why its factor can fall below as well as rise above one.

**The region gate.** After correction we measure the area of the corrected region
against the search disc (radius 8 km, 201.06 km²) and abstain when the median exceeds
τ = 0.20. **The gate is an area rule, not a coverage certificate** — passing it does not
mean the accepted snapshots are covered at the nominal rate, and the paper reports the
accepted coverage separately for exactly that reason.

**Controls.** A sampling-clock offset up to 100 ppm is the *negative control*: it leaves
the median error unchanged, so a detector that fires on it is an alarm generator. Low
SNR is the *positive control*: it degrades the estimate through a mechanism the head
does represent.

---

## 2. What is and is not in this repository

**Included.** All source code; the six frozen model checkpoints; every artifact that
backs a number in the paper; the preregistration documents for the claims the paper
describes as preregistered; the paper source.

**Not included.** The generated datasets. `output_final/` is 85 MB and
`output_final_ood/` is 69 MB, and the raw IQ they were built from is larger still —
which is the same reason the paper gives for why no public corpus carries raw
multi-receiver IQ with position labels. Everything is regenerated deterministically
from the seeds in Sec. 5 below. Also excluded: internal working notes, superseded
experiment variants, and the ~180 artifacts from lines of work that the paper does not
cite.

---

## 3. Layout

```
rfgeo/                    core package: waveform, geometry, channel, CRLB, GCC, head, CP
scripts/                  main pipeline (generate → train → calibrate → shift → evaluate)
scripts/analysis/         the analyses that produce the reported numbers
artifacts/                every artifact cited by the paper (JSON / NPZ)
artifacts/beta_phat/      per-arm conformal summaries for the whitening ablation
output_final_der/         frozen checkpoint used throughout the paper
output_seed*/             the five training seeds behind every [min–max] range
prereg/                   preregistration documents, written before the runs
paper/                    LaTeX source
MANIFEST.md               reported number → artifact → producing script
```

`MANIFEST.md` is the entry point for verification. Every quantitative claim in the
paper appears there with the file that contains it and the script that produced it.

---

## 4. Environment

```
Python 3.14.5
torch 2.13.0+cpu, numpy, scipy
CPU only (16 threads); no GPU is required or used
```

The frozen checkpoints were trained under torch 2.5.1. `scripts/train_der.py` is
bit-deterministic across that change: retraining seed 0 under torch 2.13 reproduces all
34 parameter tensors with `max|diff| = 0`. Not every auxiliary harness in this project
shares that property, so **compare against the shipped artifacts rather than against
absolute numbers you obtain from a different environment**.

---

## 5. Reproducing the results

### 5.1 Verify the reported numbers without retraining (minutes)

The analysis scripts read the shipped artifacts and recompute the derived quantities.
For example, the Sec. 5.3 claims:

```bash
python scripts/analysis/sec53_derived_j.py
```

```
(a) in-dist union      0.8917--0.9183      # paper: coverage 0.892–0.918
(b) silent ssang       n=6 max|dcov|=0.0882 # paper: six pairs, at most 0.09
(c) fire-no-repair     n=3  det 0.862--1.000 # paper: three pairs fire at 0.86–1.00
(d) Spearman           0.800--0.976        # paper: Spearman 0.80–0.98
(e) LS 0.0132--0.0235 vs Mondrian 0.0175--0.0293
(f) gate: repair max 18.5% vs abstain min 234.9%  # paper: ≤19% vs ≥235%
```

Others in the same class: `triage_gate.py`, `snapshot_gate_j.py`, `ceiling_pairing.py`,
`sec51_mondrian_j.py`, `part12_derived.py`.

### 5.2 Regenerate the baseline (hours, CPU-only)

```bash
python scripts/gen_data.py      --out output_final --channel-mode tdl \
                                --tdl-profile TDL-D --std-pn gold
python scripts/train_der.py     --out output_final --res-out output_final_der \
                                --seed 0 --threads 16
python scripts/run_conformal.py --out output_final \
                                --der-model output_final_der/der_model.pt \
                                --res-out output_final_cp --threads 16
python scripts/run_mondrian.py
```

Defaults are train 8000 / calibration 4000 / test 2000; four receivers on an 8 km-radius
circle; `f_s = 2` MHz, carrier 1.5 GHz, SNR ~ U[−10, 20] dB; TDL-D channel with
Gold-code DSSS.

> `--channel-mode tdl --std-pn gold` must be given explicitly; the argument defaults are
> a legacy random channel. `run_mondrian.py` has its input and output paths hard-coded
> to the names above — run it from the repository root, unmodified.

### 5.3 Regenerate the shifted evaluation sets

```bash
python scripts/gen_shift.py  --out output_final_ood --n 600 --seed 777 \
                             --channel-mode tdl --tdl-profile TDL-D --std-pn gold
python scripts/eval_shift.py --der-model output_final_der/der_model.pt \
                             --calib output_final --shift-dir output_final_ood \
                             --res-out output_final_ood/shift_eval.json
```

One factor is changed at a time in graded levels, 600 snapshots per level, spanning
propagation, emitter bandwidth and receiver hardware.

### 5.4 Shift-set provenance

Filenames carry the evaluation resource. Selection and confirmation use **disjoint**
shift seeds and generation caches — that separation is the point, so the two are never
mixed within a reported number. The confirmation runs behind Table 2 use shift seeds
92001/92002 with generation sets `j1`–`j3`; the detector confirmations
(`phinorm_confirm.py`, `ruleresid_confirm.py`) use four held-out models × five
generation seeds, giving the 20-cell grids the paper reports.

`ls040_confirm.json` is produced by

```bash
python scripts/analysis/ls035_confirm.py --arm 0.40
```

The script scores a neighbouring κ under the same test; the measurement is shared with
`ls035_confirm.json` and only the scored arm differs.

---

## 6. Preregistration

`prereg/` contains the documents written **before** the corresponding runs. They fix the
measurement, the thresholds, the falsifier, and the verdict table in advance. Two points
a reader should check for themselves:

- Thresholds were not adjusted after seeing results. Where a prediction landed in a
  region the verdict table did not define, that is reported as undefined rather than
  resolved in the favourable direction.
- `κ` and the family→normalizer assignment are design-time constants, frozen before
  evaluation and confirmed on new shift sets. A sweep shows coverage still rising at the
  assigned `κ`; we did not move to it.

---

## 7. Limitations

- **All evidence is synthetic.** The forward model is known exactly, which favours
  classical estimators and understates sim-to-real shift. No field data.
- **The region gate is an area rule.** It does not certify coverage on accepted
  snapshots, and the paper reports accepted coverage separately.
- **Abstention and repair both select subsets of the stream**, so the accepted set has
  no finite-sample guarantee; selective calibration is left to future work.
- **The family→normalizer map is fixed at design time.** Choosing the normalizer
  without the family label is not evaluated.
- Absolute numbers from a different software environment may differ; verify against the
  shipped artifacts.

---

## 8. Citation and license

*(BibTeX entry to be added on acceptance.)*

This repository is released under the MIT License. See `LICENSE` for details.
