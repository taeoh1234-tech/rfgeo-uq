# Manifest — every reported number maps to a file

Format: claim (paper location) -> artifact -> script that produced it.

### Table 1 rows 1-3 (0.9925/2.73, 0.9020/0.43, 0.9085/0.33/0.903/0.011)
*Sec. 3*
- `artifacts/conformal.json`
- `artifacts/mondrian.json`
- produced by: `scripts/run_conformal.py`, `scripts/run_mondrian.py`

### Table 1 row 4 and the 13 recalibrated configurations (0.894-0.909)
*Sec. 3*
- `artifacts/tdl_mixed_bias.json`
- produced by: `scripts/analysis/tdl_mixed_bias.py`

### beta-PHAT ablation over five seeds (Delta approx 0.07, p<0.03)
*Sec. 2.2*
- `artifacts/beta_phat/*_conformal.json` (16 arms)
- produced by: `scripts/gen_data.py --beta ... + scripts/run_conformal.py`

### Rule coefficients 23.26 / 3.02 / -1.89 / -0.99; cross-validated R^2 0.54-0.60
*Sec. 4.1, Eq. (3)*
- `artifacts/epi_mapping_transfer.json`
- produced by: `scripts/analysis/epi_mapping_transfer.py`, `scripts/analysis/gcc_curve_atlas.py`, `scripts/analysis/make_fig2_ruletransfer_band.py`

### Rule-transfer ratios 0.96 / 1.66 / 0.33 / 0.28 / 0.34 / 0.71 / 0.08; Spearman 0.74 and 0.85
*Sec. 4.2, Fig. 2*
- `artifacts/epi_mapping_transfer.json`
- `artifacts/gcc_curve_atlas.json`
- `artifacts/part12_derived.json`
- produced by: `scripts/analysis/epi_mapping_transfer.py`, `scripts/analysis/gcc_curve_atlas.py`, `scripts/analysis/make_fig2_ruletransfer_band.py`, `scripts/analysis/part12_derived.py`

### Feature-norm mechanism: -1.10..-1.15, 1.10-1.32x, 0.75-0.91x over five seeds
*Sec. 4.3*
- `artifacts/mechanism_norm_vs_direction.json`
- `artifacts/seed_replication_mechanism.json`
- produced by: `scripts/analysis/mechanism_norm_vs_direction.py`, `scripts/analysis/seed_replication_mechanism.py`

### beta* = 28-47% at the strongest multipath level (mean 0.38)
*Sec. 5.1 and Table 2*
- `artifacts/ceiling_pairing.json`
- produced by: `scripts/analysis/ceiling_pairing.py`

### Group-conditional lift 0.157->0.661 versus 0.142->0.199
*Sec. 5.1*
- `artifacts/sec51_mondrian_j.json`
- produced by: `scripts/analysis/sec51_mondrian_j.py`

### Table 2, detection columns (all cells, five seeds)
*Table 2*
- `artifacts/detector_j.json`
- produced by: `scripts/analysis/detector_j.py`, `scripts/analysis/sec53_derived_j.py`

### Table 2, repair columns (coverage before->after, region fraction)
*Table 2, Sec. 5.3*
- `artifacts/corrector_j_arms.json`
- `artifacts/ls040_confirm.json`
- `artifacts/sec53_derived_j.json`
- produced by: `scripts/analysis/corrector_j_arms.py`, `scripts/analysis/ls035_confirm.py --arm 0.40`, `scripts/analysis/sec51_mondrian_j.py`, `scripts/analysis/sec53_derived_j.py`, `scripts/analysis/snapshot_gate_j.py`, `scripts/analysis/triage_gate.py`

### Region gate: <=19% versus >=235%; accepted coverage 0.53-0.58; carrier offset 21% at 0.44
*Sec. 5.3*
- `artifacts/triage_gate.json`
- `artifacts/snapshot_gate_j.json`
- `artifacts/snapshot_gate_j_persample.npz`
- produced by: `scripts/analysis/snapshot_gate_j.py`, `scripts/analysis/triage_gate.py`

### Classical integrity controls: 0.80-0.92, at most 0.52, repair at most 0.62
*Sec. 5.3*
- `artifacts/detector_j_sqm.json`
- `artifacts/corrector_j_sqm.json`
- `artifacts/corrector_clo_arm.json`
- produced by: `scripts/analysis/corrector_clo_arm.py`, `scripts/analysis/corrector_j_sqm.py`, `scripts/analysis/detector_j_sqm.py`

### Confirmation: 20 of 20 cells; six pre-registered predictions
*Sec. 5.2*
- `artifacts/phinorm_confirm.json`
- `artifacts/ruleresid_confirm.json`
- produced by: `scripts/analysis/phinorm_confirm.py`, `scripts/analysis/ruleresid_confirm.py`

### Classical baseline: SRP-PHAT 7.9 m versus 150.3 m
*Sec. 1*
- `artifacts/classical_corrected.json`
- `artifacts/srp_interp_validation.json`
- produced by: `scripts/analysis/classical_corrected.py`, `scripts/analysis/srp_interp_validation.py`

### 20 m receiver-survey error: coverage 0.887->0.523 while AUROC stays 0.50
*Sec. 1*
- `artifacts/srp_silent_failure.json`
- produced by: `scripts/analysis/srp_silent_failure.py`
