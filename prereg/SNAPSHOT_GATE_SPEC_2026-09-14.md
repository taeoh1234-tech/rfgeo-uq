# SNAPSHOT_GATE_SPEC — τ=0.20 동결 게이트의 스냅샷 단위 평가 명세 (0A)

> **작성 시점: 2026-09-14, 결과 계산 전 고정.** (PRISM 조건부 승인 0A 단계)
> 이 문서는 분석 **명세(specification)** 다. 논문에서 이 분석을 "사전등록(preregistered)"
> 이라고 부르지 않는다 — 사전등록 확증 실험이 아니라, **결과를 보기 전에 명세를 고정한
> 기술(記述) 분석**이다 (PRISM 합의 사항).

---

## 0. 범위 선언 (가장 중요)

이 분석이 재는 것은 **"사전 배정된 normalizer + 동결 κ + 설계 상수 τ=0.20" 아래에서,
스냅샷 단위 게이트(교정 영역 / 탐색 원반 ≤ τ)가 주는 acceptance rate·선택(accepted)
커버리지·선택 영역**이다.

- ✅ 이것은 **family-wise gate 평가**다: 조건(family)마다 표 2가 확증한 arm 이 배정돼
  있고, 그 배정 아래에서 게이트가 스냅샷 수준에서 어떻게 행동하는가.
- 🔴 이것은 **런타임 selector 검증이 아니다**: 배정은 조건 라벨(오퍼레이터가 모르는
  정보)로 이뤄지므로, "스냅샷 하나가 들어왔을 때 어느 arm 을 쓸지 고르는 정책"은
  이 분석의 대상이 아니며 그런 주장을 하지 않는다.
- 🔴 τ=0.20 은 이 분석이 고르는 값이 아니라 **기존 설계 상수**다
  (ls035_confirm L2/L4 게이트, sec53_derived_j R-게이트, triage_gate.py 전부 0.20).
  τ 스윕은 보고 전용이며 τ 를 재선택하지 않는다.

## 1. 자원·모델 (전부 동결, 재학습·재생성 없음)

- 모델 5개: `m0..m3` = output_seed2026{0,1,2,3}628_clean, `seed0` = output_seed0.
- 자원 j: 이동시드 92001/92002 (`output_ood_s9200X/shift_*.npz`) + 생성세트 j1~j3
  (`전처리 단계/output_rank1/cfs_cache/*_j{1,2,3}.npz` — **기존 캐시 그대로 사용**).
- 보정: `output_final/gcc_calib.npz` (n=4000). FIT = calib[0:2000], CAL = calib[2000:4000]
  — ls040_confirm·corrector_j_arms(PRIMARY)와 동일한 분할.

## 2. 조건 × 배정 대응표 (triage_gate.py ASSIGN 과 1:1, 사전 존재)

| # | 조건 | 배정 arm@κ | 하네스 | 데이터 | n/모델 | 논문 배정 |
|---|---|---|---|---|---|---|
| 1 | in-dist 저SNR bin | LS@0.40 | ls040 | IND-D j1~j3 풀링, SNR bin0 (snr<−5) | ~400 | repair |
| 2 | delay_3 (multipath) | RULE@0.6 | corrector PRIMARY | 92001+92002 풀링 | 1200 | repair |
| 3 | occ 0.05 (bandwidth) | PHI@1.5 | corrector PRIMARY | j1~j3 풀링 | 1800 | repair |
| 4 | kf_3 (fading) | LS@0.40 | ls040 | 92001+92002 | 1200 | repair |
| 5 | TDL-A | LS@0.40 | ls040 | j1~j3 | 1800 | repair |
| 6 | TDL-C | LS@0.40 | ls040 | j1~j3 | 1800 | repair |
| 7 | sco_4 (benign control) | LS@0.40 | ls040 | 92001+92002 | 1200 | repair(무개입 통과) |
| 8 | cfo_4 | LS@0.40 | ls040 | 92001+92002 | 1200 | **abstain** |
| 9 | snr_2 | LS@0.40 | ls040 | 92001+92002 | 1200 | **abstain** |

- control 행 처리: sco_4 는 **음성 대조**로서 다른 조건과 완전히 동일하게 계산한다.
  기대(명세로 고정): 높은 acceptance + accepted 커버리지 ≈ 명목. 이 기대가 깨지면
  그대로 보고한다.
- TDL-A/C 취급: 캐시 npz(j1~j3, 각 600)에서 적재. snr 필드는 in-dist bin 분할에만 사용.
- 참고 행(판정 무관): IND-D 전 구간(모든 bin) acceptance 도 기록한다.

## 3. 양(quantity)의 정의

각 하네스는 원 스크립트(ls035_confirm.py `--arm 0.40` / corrector_j_arms.py PRIMARY)의
코드 경로를 **그대로 복제**한다 (z 표준화·클리핑 규약 포함: ls 하네스의 LS z 는 클리핑
없음, corrector 하네스의 RULE/PHI z 는 ZC·ZH 모두 max(z,0) 클리핑 — 원본과 동일).

- 스냅샷별 영역: `A_i = π (Q·g_i)² · sdet_i · SCALE² / 1e6` [km²],
  `g_i = exp(κ·z_i)`, sdet = √det(Σa+Σe) (정규화 좌표), SCALE = 5000.
- 게이트 분율: `frac_i = A_i / DISC`, DISC = π·8000²/1e6 = 201.0619 km² (탐색 원반).
- **게이트: accept_i ⟺ frac_i ≤ τ = 0.20.**
- 커버리지: `hit_i ⟺ M_i / g_i ≤ Q`.
- Q: `fsq(CAL_M / g_CAL, α=0.10)` = 보정 점수의 **⌈(n+1)(1−α)⌉번째 순서통계량**
  (n=2000 → 1801번째). 확증 계열(fsq)과 동일 규약. 파이프라인
  `_finite_sample_quantile`(한 순서통계량 위)과의 차이는 **별도 stage-1 측정**
  (`quantile_convention_impact.py`)으로 정량화하며 이 분석에는 섞지 않는다.

## 4. 산출 지표 (모델 × 조건)

n, n_accept, **acc_rate**, cov_all(재현 검증 겸용), **cov_accept**, cov_reject,
med_frac_accept, p90_frac_accept, med_area_km2_accept.
집계: 5모델 **min–max + mean** (표 2 관례와 동일한 모델-범위 보고).
n_accept = 0 인 셀은 cov_accept = null 로 두고 분모를 병기한다.

## 5. 재현 관문 GATE-R (결과 해석 전 통과 필수)

새 스크립트가 접촉하는 **모든 기존 셀**을 재계산해 저장 JSON 과 대조한다.
상대오차 ≤ 1e-9 (기대: 비트 동일). 하나라도 실패하면 **abort** — 결과를 해석하지 않는다.

- ls040_confirm.json: `LS@0.40|{m}|{kf_3,cfo_4,snr_2,sco_4}` (풀링),
  `LS@0.40|{m}|{IND-D,TDL-A,TDL-C}|{j1,j2,j3}`, `LS@0.40|{m}|{j1,j2,j3}|__bins__`.
- corrector_j_arms.json: `PRIMARY|RULE|0.6|{m}|delay_3|{92001,92002}`,
  `PRIMARY|PHI|1.5|{m}|occ 0.05|{j1,j2,j3}`.

## 6. τ 스윕 (보고 전용)

τ ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50} 격자에서 acc_rate·cov_accept 를
기록한다. **τ 재선택 없음** — "τ 에 강건" 문장은 이 스윕이 실제로 지지할 때만,
지지하는 범위만큼만 쓴다 (PRISM 조건).

## 7. 해석 한계 (결과 확인 전 고정)

1. acceptance 가 낮은 조건의 cov_accept 는 분모가 작아 이항 SE 가 크다 — 분모 병기,
   n_accept < 50 인 셀의 cov_accept 는 수치 인용 금지(범위 서술만).
2. abstain 조건(cfo_4·snr_2)의 acc_rate = **누출률**(잘못 통과)로 해석한다.
3. 이 분석 결과를 selector 검증·사전등록 확증으로 승격하지 않는다 (§0).
4. accepted 커버리지에는 유한표본 보장이 **없다** (accept 는 데이터 의존 선택 —
   §15-22 의 선택조건부 문제와 같은 구조). 기술 통계로만 보고한다.

## 8. 산출물

- 코드: `scripts/data/snapshot_gate_j.py` (기존 파일 무수정, 신규만)
- 결과: `전처리 단계/output_rank1/snapshot_gate_j.json`
  + 스냅샷 배열 `snapshot_gate_j_persample.npz` (frac, hit, 인덱스)
