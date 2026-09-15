"""
patch_waveform.py — waveform.py 에 '추가'할 코드 블록.

목적
----
1) DSSS의 PN 시퀀스를 무작위(rng.choice)에서 표준 코드(m-sequence/Gold)로
   교체할 수 있게 한다.  (STEP 결론: PN 종류는 Brms/CRLB에 무관 → '공짜' 방어강화)
2) 확산율(spread_factor)은 자유/고정 변수로 유지 (Brms 무관, 검증됨).
3) sps 를 파형 전체에서 통일할 수 있는 경로를 제공하여
   'sps 파형별 상이(현행)' vs 'sps=2 통일' 데이터셋을 각각 생성/비교 가능케 한다.

설계 원칙
---------
- 기존 gen_random_modulation/gen_dsss/gen_fhss/gen_burst 는 '그대로 둔다'
  (byte 재현성/기존 결과 보존).  여기서는 표준 PN 유틸과, PN 종류를 인자로 받는
  gen_dsss_std 만 추가한다.
- generate_waveform 디스패처가 kwargs 를 그대로 흘려보내므로,
  dataset 쪽에서 pn_kind / sps 를 주입하면 자동 반영된다.

로컬 적용 방법
--------------
아래 블록을 rfgeo/waveform.py 끝에 붙여넣거나,
scripts 에서 `from rfgeo import waveform as wf; wf.register_std_dsss()` 호출.
(register_std_dsss 는 WAVEFORM_GENERATORS["dsss"] 를 표준 PN 버전으로 교체)
"""
import numpy as np


# ---------------------------------------------------------------------------
# 1) 표준 PN 코드 생성기: m-sequence(최대길이) 및 Gold code
# ---------------------------------------------------------------------------
# LFSR 피드백 탭(원시다항식). 인덱스는 1-based 탭 위치.
# 참고: Proakis & Salehi; Sarwate & Pursley 1980 (Gold/m-sequence 상관특성).
_PRIMITIVE_TAPS = {
    5:  [5, 2],          # x^5 + x^2 + 1
    6:  [6, 1],          # x^6 + x + 1
    7:  [7, 3],          # x^7 + x^3 + 1
    8:  [8, 6, 5, 1],    # x^8 + x^6 + x^5 + x + 1
    9:  [9, 5],          # x^9 + x^5 + 1
    10: [10, 7],         # x^10 + x^7 + 1
}
# Gold 코드용 '선호쌍(preferred pair)' 두 번째 다항식(동일 차수, 상호상관 3-valued).
_PREFERRED_PAIR_2 = {
    5:  [5, 4, 3, 2],    # 선호쌍 상대 다항식(차수 5)
    6:  [6, 5, 2, 1],
    7:  [7, 3, 2, 1],
    9:  [9, 6, 4, 3],
    10: [10, 8, 3, 2],
}


def _lfsr_mseq(taps, n_bits, seed=None):
    """LFSR 로 최대길이 시퀀스(길이 2^n_bits - 1) 생성, ±1 반환."""
    reg = np.ones(n_bits, dtype=np.int8) if seed is None else np.array(seed, np.int8)
    if reg.size != n_bits:
        reg = np.ones(n_bits, dtype=np.int8)
    length = (1 << n_bits) - 1
    out = np.empty(length, dtype=np.int8)
    tap_idx = [t - 1 for t in taps]
    for i in range(length):
        out[i] = reg[-1]
        fb = 0
        for t in tap_idx:
            fb ^= reg[t]
        reg = np.roll(reg, 1)
        reg[0] = fb
    return 1 - 2 * out  # {0,1} -> {+1,-1}


def _nearest_bits(spread_factor):
    """spread_factor 를 담을 수 있는 최소 LFSR 비트수(코드길이>=spread_factor)."""
    for nb in sorted(_PRIMITIVE_TAPS):
        if (1 << nb) - 1 >= spread_factor:
            return nb
    return max(_PRIMITIVE_TAPS)


def pn_sequence(spread_factor, kind="gold", rng=None, phase=None):
    """
    길이 spread_factor 의 ±1 PN 칩 시퀀스를 표준 코드에서 잘라 반환.

    kind : "mseq" | "gold" | "random"
      - random 은 기존 동작(rng.choice ±1)과 동일 (하위호환/대조군).
    rng  : random kind 또는 Gold 위상 선택에 사용.
    phase: Gold/mseq 순환 위상 오프셋(정수). None 이면 rng 로 뽑음.
    """
    if kind == "random":
        r = rng if rng is not None else np.random.default_rng()
        return r.choice(np.array([-1, 1], dtype=np.int8), size=spread_factor)

    nb = _nearest_bits(spread_factor)
    m1 = _lfsr_mseq(_PRIMITIVE_TAPS[nb], nb)
    if kind == "mseq":
        base = m1
    elif kind == "gold":
        taps2 = _PREFERRED_PAIR_2.get(nb, _PRIMITIVE_TAPS[nb])
        m2 = _lfsr_mseq(taps2, nb)
        # Gold 코드 = m1 XOR (shift(m2)). ±1 도메인에서는 원소곱.
        if phase is None:
            r = rng if rng is not None else np.random.default_rng()
            phase = int(r.integers(0, m2.size))
        base = m1 * np.roll(m2, phase)
    else:
        raise ValueError(f"unknown pn kind '{kind}'")

    if phase is None:
        r = rng if rng is not None else np.random.default_rng()
        phase = int(r.integers(0, base.size))
    seq = np.roll(base, phase)[:spread_factor]
    if seq.size < spread_factor:  # 코드가 짧으면 순환 반복
        reps = int(np.ceil(spread_factor / seq.size))
        seq = np.tile(seq, reps)[:spread_factor]
    return seq.astype(np.int8)


# ---------------------------------------------------------------------------
# 2) 표준 PN 을 쓰는 DSSS (기존 gen_dsss 는 보존; 이건 신규)
# ---------------------------------------------------------------------------
def gen_dsss_std(n, fs, rng, spread_factor=16, sps=2, order=4,
                 pn_kind="gold", _rrc=None, _sym=None):
    """
    표준 PN(Gold/m-sequence) DSSS.  대역폭 결정요소(sps, spread_factor)는
    기존 gen_dsss 와 동일 경로 → Brms/CRLB 동일(검증됨). 바뀌는 것은 칩 부호규칙뿐.

    _rrc/_sym: 내부 주입용(테스트). 기본은 waveform 모듈의 함수 사용.
    """
    from rfgeo import waveform as _wf  # 실제 파이프라인 함수 재사용(재구성 아님)
    rrc = _rrc or _wf._rrc_taps
    symfn = _sym or _wf._random_symbols
    norm = _wf._normalize_power

    chip_rate_samples = sps
    n_chips = n // chip_rate_samples + spread_factor
    n_sym = n_chips // spread_factor + 1
    data = symfn(n_sym, order, rng)
    # ★ 유일한 변경점: 무작위 PN → 표준 PN. 심볼마다 위상만 달리한 Gold 코드.
    pn = np.stack([pn_sequence(spread_factor, kind=pn_kind, rng=rng)
                   for _ in range(n_sym)], axis=0)
    chips = (data[:, None] * pn).reshape(-1)
    up = np.repeat(chips, chip_rate_samples)[:n]
    if up.size < n:
        up = np.pad(up, (0, n - up.size))
    taps = rrc(0.35, sps)
    sig = np.convolve(up, taps, mode="same")[:n]
    return norm(sig)


# ---------------------------------------------------------------------------
# 3) 등록 헬퍼: WAVEFORM_GENERATORS 를 표준 PN / 통일 sps 로 스위칭
# ---------------------------------------------------------------------------
def register_std_dsss(pn_kind="gold"):
    """
    rfgeo.waveform.WAVEFORM_GENERATORS['dsss'] 를 표준 PN 버전으로 교체.
    되돌리려면 restore_random_dsss() 호출.
    """
    from rfgeo import waveform as _wf
    if not hasattr(_wf, "_ORIG_DSSS"):
        _wf._ORIG_DSSS = _wf.WAVEFORM_GENERATORS["dsss"]
    def _dsss_std(n, fs, rng, **kw):
        kw.setdefault("pn_kind", pn_kind)
        return gen_dsss_std(n, fs, rng, **kw)
    _wf.WAVEFORM_GENERATORS["dsss"] = _dsss_std


def restore_random_dsss():
    from rfgeo import waveform as _wf
    if hasattr(_wf, "_ORIG_DSSS"):
        _wf.WAVEFORM_GENERATORS["dsss"] = _wf._ORIG_DSSS
