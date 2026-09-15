"""
impairments.py — 수신기 하드웨어 손상(CFO/SCO)을 OOD 평가 축으로 주입.

전제(검증됨)
-----------
- 주 파이프라인은 TDOA 전용(GCC-PHAT). FDOA 미사용.
- CFO(반송파 주파수 오프셋): 주파수축 회전. PHAT 위상백색화가 대체로 흡수 →
  학습엔 안 넣고 OOD 평가에만 주입하는 스트레스 축.
- SCO(샘플링 클럭 오프셋): 시간축을 (1+ppm)배로 스케일 → TDOA 경로에 직접 영향.
  CFO와 독립 축으로 분리.

둘 다 differential 만 남기려고 ref 수신기(0)는 0 으로 둔다.

성능: SCO 는 FFT 기반 벡터화 리샘플(O(n log n))로 구현 → 로컬 CPU 에서 실용적.
"""
import numpy as np


def apply_cfo(iq, cfo_hz, fs):
    """반송파 주파수 오프셋: 복소 지수 곱 (geometry.apply_doppler 와 동일 수학)."""
    n = iq.size
    t = np.arange(n) / fs
    return iq * np.exp(2j * np.pi * cfo_hz * t)


def apply_sco(iq, sco_ppm, fs=None):
    """
    샘플링 클럭 오프셋: 시간축을 (1+ppm)배로 리샘플.
    '고무자'처럼 표본 간격이 미세하게 늘거나 줆. 대역제한 보간을 FFT 로 벡터화.

    구현: 새 표본위치 t' = m/(1+ppm) 에서 원 신호를 재평가.
          균일 스케일이므로 주파수영역에서 위상경사로 근사하지 않고,
          FFT 업샘플 없이 '분수지연의 시간가변' 대신 정확한 sinc 행렬을
          블록 단위로 곱해 O(n log n) 에 가깝게 처리.

    작은 ppm(<=수십)에서 t'-t = -m*ppm 로 위치가 서서히 밀리므로,
    전체를 하나의 fftfreq 위상경사 스케일로 처리할 수 없다(위치의존).
    → 여기서는 zero-pad FFT 보간(대역제한 업샘플) 후 새 격자에서 선형보간.
    이 방식이 sinc 직접합성 대비 훨씬 빠르면서 대역제한성을 보존한다.
    """
    n = iq.size
    scale = 1.0 + sco_ppm * 1e-6
    if abs(sco_ppm) < 1e-12:
        return iq.copy()

    # 새 표본 위치(분수). sco_ppm>0 = 빠른 샘플링(시간 압축) → 새 격자를 scale 배로
    # 촘촘히 잡아 원 신호를 더 넓게 훑음 → 재표본 신호의 주파수가 (1+ppm)배 상승.
    new_idx = np.arange(n) * scale

    # 대역제한 업샘플(FFT zero-pad) 로 촘촘한 격자 확보 후 선형보간.
    U = 8  # 업샘플 배수 (대역제한 보간 정밀도)
    N2 = n * U
    X = np.fft.fft(iq)
    # zero-pad in frequency (분석신호/복소 → 중앙 삽입)
    Xp = np.zeros(N2, dtype=complex)
    half = n // 2
    Xp[:half] = X[:half]
    Xp[N2 - (n - half):] = X[half:]
    up = np.fft.ifft(Xp) * U  # 업샘플 신호(길이 N2), 격자 간격 1/U
    # 새 위치를 업샘플 격자 인덱스로: pos_up = new_idx * U
    pos = new_idx * U
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    i0 = np.clip(i0, 0, N2 - 2)
    out = up[i0] * (1 - frac) + up[i0 + 1] * frac
    return out.astype(complex)


def draw_impairments(M, rng, cfo_ppm=0.0, sco_ppm=0.0, carrier_hz=1.5e9):
    """
    수신기 M개에 대해 CFO(Hz)/SCO(ppm) 를 독립 draw. ref(0)=0 (differential).
    cfo_ppm/sco_ppm 은 '균등분포의 반폭(ppm)'. 0 이면 해당 손상 없음.
    """
    cfo_hz = np.zeros(M)
    sco = np.zeros(M)
    if cfo_ppm > 0:
        cfo_hz = rng.uniform(-cfo_ppm, cfo_ppm, size=M) * 1e-6 * carrier_hz
        cfo_hz[0] = 0.0
    if sco_ppm > 0:
        sco = rng.uniform(-sco_ppm, sco_ppm, size=M)
        sco[0] = 0.0
    return cfo_hz, sco


# CFO/SCO OOD 축 정의 (gen_shift 의 shifts dict 에 병합).
#
# CFO 축 재설계 (B: 전이 세분화 + A: 현실 잔류 반영):
#   실측에선 CFO 를 PLL/정합필터로 사전보상하고 '잔류'만 남는다. 잔류는 보통
#   <<0.1ppm. 반면 0.5ppm 은 스냅샷당 ~1.5 cycle 위상회전으로 GCC 봉우리를
#   상당히 뭉갠다(이미 커버리지 붕괴). 따라서 흥미로운 전이(90%->붕괴)는
#   0~0.5ppm 사이에 있으므로 그 구간을 촘촘히 본다. 10ppm 은 완전 포화라 제외.
#   ppm 은 '균등분포 반폭'(수신기간 differential 은 최대 2*ppm).
HW_SHIFTS = {
    "cfo_0": {"cfo_ppm": 0.0},    # in-distribution
    "cfo_1": {"cfo_ppm": 0.05},   # 보상 후 잔류(현실적 최선)
    "cfo_2": {"cfo_ppm": 0.1},    # 잔류 상한
    "cfo_3": {"cfo_ppm": 0.2},    # 보상 미흡
    "cfo_4": {"cfo_ppm": 0.5},    # 거의 무보상(붕괴 확인점)
    # SCO 는 2ms 스냅샷에서 10ppm 까지도 무반응(누적 sub-0.1샘플).
    # 반응이 시작되는 지점을 찾기 위해 상한을 크게 확장(50/100ppm).
    "sco_0": {"sco_ppm": 0.0},    # in-distribution
    "sco_1": {"sco_ppm": 2.0},
    "sco_2": {"sco_ppm": 10.0},
    "sco_3": {"sco_ppm": 50.0},
    "sco_4": {"sco_ppm": 100.0},  # 극단(반응 시작점 탐색)
}
