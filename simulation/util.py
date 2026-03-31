import torch
from CFD import generate_multiband_smooth_noise_fft

def create_sphere_solid_mask(shape, center = None, radius = None, dim = 3, device=None):
    """
    solid_cell: (Nz, Ny, Nx)
    중심 기준 가장 큰 원 내부=0, 외부=1 로 설정
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Nz, Ny, Nx = shape

    # 좌표 생성
    z = torch.arange(Nz, device=device)
    y = torch.arange(Ny, device=device)
    x = torch.arange(Nx, device=device)

    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

    # 중심 좌표
    if(not center):
        cz = (Nz - 1) / 2
        cy = (Ny - 1) / 2
        cx = (Nx - 1) / 2
    else:
        cx, cy, cz = center

    # 반지름 (inscribed circle)
    if(not radius):
        if(dim == 3):
            radius = min(Nx, Ny, Nz) / 2 - 1
        else:
            radius = min(Nx, Ny) / 2 - 1

    # 거리 계산
    if(dim == 3):
        dist2 = (yy - cy)**2 + (xx - cx)**2 + (zz - cz)**2
    else:
        dist2 = (yy - cy)**2 + (xx - cx)**2

    # 원 내부: 0, 외부: 1
    circle_mask = dist2 > radius**2   # True = 외부

    return circle_mask

def create_boundary_band_solid_mask(
    shape,
    r_k0,
    noise_threshold,
    boundary_band_radius,
    x_domain,
    y_domain,
    z_domain,
    center=None,
    band_half_thickness=None,
    device=None
):
    """
    solid_cell: (Nz, Ny, Nx) bool or any tensor
    boundary_band_radius: 물리 단위의 반지름 (DX,DY,DZ와 같은 단위)
    DX, DY, DZ: 각 축 격자 간격 (물리 단위)
    band_half_thickness: 밴드 두께의 절반(물리 단위). None이면 min(DX,DY,DZ)로 설정.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Nz, Ny, Nx = shape
    
    dx = (x_domain[1] - x_domain[0]) / Nx
    dy = (y_domain[1] - y_domain[0]) / Ny
    dz = (z_domain[1] - z_domain[0]) / Nz

    # 밴드 두께(물리 단위): 기본은 가장 큰 격자 간격 기준으로 1셀 정도 폭
    if band_half_thickness is None:
        band_half_thickness = max(dx, dy, dz)

    # 인덱스 좌표
    z = torch.arange(Nz, device=device, dtype=torch.float32)
    y = torch.arange(Ny, device=device, dtype=torch.float32)
    x = torch.arange(Nx, device=device, dtype=torch.float32)

    # (Nz, Ny, Nx)로 맞추기
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

    # 중심(인덱스 단위)
    if(center is None):
        cz = (Nz - 1) / 2.0
        cy = (Ny - 1) / 2.0
        cx = (Nx - 1) / 2.0
    else:
        cz = (center[2] - z_domain[0]) / dz - 0.5
        cy = (center[1] - y_domain[0]) / dy - 0.5
        cx = (center[0] - x_domain[0]) / dx - 0.5

    # 물리 좌표 거리^2 (비등방 격자 고려 → 물리적으로 구형)
    dist2_phys = ((xx - cx) * dx) ** 2 + ((yy - cy) * dy) ** 2 + ((zz - cz) * dz) ** 2

    r = float(boundary_band_radius)
    t = float(band_half_thickness)

    # 물리 반지름 r 주변의 얇은 쉘(밴드)
    boundary_band = (dist2_phys > (r - t) ** 2) & (dist2_phys < (r + t) ** 2)

    noise = generate_multiband_smooth_noise_fft((Nz, Ny, Nx), [r_k0], [1.0], device=device)

    # 숫자가 작을 수록 구멍이 커짐.
    noise_mask = noise > noise_threshold

    boundary_band = boundary_band & ~noise_mask

    return boundary_band

def _safe_normalize(x, eps=1e-8, q_low=None, q_high=None):
    """
    x를 [0,1]로 정규화.
    q_low, q_high를 주면 percentile clipping 후 normalize.
    """
    x = x.float()

    if q_low is not None and q_high is not None:
        lo = torch.quantile(x.flatten(), q_low)
        hi = torch.quantile(x.flatten(), q_high)
        x = torch.clamp(x, lo, hi)
    else:
        lo = torch.min(x)
        hi = torch.max(x)

    return (x - lo) / (hi - lo + eps)


def _gradient_magnitude_3d(s, dz=1.0, dy=1.0, dx=1.0):
    """
    s: (nz, ny, nx)
    중앙차분 기반 gradient magnitude 계산
    """
    gz = torch.zeros_like(s)
    gy = torch.zeros_like(s)
    gx = torch.zeros_like(s)

    gz[1:-1] = (s[2:] - s[:-2]) / (2.0 * dz)
    gy[:, 1:-1] = (s[:, 2:] - s[:, :-2]) / (2.0 * dy)
    gx[:, :, 1:-1] = (s[:, :, 2:] - s[:, :, :-2]) / (2.0 * dx)

    # boundary는 one-sided difference
    gz[0] = (s[1] - s[0]) / dz
    gz[-1] = (s[-1] - s[-2]) / dz

    gy[:, 0] = (s[:, 1] - s[:, 0]) / dy
    gy[:, -1] = (s[:, -1] - s[:, -2]) / dy

    gx[:, :, 0] = (s[:, :, 1] - s[:, :, 0]) / dx
    gx[:, :, -1] = (s[:, :, -1] - s[:, :, -2]) / dx

    return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)


def volume_render_y(
    rho,
    dy=1.0,
    density_scale=8.0,
    grad_scale=4.0,
    tone_mapper="log1p",
    tone_param=10.0,
    use_gradient=True,
    density_weight=1.0,
    grad_weight=0.35,
    percentile_clip=(0.01, 0.995),
    white_background=False,
):
    """
    y축 방향 volume rendering

    Parameters
    ----------
    rho : torch.Tensor
        shape (nz, ny, nx)
    dy : float
        y축 셀 간격
    density_scale : float
        density opacity 강도
    grad_scale : float
        gradient opacity 강도
    tone_mapper : str
        'linear', 'log1p', 'asinh', 'gamma'
    tone_param : float
        tone mapping 파라미터
        - log1p/asinh일 때 강도
        - gamma일 때 gamma 값
    use_gradient : bool
        gradient magnitude를 opacity에 섞을지 여부
    density_weight : float
        color 계산 시 density 비중
    grad_weight : float
        color 계산 시 gradient 비중
    percentile_clip : tuple or None
        (low, high) quantile clipping
    reverse_y : bool
        True면 y=ny-1 -> 0 방향으로 렌더링
    white_background : bool
        True면 흰 배경 위에 합성, False면 검은 배경

    Returns
    -------
    image : torch.Tensor
        shape (nz, nx), [0,1]
    aux : dict
        디버깅용 중간 결과
    """
    assert rho.ndim == 3, f"rho.shape should be (nz, ny, nx), got {rho.shape}"
    rho = rho.float()

    # 1) density tone mapping용 base signal
    if tone_mapper == "linear":
        rho_tone = rho
    elif tone_mapper == "log1p":
        rho_tone = torch.log1p(tone_param * torch.clamp(rho, min=0.0))
    elif tone_mapper == "asinh":
        rho_tone = torch.asinh(tone_param * torch.clamp(rho, min=0.0))
    elif tone_mapper == "gamma":
        rho_pos = torch.clamp(rho, min=0.0)
        rho_norm = _safe_normalize(rho_pos)
        rho_tone = rho_norm ** (1.0 / tone_param)
    else:
        raise ValueError(f"Unknown tone_mapper: {tone_mapper}")

    # 2) gradient magnitude
    if use_gradient:
        grad = _gradient_magnitude_3d(rho, dy=dy, dx=1.0, dz=1.0)
    else:
        grad = torch.zeros_like(rho)

    # 3) 정규화
    if percentile_clip is not None:
        ql, qh = percentile_clip
        rho_n = _safe_normalize(rho_tone, q_low=ql, q_high=qh)
        grad_n = _safe_normalize(grad, q_low=ql, q_high=qh)
    else:
        rho_n = _safe_normalize(rho_tone)
        grad_n = _safe_normalize(grad)

    # 4) color signal
    # density + edge를 함께 써서 섬세한 구조를 드러냄
    color = density_weight * rho_n + grad_weight * grad_n
    color = _safe_normalize(color)

    # 5) opacity(알파)
    # density와 gradient를 같이 사용
    sigma = density_scale * rho_n + grad_scale * grad_n

    alpha = 1.0 - torch.exp(-sigma * dy)
    alpha = torch.clamp(alpha, 0.0, 1.0)

    # 7) front-to-back compositing
    # T_i = Π_{j<i} (1 - alpha_j)
    one_minus_alpha = 1.0 - alpha
    eps = 1e-10

    trans = torch.cumprod(one_minus_alpha + eps, dim=1)
    trans_exclusive = torch.ones_like(trans)
    trans_exclusive[:, 1:, :] = trans[:, :-1, :]

    weights = trans_exclusive * alpha

    image = torch.sum(weights * color, dim=1)  # (nz, nx)

    if white_background:
        acc_alpha = torch.sum(weights, dim=1)
        image = image + (1.0 - acc_alpha)

    image = torch.clamp(image, 0.0, 1.0)

    aux = {
        "rho_tone": rho_tone,
        "grad": grad,
        "rho_n": rho_n,
        "grad_n": grad_n,
        "color": color,
        "alpha": alpha,
        "weights": weights,
    }
    return image, aux

import numpy as np
from scipy.io.wavfile import write

def save_wav_from_pressure_traces(
    pressure_traces,
    output_path="explosion.wav",
    sr=44100,
    p_outer=None,
    channels=None,
    mix_weights=None,
    shock_weight=0.35,
    rumble_weight=0.85,
    texture_weight=0.20,
    rumble_smooth_ms=8.0,
    fade_in_ms=3.0,
    fade_out_ms=30.0,
):
    """
    pressure_traces:
        {
            "t": [...],
            "center": [...],
            "near1": [...],
            ...
        }

    - t의 불균일 샘플 간격을 고려해서 uniform audio timeline으로 resampling
    - np.gradient(signal, t) 로 시간축 반영한 변화량 계산
    - wav 저장
    """

    # --------------------------------------------------
    # 1) time axis
    # --------------------------------------------------
    t = np.asarray(pressure_traces["t"], dtype=np.float64)

    if t.ndim != 1 or len(t) < 2:
        raise ValueError("pressure_traces['t'] must be a 1D array with at least 2 samples.")

    # 정렬 보장
    order = np.argsort(t)
    t = t[order]

    # 중복 시간 제거 (np.interp는 x strictly increasing이 안전함)
    unique_mask = np.ones_like(t, dtype=bool)
    unique_mask[1:] = np.diff(t) > 0
    t = t[unique_mask]

    if len(t) < 2:
        raise ValueError("Need at least two unique time samples in pressure_traces['t'].")

    # --------------------------------------------------
    # 2) 어떤 pressure trace들을 쓸지 선택
    # --------------------------------------------------
    if channels is None:
        channels = [
            "center",
            "near1", "near2", "near3", "near4",
            "far1", "far2", "far3", "far4",
        ]

    valid_channels = [ch for ch in channels if ch in pressure_traces]
    if not valid_channels:
        raise ValueError("No valid pressure channels found in pressure_traces.")

    if mix_weights is None:
        # 기본적으로 center를 좀 더 크게
        mix_weights = {}
        for ch in valid_channels:
            if ch == "center":
                mix_weights[ch] = 1.0
            elif ch.startswith("near"):
                mix_weights[ch] = 0.7
            elif ch.startswith("far"):
                mix_weights[ch] = 0.45
            else:
                mix_weights[ch] = 0.5

    # --------------------------------------------------
    # 3) 여러 probe를 하나의 pressure signal로 합치기
    # --------------------------------------------------
    mixed_pressure = np.zeros_like(t, dtype=np.float64)
    total_w = 0.0

    for ch in valid_channels:
        p = np.asarray(pressure_traces[ch], dtype=np.float64)[order][unique_mask]
        w = float(mix_weights.get(ch, 1.0))
        mixed_pressure += w * p
        total_w += w

    mixed_pressure /= max(total_w, 1e-12)

    # ambient pressure 자동 추정
    if p_outer is None:
        # 초반 5% median을 ambient로 추정
        n0 = max(8, int(0.05 * len(mixed_pressure)))
        p_outer = np.median(mixed_pressure[:n0])

    # baseline 제거
    signal = mixed_pressure - p_outer

    # --------------------------------------------------
    # 4) 시간축을 고려한 성분 분해
    # --------------------------------------------------
    # shock: 시간 미분 기반 (불균일 dt 반영)
    shock = np.gradient(signal, t)

    # rumble: 저주파/완만한 압력 변화
    # 불균일 시계열에서 먼저 uniform으로 옮긴 뒤 smoothing 하는 게 더 안정적이라
    # 아래에서 uniform resample 후 smoothing 적용
    #
    # texture: signal에서 rumble을 뺀 잔차 성분
    #
    # 먼저 uniform timeline 생성
    t0 = t[0]
    t1 = t[-1]
    duration = t1 - t0

    if duration <= 0:
        raise ValueError("Non-positive duration in time axis.")

    audio_t = np.arange(t0, t1, 1.0 / sr, dtype=np.float64)
    if len(audio_t) < 2:
        raise ValueError("Audio timeline too short. Increase duration or reduce sample rate.")

    # resample to uniform timeline
    signal_u = np.interp(audio_t, t, signal)
    shock_u  = np.interp(audio_t, t, shock)

    # rumble smoothing
    rumble_kernel_len = max(3, int(sr * rumble_smooth_ms / 1000.0))
    if rumble_kernel_len % 2 == 0:
        rumble_kernel_len += 1

    kernel = np.hanning(rumble_kernel_len)
    kernel /= kernel.sum()

    rumble_u = np.convolve(signal_u, kernel, mode="same")
    texture_u = signal_u - rumble_u

    # --------------------------------------------------
    # 5) 성분 조합
    # --------------------------------------------------
    sound = (
        shock_weight * shock_u +
        rumble_weight * rumble_u +
        texture_weight * texture_u
    )

    # --------------------------------------------------
    # 6) click 방지: fade in / fade out
    # --------------------------------------------------
    fade_in_n = int(sr * fade_in_ms / 1000.0)
    fade_out_n = int(sr * fade_out_ms / 1000.0)

    if fade_in_n > 1:
        sound[:fade_in_n] *= np.linspace(0.0, 1.0, fade_in_n)

    if fade_out_n > 1:
        sound[-fade_out_n:] *= np.linspace(1.0, 0.0, fade_out_n)

    # --------------------------------------------------
    # 7) normalize
    # --------------------------------------------------
    sound = sound - np.mean(sound)

    peak = np.max(np.abs(sound))
    if peak > 0:
        sound = sound / peak

    # 살짝 headroom
    sound = sound

    # int16 wav
    wav = 5 * np.int16(np.clip(sound, -1.0, 1.0) * 32767)

    # --------------------------------------------------
    # 8) save
    # --------------------------------------------------
    write(output_path, sr, wav)

    return {
        "audio_t": audio_t,
        "signal_u": signal_u,
        "shock_u": shock_u,
        "rumble_u": rumble_u,
        "texture_u": texture_u,
        "sound": sound,
        "sr": sr,
        "p_outer": p_outer,
        "output_path": output_path,
    }