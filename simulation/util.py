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

def create_boundary_band_solid_mask(shape, r_k0, noise_threshold, boundary_band_radius, dx, dy, dz, center = None, band_half_thickness=None, device=None):
    """
    solid_cell: (Nz, Ny, Nx) bool or any tensor
    boundary_band_radius: 물리 단위의 반지름 (DX,DY,DZ와 같은 단위)
    DX, DY, DZ: 각 축 격자 간격 (물리 단위)
    band_half_thickness: 밴드 두께의 절반(물리 단위). None이면 min(DX,DY,DZ)로 설정.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Nz, Ny, Nx = shape

    # 밴드 두께(물리 단위): 기본은 가장 작은 격자 간격 기준으로 1셀 정도 폭
    if band_half_thickness is None:
        band_half_thickness = min(dx, dy, dz)

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
        cx, cy, cz = center

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