import torch
from CFD import generate_multiband_smooth_noise_fft

def create_sphere_solid_mask(shape, dim = 3, device=None):
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
    cz = (Nz - 1) / 2
    cy = (Ny - 1) / 2
    cx = (Nx - 1) / 2
    # 반지름 (inscribed circle)
    if(dim == 3):
        r = min(Nx, Ny, Nz) / 2 - 1
    else:
        r = min(Nx, Ny) / 2 - 1

    # 거리 계산
    if(dim == 3):
        dist2 = (yy - cy)**2 + (xx - cx)**2 + (zz - cz)**2
    else:
        dist2 = (yy - cy)**2 + (xx - cx)**2

    # 원 내부: 0, 외부: 1
    circle_mask = dist2 > r**2   # True = 외부

    return circle_mask

def create_boundary_band_solid_mask(shape, r_k0, noise_threshold, boundary_band_radius, dx, dy, dz, band_half_thickness=None, device=None):
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
    cz = (Nz - 1) / 2.0
    cy = (Ny - 1) / 2.0
    cx = (Nx - 1) / 2.0

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
