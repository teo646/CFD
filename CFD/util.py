import torch
import numpy as np

def W_to_U(W, GAMMA):
    """
    Convert primitive variables to conserved variables.
    
    Parameters:
    -----------
    W : torch.Tensor
        Primitive variables, shape (..., 5) - [rho, u, v, w, p]
    
    Returns:
    --------
    U : torch.Tensor
        Conserved variables, shape (..., 5) - [rho, rho*u, rho*v, rho*w, E]
    """
    rho = W[..., 0]
    u = W[..., 1]
    v = W[..., 2]
    w = W[..., 3]
    p = W[..., 4]
    E = p / (GAMMA - 1) + 0.5 * rho * (u**2 + v**2 + w**2)
    return torch.stack([rho, u * rho, v * rho, w * rho, E], dim=-1)

def U_to_W(U, GAMMA):
    """
    Convert conserved variables to primitive variables.
    
    Parameters:
    -----------
    U : torch.Tensor
        Conserved variables, shape (..., 5) - [rho, rho*u, rho*v, rho*w, E]
    
    Returns:
    --------
    W : torch.Tensor
        Primitive variables, shape (..., 5) - [rho, u, v, w, p]
    """
    rho = torch.clamp(U[..., 0], min=1e-10)
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    w = U[..., 3] / rho
    E = U[..., 4]
    p = (GAMMA - 1) * (E - 0.5 * rho * (u**2 + v**2 + w**2))
    p = torch.clamp(p, min=1e-10)
    return torch.stack([rho, u, v, w, p], dim=-1)

def W_to_F(W, GAMMA, normal='x'):
    """
    Convert primitive variables to flux vector.
    
    Parameters:
    -----------
    W : torch.Tensor
        Primitive variables, shape (..., 5) - [rho, u, v, w, p]
    """
    rho = W[..., 0]
    u = W[..., 1]
    v = W[..., 2]
    w = W[..., 3]
    p = W[..., 4]
    E = p / (GAMMA - 1) + 0.5 * rho * (u**2 + v**2 + w**2)
    
    if normal == 'x':
        F = torch.stack([rho * u, rho * u * u + p, rho * u * v, rho * u * w, (E + p) * u], dim=-1)
    elif normal == 'y':
        F = torch.stack([rho * v, rho * u * v, rho * v * v + p, rho * v * w, (E + p) * v], dim=-1)
    elif normal == 'z':
        F = torch.stack([rho * w, rho * u * w, rho * v * w, rho * w * w + p, (E + p) * w], dim=-1)
    else:
        raise ValueError("normal must be 'x' or 'y' or 'z'")

    return F

def generate_multiband_smooth_noise_fft(
    shape,
    r_k0_list,   # 비례 계수
    weight_list, # 밴드별 가중치
    dx=1,
    dy=1,
    dz=1,  # 비등방 격자 간격 추가
    device=None,
    eps=1e-12,
):
    """
    비등방 격자(Anisotropic Grid)를 지원하는 FFT 기반 멀티밴드 노이즈 생성.
    """
    if len(r_k0_list) != len(weight_list):
        raise ValueError("r_k0_list and weight_list must have the same length.")

    nz, ny, nx = shape

    # 1. 물리적 주파수 그리드 생성 (Physical frequencies: cycles per unit length)
    # d 인자를 주어 각 축의 물리적 간격을 반영합니다.
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    kz = np.fft.fftfreq(nz, d=dz)
    
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    
    # K2는 물리적 주파수의 제곱 합 (1/length^2 단위)
    K2 = KX**2 + KY**2 + KZ**2

    # 2. 물리적 차단 길이(Cut-off length) 설정
    # 기존 r_k0가 "평균 셀 개수"에 비례했다면, 
    # 이제는 "평균 물리적 도메인 크기"에 비례하도록 설정하여 일관성을 유지합니다.
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz
    mean_L = (Lx + Ly + Lz) / 3.0
    
    # k0_phys는 물리적 길이(length) 단위를 가집니다.
    k0_list = [float(r) * mean_L for r in r_k0_list]

    spectrum = np.zeros((nz, ny, nx), dtype=np.complex128)

    for w, k0_phys in zip(weight_list, k0_list):
        # 복소 화이트 노이즈 생성
        band = np.random.randn(nz, ny, nx) + 1j * np.random.randn(nz, ny, nx)
        
        # 가우시안 저역 통과 필터 (등방성 유지)
        # exp(-K_phys^2 * k0_phys^2) 형태가 되어 물리 공간에서 일정한 반경을 가짐
        band *= np.exp(-K2 * (k0_phys**2))
        spectrum += float(w) * band

    # 역푸리에 변환으로 실공간 노이즈 생성
    noise = np.fft.ifftn(spectrum).real
    
    # 표준화 (Unit Standard Deviation)
    std = np.std(noise)
    noise = noise / (std + eps)

    return torch.from_numpy(noise.astype(np.float32)).to(device)

def create_explosion_initial_condition(
        RESOLUTION,
        x_domain,
        y_domain,
        z_domain,
        explosion_center,
        rho_inner,
        p_inner,
        rho_outer,
        p_outer,
        sigma,
        noise,
        r_k0_list = [0, 0, 0],
        weight_list = [0, 0, 0],
        device = None
    ):
    """
    3D 폭발에서 복잡한 비대칭 구조가 성장하도록 설계한 초기 조건.
    - 구대칭 폭발(가우시안) + 외부 다중스케일 밀도 클럼프 + 쉘(접촉면 부근) perturb
    """ 
    nz, ny, nx = RESOLUTION

    CELL = torch.zeros((nz, ny, nx, 5), device=device)

    explosion_center_x = explosion_center[0]
    explosion_center_y = explosion_center[1]
    explosion_center_z = explosion_center[2]
    # 각 셀의 중심 좌표 계산 (ghost cell 제외한 실제 셀만)
    x_coords = torch.linspace(x_domain[0], x_domain[1], nx, device=device)
    y_coords = torch.linspace(y_domain[0], y_domain[1], ny, device=device)
    z_coords = torch.linspace(z_domain[0], z_domain[1], nz, device=device)
    Z, Y, X = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

    # 중심으로부터의 거리 계산
    if(nz == 1):
        distances2 = (X - explosion_center_x)**2 + (Y - explosion_center_y)**2
    else:
        distances2 = (X - explosion_center_x)**2 + (Y - explosion_center_y)**2 + (Z - explosion_center_z)**2
    # === Smooth Gaussian Profile ===
    # exp(-r²/(2σ²)) 형태
    gaussian_profile = torch.exp(-distances2 / (2 * sigma**2))

    # 기본값 설정 (외부 영역) - ghost cell 포함 전체
    CELL[..., 0] = rho_outer # rho (low density)
    CELL[..., 4] = p_outer    # p (low pressure)

    # 폭발 영역 설정 (고압, 고밀도) - 실제 셀만 (ghost cell 제외)
    CELL[..., 0] += (rho_inner - rho_outer) * gaussian_profile    # rho (high density)
    CELL[..., 4] += (p_inner - p_outer) * gaussian_profile     # p (high pressure)
    z, y, x = gaussian_profile.shape
    rho_noise_field = generate_multiband_smooth_noise_fft(
                                                        (z, y, x),
                                                        r_k0_list,  
                                                        weight_list,
                                                        device=device,
                                                        eps=1e-12,
                                                    )
    CELL[..., 0] += noise * rho_noise_field

    # final safety
    CELL[..., 0] = torch.clamp(CELL[..., 0], min=1e-10)
    CELL[..., 4] = torch.clamp(CELL[..., 4], min=1e-10)

    return CELL

def create_sphere_mask(
        RESOLUTION,
        x_domain,
        y_domain,
        z_domain,
        radius,
        center = None,
        device=None
    ):

    nz, ny, nx = RESOLUTION
    
    CELL = torch.zeros(RESOLUTION, device=device)

    if center is None:
        cx, cy, cz = (0, 0, 0)
    else:
        cx, cy, cz = center
    # 각 셀의 중심 좌표 계산 (ghost cell 제외한 실제 셀만)
    x_coords = torch.linspace(x_domain[0], x_domain[1], nx, device=device)
    y_coords = torch.linspace(y_domain[0], y_domain[1], ny, device=device)
    z_coords = torch.linspace(z_domain[0], z_domain[1], nz, device=device)
    Z, Y, X = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

    # 중심으로부터의 거리 계산
    if(nz == 1):
        distances2 = (X - cx)**2 + (Y - cy)**2
    else:
        distances2 = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
    # === Smooth Gaussian Profile ===
    # exp(-r²/(2σ²)) 형태
    return distances2 < radius ** 2    