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
    r_k0_list,   # e.g. [1/8, 1/4, 1/2]  (ratio to "cells" scale)
    weight_list, # e.g. [1.0, 0.7, 0.3]
    device=None,
    eps=1e-12,
):
    """
    Multi-band smooth noise using FFT with multiple Gaussian low-pass envelopes.

    Parameters
    ----------
    shape : (nz, ny, nx)
        Grid resolution.
    r_k0_list : list[float]
        Each r_k0 is a ratio to the average cell count:
            k0_i = r_k0_i * ((nx + ny + nz) / 3)
        Using ratios makes it behave similarly across resolutions.
        Larger r_k0 -> smoother (stronger low-pass), smaller r_k0 -> rougher.
    weight_list : list[float]
        Mixing weights per band. Same length as r_k0_list.
    device : torch.device or str or None
        Target device for returned tensor.
    eps : float
        Numerical stability for std.

    Returns
    -------
    torch.Tensor
        Noise field of shape (nz, ny, nx), standardized to unit std.
    """
    if len(r_k0_list) != len(weight_list):
        raise ValueError("r_k0_list and weight_list must have the same length.")

    nz, ny, nx = shape
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("shape must be positive in all dims.")

    # Frequency grids (cycles per sample)
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kz = np.fft.fftfreq(nz)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2

    # Convert ratios -> k0 in "cell-count" scale (resolution-invariant-ish)
    mean_n = (nx + ny + nz) / 3.0
    k0_list = [float(r) * mean_n for r in r_k0_list]

    spectrum = np.zeros((nz, ny, nx), dtype=np.complex128)

    for w, k0 in zip(weight_list, k0_list):
        # Complex white noise
        band = np.random.randn(nz, ny, nx) + 1j * np.random.randn(nz, ny, nx)
        # Gaussian low-pass envelope
        band *= np.exp(-K2 * (k0**2))
        spectrum += float(w) * band

    noise = np.fft.ifftn(spectrum).real
    std = np.std(noise)
    noise = noise / (std + eps)

    return torch.from_numpy(noise).to(device)

def gradient_scalar_field(p, dx, dy, dz):
    """
    Compute ∇p of a scalar field using finite differences.

    Parameters
    ----------
    p : torch.Tensor
        Scalar field with shape (Nz, Ny, Nx)
    dx, dy, dz : float
        Grid spacing

    Returns
    -------
    grad_p : torch.Tensor
        Gradient field with shape (3, Nz, Ny, Nx)
        grad_p[0] = ∂p/∂x
        grad_p[1] = ∂p/∂y
        grad_p[2] = ∂p/∂z
    """

    Nz, Ny, Nx = p.shape

    dpdx = torch.zeros_like(p)
    dpdy = torch.zeros_like(p)
    dpdz = torch.zeros_like(p)

    # central difference (interior)
    dpdx[..., 1:-1] = (p[..., 2:] - p[..., :-2]) / (2 * dx)
    dpdy[:, 1:-1, :] = (p[:, 2:, :] - p[:, :-2, :]) / (2 * dy)
    dpdz[1:-1, :, :] = (p[2:, :, :] - p[:-2, :, :]) / (2 * dz)

    # forward/backward difference (boundaries)
    dpdx[..., 0]  = (p[..., 1] - p[..., 0]) / dx
    dpdx[..., -1] = (p[..., -1] - p[..., -2]) / dx

    dpdy[:, 0, :]  = (p[:, 1, :] - p[:, 0, :]) / dy
    dpdy[:, -1, :] = (p[:, -1, :] - p[:, -2, :]) / dy

    dpdz[0, :, :]  = (p[1, :, :] - p[0, :, :]) / dz
    dpdz[-1, :, :] = (p[-1, :, :] - p[-2, :, :]) / dz

    grad_p = torch.stack([dpdx, dpdy, dpdz], dim=0)

    return grad_p

def create_explosion_initial_condition(
        nx,
        ny,
        nz,
        x_domain,
        y_domain,
        z_domain,
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

    CELL = torch.zeros((nz, ny, nx, 5), device=device)

    center_x = (x_domain[0] + x_domain[1]) / 2
    center_y = (y_domain[0] + y_domain[1]) / 2
    center_z = (z_domain[0] + z_domain[1]) / 2
    # 각 셀의 중심 좌표 계산 (ghost cell 제외한 실제 셀만)
    x_coords = torch.linspace(x_domain[0], x_domain[1], nx, device=device)
    y_coords = torch.linspace(y_domain[0], y_domain[1], ny, device=device)
    z_coords = torch.linspace(z_domain[0], z_domain[1], nz, device=device)
    Z, Y, X = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

    # 중심으로부터의 거리 계산
    if(nz == 1):
        distances2 = (X - center_x)**2 + (Y - center_y)**2
    else:
        distances2 = (X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2
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