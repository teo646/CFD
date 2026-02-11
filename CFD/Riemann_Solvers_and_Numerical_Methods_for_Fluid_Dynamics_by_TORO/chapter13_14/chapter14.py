import torch

def minbee_limited_slope(dL, dR, eps=1e-12):
    """
    dL = W_i - W_{i-1}
    dR = W_{i+1} - W_i
    returns limited slope Δ_i (same shape as dL/dR)
    """
    r = dL / (dR + eps)
    phi = torch.clamp(r, min=0.0, max=1.0)  # max(0, min(1, r))
    return phi * dR

def muscl_reconstruction(CELL, dx, dt, GAMMA, normal='x'):
    """
    Perform MUSCL-Hancock-type reconstruction (without TVD limiter).

    Parameters
    ----------
    CELL : torch.Tensor
        Cell-averaged primitive variables, shape (Nz+2, Ny+2, Nx+2, 5)
        [rho, u, v, w, p]

    dt : float
        Time step

    normal : str
        'x' or 'y' or 'z'

    Returns
    -------
    WL, WR : torch.Tensor
        Reconstructed left/right states at interfaces along `normal`.

        For normal='x': shape (Nz+2, Ny+2, Nx, 5)
        For normal='y': shape (Nz+2, Ny, Nx+2, 5)
        For normal='z': shape (Nz, Ny+2, Nx+2, 5)

    Wil' = Wi - 0.5 * Di - 0.5 * dt / dx * A Di
    WiR' = Wi + 0.5 * Di - 0.5 * dt / dx * A Di
    
    """
    # -------- pick normal velocity component u_n --------
    if normal == 'x':
        Wi = CELL[:, :, 1:-1, :]           # i
        dL = CELL[:, :, 1:-1, :] - CELL[:, :, 0:-2, :]  # i - (i-1)
        dR = CELL[:, :, 2:  , :] - CELL[:, :, 1:-1, :]  # (i+1) - i
    elif normal == 'y':
        Wi = CELL[:, 1:-1, :, :]
        dL = CELL[:, 1:-1, :, :] - CELL[:, 0:-2, :, :]
        dR = CELL[:, 2:  , :, :] - CELL[:, 1:-1, :, :]
    elif normal == 'z':
        Wi = CELL[1:-1, :, :, :]
        dL = CELL[1:-1, :, :, :] - CELL[0:-2, :, :, :]
        dR = CELL[2:  , :, :, :] - CELL[1:-1, :, :, :]

    Di = minbee_limited_slope(dL, dR, eps=1e-12)
    #calculate (A(Wi) + B(Wi) + C(Wi)) Di = K
    rho = Wi[..., 0]
    u = Wi[..., 1]
    v = Wi[..., 2]
    w = Wi[..., 3]
    p = Wi[..., 4]

    rho_x = Di[..., 0]
    u_x = Di[..., 1]
    v_x = Di[..., 2]
    w_x = Di[..., 3]
    p_x = Di[..., 4]

    K = torch.zeros_like(Wi)

    if(normal == 'x'):
        K[..., 0] = u * rho_x + rho * u_x
        K[..., 1] = u * u_x + p_x / rho
        K[..., 2] = u * v_x
        K[..., 3] = u * w_x
        K[..., 4] = u * p_x + GAMMA * p * u_x
    elif(normal == 'y'):
        K[..., 0] = v * rho_x + rho * v_x
        K[..., 1] = v * u_x
        K[..., 2] = v * v_x + p_x / rho
        K[..., 3] = v * w_x
        K[..., 4] = v * p_x + GAMMA * p * v_x
    elif(normal == 'z'):
        K[..., 0] = w * rho_x + rho * w_x
        K[..., 1] = w * u_x
        K[..., 2] = w * v_x
        K[..., 3] = w * w_x + p_x / rho
        K[..., 4] = w * p_x + GAMMA * p * w_x
    else:
        raise ValueError("normal must be 'x' or 'y' or 'z'")

    WL = Wi - 0.5 * Di - 0.5 * dt / dx * K
    WR = Wi + 0.5 * Di - 0.5 * dt / dx * K

    # WL and Wr are the left and right states at the cell.
    # the function is returning the left and right states at the interface between the cell and the next cell.

    if(normal == 'x'):
        return WR[:, :, :-1, :], WL[:, :, 1:, :]
    elif(normal == 'y'):
        return WR[:, :-1, :, :], WL[:, 1:, :, :]
    elif(normal == 'z'):
        return WR[:-1, :, :, :], WL[1:, :, :, :]
    else:
        raise ValueError("normal must be 'x' or 'y' or 'z'")


#고차 방식을 사용하기 위해서는 fiction cell이 2개가 필요하며 이에 따라 update 방식도 바꿘다. 
def reflective_bc(CELL: "torch.Tensor", normal='x'):
    """
    Apply reflective (wall) boundary condition on both sides along `normal`.

    CELL: (..., 5) with [rho, u, v, w, p]
    normal: 'x' or 'y' or 'z'
    """
    # which spatial axis is normal?
    axis = {'z': 0, 'y': 1, 'x': 2}[normal]
    # which velocity component index is normal? (u=1, v=2, w=3)
    v_idx = {'x': 1, 'y': 2, 'z': 3}[normal]

    # --- left side ghosts: [1] <- [2], [0] <- [3] ---
    # copy everything first
    src1 = [slice(None), slice(None), slice(None)]
    src2 = [slice(None), slice(None), slice(None)]
    dst1 = [slice(None), slice(None), slice(None)]
    dst2 = [slice(None), slice(None), slice(None)]

    dst1[axis] = 1; src1[axis] = 2
    dst2[axis] = 0; src2[axis] = 3

    CELL[tuple(dst1) + (slice(None),)] = CELL[tuple(src1) + (slice(None),)]
    CELL[tuple(dst2) + (slice(None),)] = CELL[tuple(src2) + (slice(None),)]

    # flip only normal velocity
    CELL[tuple(dst1) + (v_idx,)] *= -1
    CELL[tuple(dst2) + (v_idx,)] *= -1

    # --- right side ghosts: [-2] <- [-3], [-1] <- [-4] ---
    src3 = [slice(None), slice(None), slice(None)]
    src4 = [slice(None), slice(None), slice(None)]
    dst3 = [slice(None), slice(None), slice(None)]
    dst4 = [slice(None), slice(None), slice(None)]

    dst3[axis] = -2; src3[axis] = -3
    dst4[axis] = -1; src4[axis] = -4

    CELL[tuple(dst3) + (slice(None),)] = CELL[tuple(src3) + (slice(None),)]
    CELL[tuple(dst4) + (slice(None),)] = CELL[tuple(src4) + (slice(None),)]

    CELL[tuple(dst3) + (v_idx,)] *= -1
    CELL[tuple(dst4) + (v_idx,)] *= -1

    return CELL

def transmissive_bc(CELL: "torch.Tensor", normal='x'):
    """
    Apply transmissive (zero-gradient) boundary condition
    on both sides along `normal`.

    CELL: (..., 5) with [rho, u, v, w, p]
    normal: 'x' or 'y' or 'z'
    """

    # spatial axis index
    axis = {'z': 0, 'y': 1, 'x': 2}[normal]

    # --- left side ghosts ---
    src1 = [slice(None), slice(None), slice(None)]
    src2 = [slice(None), slice(None), slice(None)]
    dst1 = [slice(None), slice(None), slice(None)]
    dst2 = [slice(None), slice(None), slice(None)]

    dst1[axis] = 1; src1[axis] = 2
    dst2[axis] = 0; src2[axis] = 3

    CELL[tuple(dst1) + (slice(None),)] = CELL[tuple(src1) + (slice(None),)]
    CELL[tuple(dst2) + (slice(None),)] = CELL[tuple(src2) + (slice(None),)]

    # --- right side ghosts ---
    src3 = [slice(None), slice(None), slice(None)]
    src4 = [slice(None), slice(None), slice(None)]
    dst3 = [slice(None), slice(None), slice(None)]
    dst4 = [slice(None), slice(None), slice(None)]

    dst3[axis] = -2; src3[axis] = -3
    dst4[axis] = -1; src4[axis] = -4

    CELL[tuple(dst3) + (slice(None),)] = CELL[tuple(src3) + (slice(None),)]
    CELL[tuple(dst4) + (slice(None),)] = CELL[tuple(src4) + (slice(None),)]

    return CELL