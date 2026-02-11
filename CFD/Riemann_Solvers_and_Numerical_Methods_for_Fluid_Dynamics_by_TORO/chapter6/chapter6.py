'''
Exact Riemann solver for 1 dimensional Euler equations.
Revised to support 3 dimensional Euler equations.
'''
import torch

RIEMANN_SOLVER_TOL = 1e-6

def compute_f_and_df(p, W_L, W_R, GAMMA):
    """
    W_L, W_R: (..., 5) - [rho, u, v, w, p] primitive variables
    p: (...,) - pressure guess
    """
    left_rho = W_L[..., 0]
    left_p = W_L[..., 4]
    right_rho = W_R[..., 0]
    right_p = W_R[..., 4]
    
    Al = 2 / (GAMMA + 1) / left_rho
    Ar = 2 / (GAMMA + 1) / right_rho
    Bl = (GAMMA - 1) / (GAMMA + 1) * left_p
    Br = (GAMMA - 1) / (GAMMA + 1) * right_p
    al = torch.sqrt(GAMMA * left_p / left_rho)
    ar = torch.sqrt(GAMMA * right_p / right_rho)
    
    left_shock_cell = p > left_p
    right_shock_cell = p > right_p

    #left rarefaction wave
    fl = 2 * al / (GAMMA - 1) * ((p/left_p)**((GAMMA - 1)/(2 * GAMMA)) - 1)
    d_fl = 1 / left_rho / al * (p/left_p)**(-(GAMMA + 1)/(2 * GAMMA))

    #overide left shock wave
    fl[left_shock_cell] = ((p - left_p) * torch.sqrt(Al / (p + Bl)))[left_shock_cell]
    d_fl[left_shock_cell] = (torch.sqrt(Al / (Bl + p)) * (1 - 0.5 * (p - left_p) / (Bl + p)))[left_shock_cell]


    #right rarefaction wave
    fr = 2 * ar / (GAMMA - 1) * ((p/right_p)**((GAMMA - 1)/(2 * GAMMA)) - 1)
    d_fr = 1 / right_rho / ar * (p/right_p)**(-(GAMMA + 1)/(2 * GAMMA))
    
    #overide right shock wave
    fr[right_shock_cell] = ((p - right_p)* torch.sqrt(Ar / (p + Br)))[right_shock_cell]
    d_fr[right_shock_cell] = (torch.sqrt(Ar / (Br + p)) * (1 - 0.5 * (p - right_p) / (Br + p)))[right_shock_cell]

    return fl, d_fl, fr, d_fr

def solve_riemann_star_state(W_L, W_R, GAMMA, tol = RIEMANN_SOLVER_TOL, normal='x'):  
    """
    Get the exact Riemann solution for the Euler equations.
    Newton-Raphson iterative procedure

    p(k)= p(k-1)- f(p(k-1)) / f'(p(k-1))

    W_L, W_R: (..., 4) - [rho, u, v, p] primitive variables
    normal: 'x' or 'y' - direction of the Riemann problem
    
    return value: p*, u* (or v*), rho*l, rho*r
    """
    # Extract variables from state
    left_rho = W_L[..., 0]
    left_p = W_L[..., 4]
    right_rho = W_R[..., 0]
    right_p = W_R[..., 4]
    
    # Select velocity component based on normal direction
    if normal == 'x':
        left_u = W_L[..., 1]  # u component
        right_u = W_R[..., 1]
    elif normal == 'y':
        left_u = W_L[..., 2]  # v component
        right_u = W_R[..., 2]
    elif normal == 'z':
        left_u = W_L[..., 3]  # w component
        right_u = W_R[..., 3]
    else:
        raise ValueError("normal must be 'x' or 'y' or 'z'")

    # Initial guess for the pressure
    # Should be optimaized using Two–Rarefaction approximation, primitive variables, Two–Shock approximation.
    p = 0.5 * (left_p + right_p)
    while(True):
        prev_p = p
        fl, d_fl, fr, d_fr = compute_f_and_df(p, W_L, W_R, GAMMA)

        f = fl + fr  + right_u - left_u
        df = d_fl + d_fr

        p = p - f / df
        #음압 방지.
        p = torch.clamp(p, min=torch.tensor(1e-12, device=p.device))
        #모든 셀에서 충족하면 종료.
        if(torch.all(2 * abs(p - prev_p) < tol * (p + prev_p))):
            break
    
    fl, d_fl, fr, d_fr = compute_f_and_df(p, W_L, W_R, GAMMA)
    u = 0.5 * (left_u + right_u + fr - fl)

    left_shock_cell = p > left_p
    right_shock_cell = p > right_p

    rho_l_star = left_rho * (p / left_p) ** (1 / GAMMA)
    rho_l_star[left_shock_cell] = (left_rho * (GAMMA * (p + left_p) - left_p + p) /
                                (GAMMA * (p + left_p) - p + left_p))[left_shock_cell]

    rho_r_star = right_rho * (p / right_p) ** (1 / GAMMA)
    rho_r_star[right_shock_cell] = (right_rho * (GAMMA * (p + right_p) - right_p + p) /
                                (GAMMA * (p + right_p) - p + right_p))[right_shock_cell]

    return p, u, rho_l_star, rho_r_star

def exact_riemann_flux(W_L, W_R, GAMMA, normal='x'):
    """
    Solve local Riemann problem at s = 0 for all interfaces along one direction.
    Returns the flux at the interface.
    
    W_L, W_R: (..., 4) - [rho, u, v, p] primitive variables
    normal: 'x' or 'y' - direction of the Riemann problem
    Returns: (..., 4) - flux at the interface
    """
    # Extract variables from state
    left_rho = W_L[..., 0]
    left_p = W_L[..., 4]
    right_rho = W_R[..., 0]
    right_p = W_R[..., 4]
    
    # Select velocity component based on normal direction
    if normal == 'x':
        left_u = W_L[..., 1]  # u component
        right_u = W_R[..., 1]
        left_v = W_L[..., 2]  # v component (perpendicular)
        right_v = W_R[..., 2]
        left_w = W_L[..., 3]  # w component (perpendicular)
        right_w = W_R[..., 3]
    elif normal == 'y':
        left_u = W_L[..., 2]  # v component
        right_u = W_R[..., 2]
        left_v = W_L[..., 1]  # u component (perpendicular)
        right_v = W_R[..., 1]
        left_w = W_L[..., 3]  # w component (perpendicular)
        right_w = W_R[..., 3]
    elif normal == 'z':
        left_u = W_L[..., 3]  # w component
        right_u = W_R[..., 3]
        left_v = W_L[..., 2]  # v component (perpendicular)
        right_v = W_R[..., 2]
        left_w = W_L[..., 1]  # u component (perpendicular)
        right_w = W_R[..., 1]
    else:
        raise ValueError("normal must be 'x', 'y' or 'z'")
    
    p_star, u_star, rho_l_star, rho_r_star = solve_riemann_star_state(W_L, W_R, GAMMA, normal=normal)

    # s = 0 for Godunov flux evaluation
    s = torch.zeros_like(p_star)

    # Initialize solution variables
    shape = left_rho.shape
    rho = torch.zeros_like(left_rho)
    u = torch.zeros_like(left_u)
    p = torch.zeros_like(left_p)

    # Contact side masks (s=0)
    left_contact = s < u_star
    right_contact = ~left_contact

    # Left side: rarefaction or shock
    left_rarefaction = p_star < left_p
    left_shock = ~left_rarefaction

    # Right side: rarefaction or shock
    right_rarefaction = p_star < right_p
    right_shock = ~right_rarefaction

    # -------- Left rarefaction --------
    al = torch.sqrt(GAMMA * left_p / left_rho)
    s_hl = left_u - al

    # Region 1: left state
    mask_l1 = left_contact & left_rarefaction & (s < s_hl) 
    rho[mask_l1] = left_rho[mask_l1]
    u[mask_l1] = left_u[mask_l1]
    p[mask_l1] = left_p[mask_l1]

    # Region 2: star left
    al_star = al * (p_star / left_p) ** ((GAMMA - 1) / (2 * GAMMA))
    s_tl = u_star - al_star
    mask_l2 = left_contact & left_rarefaction & (s > s_tl)
    rho[mask_l2] = rho_l_star[mask_l2]
    u[mask_l2] = u_star[mask_l2]
    p[mask_l2] = p_star[mask_l2]

    # Region 3: inside fan
    mask_l3 = left_contact & left_rarefaction & ~(s < s_hl) & ~(s > s_tl)
    p[mask_l3] = left_p[mask_l3] * ((2 * al[mask_l3] + (GAMMA - 1) * (left_u[mask_l3] - s[mask_l3])) / (al[mask_l3] * (GAMMA + 1))) ** (2 * GAMMA / (GAMMA - 1))
    u[mask_l3] = 2 / (GAMMA + 1) * (al[mask_l3] + (GAMMA - 1) / 2 * left_u[mask_l3] + s[mask_l3])
    rho[mask_l3] = left_rho[mask_l3] * ((2 * al[mask_l3] + (GAMMA - 1) * (left_u[mask_l3] - s[mask_l3])) / (al[mask_l3] * (GAMMA + 1))) ** (2 / (GAMMA - 1))

    # -------- Left shock --------
    al = torch.sqrt(GAMMA * left_p / left_rho)
    s_l = left_u - al * torch.sqrt((GAMMA * (p_star + left_p) + p_star - left_p) / (2 * GAMMA * left_p))
    mask_ls = left_contact & left_shock & (s < s_l)
    rho[mask_ls] = left_rho[mask_ls]
    u[mask_ls] = left_u[mask_ls]
    p[mask_ls] = left_p[mask_ls]

    mask_ls2 = left_contact & left_shock & ~(s < s_l)
    rho[mask_ls2] = rho_l_star[mask_ls2]
    u[mask_ls2] = u_star[mask_ls2]
    p[mask_ls2] = p_star[mask_ls2]

    # -------- Right rarefaction --------
    ar = torch.sqrt(GAMMA * right_p / right_rho)
    s_hr = right_u + ar

    # Region 1: right state
    mask_r1 = right_contact & right_rarefaction & (s > s_hr)
    rho[mask_r1] = right_rho[mask_r1]
    u[mask_r1] = right_u[mask_r1]
    p[mask_r1] = right_p[mask_r1]

    # Region 2: star right
    ar_star = ar * (p_star / right_p) ** ((GAMMA - 1) / (2 * GAMMA))
    s_tr = u_star + ar_star
    mask_r2 = right_contact & right_rarefaction & (s < s_tr)
    rho[mask_r2] = rho_r_star[mask_r2]
    u[mask_r2] = u_star[mask_r2]
    p[mask_r2] = p_star[mask_r2]

        # Region 3: inside fan
    mask_r3 = right_contact & right_rarefaction & ~(s > s_hr) & ~(s < s_tr)
    p[mask_r3] = right_p[mask_r3] * ((2 * ar[mask_r3] + (GAMMA - 1) * (s[mask_r3] - right_u[mask_r3])) / (ar[mask_r3] * (GAMMA + 1))) ** (2 * GAMMA / (GAMMA - 1))
    u[mask_r3] = 2 / (GAMMA + 1) * (-ar[mask_r3] + (GAMMA - 1) / 2 * right_u[mask_r3] + s[mask_r3])
    rho[mask_r3] = right_rho[mask_r3] * ((2 * ar[mask_r3] + (GAMMA - 1) * (s[mask_r3] - right_u[mask_r3])) / (ar[mask_r3] * (GAMMA + 1))) ** (2 / (GAMMA - 1))

    # -------- Right shock --------
    ar = torch.sqrt(GAMMA * right_p / right_rho)
    s_r = right_u + ar * torch.sqrt((GAMMA * (p_star + right_p) + p_star - right_p) / (2 * GAMMA * right_p))
    mask_rs = right_contact & right_shock & (s > s_r)
    rho[mask_rs] = right_rho[mask_rs]
    u[mask_rs] = right_u[mask_rs]
    p[mask_rs] = right_p[mask_rs]

    mask_rs2 = right_contact & right_shock & ~(s > s_r)
    rho[mask_rs2] = rho_r_star[mask_rs2]
    u[mask_rs2] = u_star[mask_rs2]
    p[mask_rs2] = p_star[mask_rs2]

    # Calculate flux
    v = torch.zeros_like(left_v)
    v[left_contact] = left_v[left_contact]
    v[right_contact] = right_v[right_contact]

    w = torch.zeros_like(left_w)
    w[left_contact] = left_w[left_contact]
    w[right_contact] = right_w[right_contact]

    E = p / (GAMMA - 1) + 0.5 * rho * (u**2 + v**2 + w**2)
    
    flux = torch.zeros((*shape, 5), device=rho.device)
    flux[..., 0] = rho * u                    # F_rho
    flux[..., 4] = u * (E + p)                # F_E

    if normal == 'x':
        flux[..., 1] = rho * u**2 + p             # F_rhou
        flux[..., 2] = rho * u * v                # F_rhov
        flux[..., 3] = rho * u * w                # F_rhow
    elif normal == 'y':  
        flux[..., 1] = rho * u * v               # G_rhou  
        flux[..., 2] = rho * u**2 + p             # G_rhov (u is v in y-direction)
        flux[..., 3] = rho * w * u             # G_rhow
    else:
        flux[..., 1] = rho * w * u                # G_rhou
        flux[..., 2] = rho * v * u                # G_rhov
        flux[..., 3] = rho * u**2 + p             # G_rhow

    return flux

def cal_dt(CELL, cfl_coefficient, dx, GAMMA):
    """
    Calculate time step using CFL condition.
    
    Parameters:
    -----------
    CELL : torch.Tensor
        Cell data, shape (Nz+2, Ny+2, Nx+2, 5) - [rho, u, v, w, p]
    
    Returns:
    --------
    dt : float
        Time step
    """
    a = torch.sqrt(GAMMA * CELL[:, :, :, 4] / CELL[:, :, :, 0])  # sound speed
    u = CELL[:, :, :, 1]  # x-velocity
    u_max = torch.max(u.abs() + a)
    dt_x = cfl_coefficient * dx / u_max
    return dt_x

def apply_boundary_condition(CELL, normal='x'):
    """
    Apply boundary condition.
    """
    CELL[..., 0, :] = CELL[..., 1, :]
    CELL[..., -1, :] = CELL[..., -2, :]

    return CELL
    