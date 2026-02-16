import torch
from CFD import W_to_F
from CFD import W_to_U

def HLLC_Riemann_Solver(W_L, W_R, GAMMA, normal='x'):
    VEL_IDX = {
    'x': (1,2,3),
    'y': (2,3,1),
    'z': (3,1,2),
    }
    n, t1, t2 = VEL_IDX[normal]

    rho_l = W_L[..., 0].unsqueeze(-1)
    rho_r = W_R[..., 0].unsqueeze(-1)
    u_l = W_L[..., n].unsqueeze(-1)
    u_r = W_R[..., n].unsqueeze(-1)
    p_l = W_L[..., 4].unsqueeze(-1)
    p_r = W_R[..., 4].unsqueeze(-1)

    # pressure estimate
    rho_mean = (rho_l + rho_r) / 2
    a_l = torch.sqrt(GAMMA * p_l / rho_l)
    a_r = torch.sqrt(GAMMA * p_r / rho_r)
    a_mean = 0.5 * (a_l + a_r)

    p_pvrs = 0.5 * (p_l + p_r) - 0.5 * (u_r - u_l) * rho_mean * a_mean
    p_star = torch.clamp(p_pvrs, min = 0)

    # wave speed estimate
    q_l = torch.ones_like(rho_l)
    q_l[p_star > p_l] = torch.sqrt(1 + (GAMMA + 1) / (2 * GAMMA) * (p_star / p_l - 1))[p_star > p_l]

    q_r = torch.ones_like(rho_r)
    q_r[p_star > p_r] = torch.sqrt(1 + (GAMMA + 1) / (2 * GAMMA) * (p_star / p_r - 1))[p_star > p_r]

    S_l = (u_l - a_l * q_l)
    S_r = (u_r + a_r * q_r)

    S_star = (p_r - p_l + rho_l * u_l * (S_l - u_l) - rho_r * u_r * (S_r - u_r)) / (rho_l * (S_l - u_l) - rho_r * (S_r - u_r))

    #Flux calculation for left and right states
    F_l = W_to_F(W_L, GAMMA, normal=normal)
    U_l = W_to_U(W_L, GAMMA)
    F_star_l = S_star * (S_l * U_l - F_l)
    COEFF_l = S_l * (p_l + rho_l * (S_l - u_l) * (S_star - u_l))
    F_star_l[..., n] += COEFF_l.squeeze(-1)
    F_star_l[..., 4] += (COEFF_l * S_star).squeeze(-1)
    F_star_l /= (S_l - S_star)

    F_r = W_to_F(W_R, GAMMA, normal=normal)
    U_r = W_to_U(W_R, GAMMA)
    F_star_r = S_star * (S_r * U_r - F_r)
    COEFF_r = S_r * (p_r + rho_r * (S_r - u_r) * (S_star - u_r))
    F_star_r[..., n] += (COEFF_r).squeeze(-1)
    F_star_r[..., 4] += (COEFF_r * S_star).squeeze(-1)
    F_star_r /= (S_r - S_star)

    mask_l      = (0 <= S_l)
    mask_star_l = (S_l <= 0) & (0 <= S_star)
    mask_star_r = (S_star <= 0) & (0 <= S_r)

    F_hllc = torch.where(mask_l,      F_l,
            torch.where(mask_star_l, F_star_l,
            torch.where(mask_star_r, F_star_r,
                        F_r)))

    return F_hllc