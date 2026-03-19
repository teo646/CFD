import torch
from CFD import W_to_U, U_to_W

def apply_solid_cell(WL, WR, solid_cell, normal='x'):

    if(normal == 'x'):
        WL_solid_cell = solid_cell.narrow(-1, 1, solid_cell.size(-1) - 3)
        WR_solid_cell = solid_cell.narrow(-1, 2, solid_cell.size(-1) - 3)
    elif(normal == 'y'):
        WL_solid_cell = solid_cell.narrow(-2, 1, solid_cell.size(-2) - 3)
        WR_solid_cell = solid_cell.narrow(-2, 2, solid_cell.size(-2) - 3)
    elif(normal == 'z'):
        WL_solid_cell = solid_cell.narrow(-3, 1, solid_cell.size(-3) - 3)
        WR_solid_cell = solid_cell.narrow(-3, 2, solid_cell.size(-3) - 3)
    else:
        raise ValueError(f"normal must be 'x', 'y', or 'z', got {normal!r}")

    WL_reflective = WR.clone()
    WR_reflective = WL.clone()
    if(normal == 'x'):
        WL_reflective[...,1] *= -1
        WR_reflective[...,1] *= -1
    elif(normal == 'y'):
        WL_reflective[...,2] *= -1
        WR_reflective[...,2] *= -1
    elif(normal == 'z'):
        WL_reflective[...,3] *= -1
        WR_reflective[...,3] *= -1
    else:
        raise ValueError(f"normal must be 'x', 'y', or 'z', got {normal!r}")

    WL = torch.where(WL_solid_cell.unsqueeze(-1), WL_reflective, WL)
    WR = torch.where(WR_solid_cell.unsqueeze(-1), WR_reflective, WR)

    return WL, WR

def sweep(CELL, ds, dt, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='x', solid_cell = None):
    """
    Perform one directional sweep.

    Parameters
    ----------
    CELL : torch.Tensor
        (Nz+2, Ny+2, Nx+2, 5) primitive [rho,u,v,w,p]
    dt : float
        timestep
    normal : str
        'x' or 'y' or 'z'

    Returns
    -------
    CELL : torch.Tensor
        updated CELL
    """
    # 1) reconstruction + riemann flux
    WL, WR = reconstruction_method(CELL, ds, dt, GAMMA, normal=normal)
    # over write WL and WR to put obstacle
    if(solid_cell is not None):
        WL, WR = apply_solid_cell(WL, WR, solid_cell, normal=normal)


    flux = riemann_solver(WL, WR, GAMMA, normal=normal)

    # 2) pick spacing and slicing/update pattern
    if normal == 'x':
        cell_slice = (slice(None), slice(None), slice(2, -2), slice(None))
        # flux shape: (Nz+2, Ny+2, Nx+1, 5)
        flux_L = (slice(None), slice(None), slice(None, -1), slice(None))
        flux_R = (slice(None), slice(None), slice(1, None), slice(None))

    elif normal == 'y':
        cell_slice = (slice(None), slice(2, -2), slice(None), slice(None))
        # flux shape: (Nz+2, Ny+1, Nx+2, 5)
        flux_L = (slice(None), slice(None, -1), slice(None), slice(None))
        flux_R = (slice(None), slice(1, None), slice(None), slice(None))

    elif normal == 'z':
        cell_slice = (slice(2, -2), slice(None), slice(None), slice(None))
        # flux shape: (Nz+1, Ny+2, Nx+2, 5)
        flux_L = (slice(None, -1), slice(None), slice(None), slice(None))
        flux_R = (slice(1, None), slice(None), slice(None), slice(None))

    else:
        raise ValueError(f"normal must be 'x', 'y', or 'z', got {normal!r}")

    # 3) conservative update on the interior
    U_cell = W_to_U(CELL[cell_slice], GAMMA)
    U_new = U_cell + (dt / ds) * (flux[flux_L] - flux[flux_R])

    # 4) write back + apply BC
    CELL[cell_slice] = U_to_W(U_new, GAMMA)
    CELL = boundary_function(CELL, normal=normal)

    # 5) apply solid cell
    if(solid_cell is not None):
        CELL[solid_cell] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1e-6], device=CELL.device, dtype=CELL.dtype)

    return CELL

def cal_dt_split(CELL, cfl_coefficient, dx, dy, dz, GAMMA, solid_cell=None):

    rho = CELL[..., 0]
    u   = CELL[..., 1]
    v   = CELL[..., 2]
    w   = CELL[..., 3]
    p   = CELL[..., 4]

    # --- sound speed ---
    a = torch.sqrt(GAMMA * p / rho)

    if solid_cell is not None:
        fluid_mask = ~solid_cell

        if fluid_mask.sum() == 0:
            raise RuntimeError("All cells are solid. Cannot compute dt.")

        u_eff = (u.abs() + a)[fluid_mask]
        v_eff = (v.abs() + a)[fluid_mask]
        w_eff = (w.abs() + a)[fluid_mask]

        u_max = u_eff.max()
        v_max = v_eff.max()
        w_max = w_eff.max()

    else:
        u_eff = u.abs() + a
        v_eff = v.abs() + a
        w_eff = w.abs() + a

        u_max = torch.max(u_eff)
        v_max = torch.max(v_eff)
        w_max = torch.max(w_eff)

    # --- dt 계산 ---
    dt_x = cfl_coefficient * dx / u_max
    dt_y = cfl_coefficient * dy / v_max
    dt_z = cfl_coefficient * dz / w_max

    dt = torch.min(torch.min(dt_x, dt_y), dt_z)

    return dt

def strang_update(CELL, cfl_coefficient, dx, dy, dz, reconstruction_method, riemann_solver, boundary_function, GAMMA, dimension=2, solid_cell = None):
    """
    Update the solution using dimensional splitting.
    
    Parameters:
    -----------
    CELL : torch.Tensor
        Cell data, shape (Nz+2, Ny+2, Nx+2, 5) - [rho, u, v, w, p] (with ghost cells)
    
    Returns:
    --------
    CELL : torch.Tensor
        Updated cell data
    dt : float
        Time step used
    """

    if(dimension == 2):
        #put dz very large to not let it create minimum dt
        dt = cal_dt_split(CELL, cfl_coefficient, dx, dy, 1e6, GAMMA, solid_cell = solid_cell)
        CELL = sweep(CELL, dx, dt * 0.5, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='x', solid_cell = solid_cell )
        CELL = sweep(CELL, dy, dt, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='y', solid_cell = solid_cell )
        CELL = sweep(CELL, dx, dt * 0.5, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='x', solid_cell = solid_cell )
    elif(dimension == 3):
        dt = cal_dt_split(CELL, cfl_coefficient, dx, dy, dz, GAMMA, solid_cell = solid_cell)
        CELL = sweep(CELL, dx, dt * 0.5, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='x', solid_cell = solid_cell )
        CELL = sweep(CELL, dy, dt * 0.5, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='y', solid_cell = solid_cell )
        CELL = sweep(CELL, dz, dt, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='z', solid_cell = solid_cell )
        CELL = sweep(CELL, dy, dt * 0.5, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='y', solid_cell = solid_cell )
        CELL = sweep(CELL, dx, dt * 0.5, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='x', solid_cell = solid_cell )
    else:
        raise ValueError(f"dimension must be '2d' or '3d', got {dimension!r}")

    return CELL, dt