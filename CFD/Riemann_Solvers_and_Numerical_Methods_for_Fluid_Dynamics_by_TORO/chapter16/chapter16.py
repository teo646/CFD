import torch
from CFD import W_to_U, U_to_W

def sweep(CELL, dx, dt, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='x'):
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
    WL, WR = reconstruction_method(CELL, dx, dt, GAMMA, normal=normal)
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
    U_new = U_cell + (dt / dx) * (flux[flux_L] - flux[flux_R])

    # 4) write back + apply BC
    CELL[cell_slice] = U_to_W(U_new, GAMMA)
    CELL = boundary_function(CELL, normal=normal)

    return CELL

def cal_dt_split(CELL, cfl_coefficient, dx, dy, dz, GAMMA):
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
    v = CELL[:, :, :, 2]  # y-velocity
    w = CELL[:, :, :, 3]  # z-velocity
    u_max = torch.max(u.abs() + a)
    v_max = torch.max(v.abs() + a)
    w_max = torch.max(w.abs() + a)
    dt_x = cfl_coefficient * dx / u_max
    dt_y = cfl_coefficient * dy / v_max
    dt_z = cfl_coefficient * dz / w_max
    return torch.min(torch.min(dt_x, dt_y), dt_z)

def strang_update(CELL, cfl_coefficient, dx, dy, dz, reconstruction_method, riemann_solver, boundary_function, GAMMA, dimension=2):
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
        dt = cal_dt_split(CELL, cfl_coefficient, dx, dy, 1e6, GAMMA)
        CELL = sweep(CELL, dx, dt * 0.5, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='x')
        CELL = sweep(CELL, dy, dt, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='y')
        CELL = sweep(CELL, dx, dt * 0.5, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='x')
    elif(dimension == 3):
        dt = cal_dt_split(CELL, cfl_coefficient, dx, dy, dz, GAMMA)
        CELL = sweep(CELL, dx, dt * 0.5, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='x')
        CELL = sweep(CELL, dy, dt * 0.5, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='y')
        CELL = sweep(CELL, dz, dt, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='z')
        CELL = sweep(CELL, dy, dt * 0.5, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='y')
        CELL = sweep(CELL, dx, dt * 0.5, reconstruction_method, riemann_solver, boundary_function, GAMMA, normal='x')
    else:
        raise ValueError(f"dimension must be '2d' or '3d', got {dimension!r}")

    return CELL, dt