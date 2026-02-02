"""
3D Euler equations solver using Godunov's method with exact Riemann solver.
"""
import numpy as np
import torch
import random



class GodunovEuler3D:
    """
    3D Euler equations solver using Godunov's method with exact Riemann solver.
    
    Solves the 3D Euler equations using dimensional splitting and exact Riemann solver.
    Primitive variables: [rho, u, v, w, p] (density, x-velocity, y-velocity, z-velocity, pressure)
    """
    
    def __init__(
        self,
        num_cells_x=100,
        num_cells_y=100,
        num_cells_z=100,
        x_domain=[0, 1],
        y_domain=[0, 1],
        z_domain=[0, 1],
        cfl_coefficient=0.8,
        GAMMA=1.4,
        tol=1e-6,
        device=None
    ):
        """
        Initialize the 3D Euler solver.
        
        Parameters:
        -----------
        num_cells_x, num_cells_y, num_cells_z : int
            Number of cells in each direction
        x_domain, y_domain, z_domain : list
            Domain boundaries [min, max] for each direction
        cfl_coefficient : float
            CFL coefficient for time step calculation
        gamma : float
            Ratio of specific heats
        tol : float
            Tolerance for Newton-Raphson iteration
        device : torch.device, optional
            Device to run computations on. If None, uses CUDA if available, else CPU.
        """
        # Domain parameters
        self.num_cells_x = num_cells_x
        self.num_cells_y = num_cells_y
        self.num_cells_z = num_cells_z
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.z_domain = z_domain
        
        # Grid spacing
        self.dx = (x_domain[1] - x_domain[0]) / num_cells_x
        self.dy = (y_domain[1] - y_domain[0]) / num_cells_y
        self.dz = (z_domain[1] - z_domain[0]) / num_cells_z
        
        # Physical constants
        self.cfl_coefficient = cfl_coefficient
        self.GAMMA = GAMMA
        self.tol = tol
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Move tolerance to device
        self.tol_tensor = torch.tensor(self.tol, device=self.device)
    
    def compute_f_and_df(self, p, W_L, W_R):
        """
        W_L, W_R: (..., 4) - [rho, u, v, p] primitive variables
        p: (...,) - pressure guess
        """
        left_rho = W_L[..., 0]
        left_p = W_L[..., 4]
        right_rho = W_R[..., 0]
        right_p = W_R[..., 4]
        
        Al = 2 / (self.GAMMA + 1) / left_rho
        Ar = 2 / (self.GAMMA + 1) / right_rho
        Bl = (self.GAMMA - 1) / (self.GAMMA + 1) * left_p
        Br = (self.GAMMA - 1) / (self.GAMMA + 1) * right_p
        al = torch.sqrt(self.GAMMA * left_p / left_rho)
        ar = torch.sqrt(self.GAMMA * right_p / right_rho)
        
        left_shock_cell = p > left_p
        right_shock_cell = p > right_p

        #left rarefaction wave
        fl = 2 * al / (self.GAMMA - 1) * ((p/left_p)**((self.GAMMA - 1)/(2 * self.GAMMA)) - 1)
        d_fl = 1 / left_rho / al * (p/left_p)**(-(self.GAMMA + 1)/(2 * self.GAMMA))
    
        #overide left shock wave
        fl[left_shock_cell] = ((p - left_p) * torch.sqrt(Al / (p + Bl)))[left_shock_cell]
        d_fl[left_shock_cell] = (torch.sqrt(Al / (Bl + p)) * (1 - 0.5 * (p - left_p) / (Bl + p)))[left_shock_cell]


        #right rarefaction wave
        fr = 2 * ar / (self.GAMMA - 1) * ((p/right_p)**((self.GAMMA - 1)/(2 * self.GAMMA)) - 1)
        d_fr = 1 / right_rho / ar * (p/right_p)**(-(self.GAMMA + 1)/(2 * self.GAMMA))
        
        #overide right shock wave
        fr[right_shock_cell] = ((p - right_p)* torch.sqrt(Ar / (p + Br)))[right_shock_cell]
        d_fr[right_shock_cell] = (torch.sqrt(Ar / (Br + p)) * (1 - 0.5 * (p - right_p) / (Br + p)))[right_shock_cell]

        return fl, d_fl, fr, d_fr
    
    def solve_riemann_star_state(self, W_L, W_R, normal='x'):
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
        count = 0
        while(True):
            count += 1
            prev_p = p
            fl, d_fl, fr, d_fr = self.compute_f_and_df(p, W_L, W_R)

            f = fl + fr  + right_u - left_u
            df = d_fl + d_fr

            p = p - f / df
            #음압 방지.
            p = torch.clamp(p, min=torch.tensor(1e-12, device=p.device))
            #모든 셀에서 충족하면 종료.
            tol_tensor = torch.tensor(self.tol, device=p.device)
            if(torch.all(2 * abs(p - prev_p) < tol_tensor * (p + prev_p)) or count > 1000):
                break
        
        fl, d_fl, fr, d_fr = self.compute_f_and_df(p, W_L, W_R)
        u = 0.5 * (left_u + right_u + fr - fl)

        left_shock_cell = p > left_p
        right_shock_cell = p > right_p

        rho_l_star = left_rho * (p / left_p) ** (1 / self.GAMMA)
        rho_l_star[left_shock_cell] = (left_rho * (self.GAMMA * (p + left_p) - left_p + p) /
                                    (self.GAMMA * (p + left_p) - p + left_p))[left_shock_cell]

        rho_r_star = right_rho * (p / right_p) ** (1 / self.GAMMA)
        rho_r_star[right_shock_cell] = (right_rho * (self.GAMMA * (p + right_p) - right_p + p) /
                                    (self.GAMMA * (p + right_p) - p + right_p))[right_shock_cell]

        return p, u, rho_l_star, rho_r_star

    def riemann_flux(self, W_L, W_R, normal='x'):
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
        
        p_star, u_star, rho_l_star, rho_r_star = self.solve_riemann_star_state(W_L, W_R, normal=normal)

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
        if torch.any(left_contact & left_rarefaction):
            al = torch.sqrt(self.GAMMA * left_p / left_rho)
            s_hl = left_u - al

            # Region 1: left state
            mask_l1 = left_contact & left_rarefaction & (s < s_hl) 
            if torch.any(mask_l1):
                rho[mask_l1] = left_rho[mask_l1]
                u[mask_l1] = left_u[mask_l1]
                p[mask_l1] = left_p[mask_l1]

            # Region 2: star left
            al_star = al * (p_star / left_p) ** ((self.GAMMA - 1) / (2 * self.GAMMA))
            s_tl = u_star - al_star
            mask_l2 = left_contact & left_rarefaction & (s > s_tl)
            if torch.any(mask_l2):
                rho[mask_l2] = rho_l_star[mask_l2]
                u[mask_l2] = u_star[mask_l2]
                p[mask_l2] = p_star[mask_l2]

            # Region 3: inside fan
            mask_l3 = left_contact & left_rarefaction & ~(s < s_hl) & ~(s > s_tl)
            if torch.any(mask_l3):
                p[mask_l3] = left_p[mask_l3] * ((2 * al[mask_l3] + (self.GAMMA - 1) * (left_u[mask_l3] - s[mask_l3])) / (al[mask_l3] * (self.GAMMA + 1))) ** (2 * self.GAMMA / (self.GAMMA - 1))
                u[mask_l3] = 2 / (self.GAMMA + 1) * (al[mask_l3] + (self.GAMMA - 1) / 2 * left_u[mask_l3] + s[mask_l3])
                rho[mask_l3] = left_rho[mask_l3] * ((2 * al[mask_l3] + (self.GAMMA - 1) * (left_u[mask_l3] - s[mask_l3])) / (al[mask_l3] * (self.GAMMA + 1))) ** (2 / (self.GAMMA - 1))

        # -------- Left shock --------
        if torch.any(left_contact & left_shock):
            al = torch.sqrt(self.GAMMA * left_p / left_rho)
            s_l = left_u - al * torch.sqrt((self.GAMMA * (p_star + left_p) + p_star - left_p) / (2 * self.GAMMA * left_p))
            mask_ls = left_contact & left_shock & (s < s_l)
            if torch.any(mask_ls):
                rho[mask_ls] = left_rho[mask_ls]
                u[mask_ls] = left_u[mask_ls]
                p[mask_ls] = left_p[mask_ls]

            mask_ls2 = left_contact & left_shock & ~(s < s_l)
            if torch.any(mask_ls2):
                rho[mask_ls2] = rho_l_star[mask_ls2]
                u[mask_ls2] = u_star[mask_ls2]
                p[mask_ls2] = p_star[mask_ls2]

        # -------- Right rarefaction --------
        if torch.any(right_contact & right_rarefaction):
            ar = torch.sqrt(self.GAMMA * right_p / right_rho)
            s_hr = right_u + ar

            # Region 1: right state
            mask_r1 = right_contact & right_rarefaction & (s > s_hr)
            if torch.any(mask_r1):
                rho[mask_r1] = right_rho[mask_r1]
                u[mask_r1] = right_u[mask_r1]
                p[mask_r1] = right_p[mask_r1]

            # Region 2: star right
            ar_star = ar * (p_star / right_p) ** ((self.GAMMA - 1) / (2 * self.GAMMA))
            s_tr = u_star + ar_star
            mask_r2 = right_contact & right_rarefaction & (s < s_tr)
            if torch.any(mask_r2):
                rho[mask_r2] = rho_r_star[mask_r2]
                u[mask_r2] = u_star[mask_r2]
                p[mask_r2] = p_star[mask_r2]

            # Region 3: inside fan
            mask_r3 = right_contact & right_rarefaction & ~(s > s_hr) & ~(s < s_tr)
            if torch.any(mask_r3):
                p[mask_r3] = right_p[mask_r3] * ((2 * ar[mask_r3] + (self.GAMMA - 1) * (s[mask_r3] - right_u[mask_r3])) / (ar[mask_r3] * (self.GAMMA + 1))) ** (2 * self.GAMMA / (self.GAMMA - 1))
                u[mask_r3] = 2 / (self.GAMMA + 1) * (-ar[mask_r3] + (self.GAMMA - 1) / 2 * right_u[mask_r3] + s[mask_r3])
                rho[mask_r3] = right_rho[mask_r3] * ((2 * ar[mask_r3] + (self.GAMMA - 1) * (s[mask_r3] - right_u[mask_r3])) / (ar[mask_r3] * (self.GAMMA + 1))) ** (2 / (self.GAMMA - 1))

        # -------- Right shock --------
        if torch.any(right_contact & right_shock):
            ar = torch.sqrt(self.GAMMA * right_p / right_rho)
            s_r = right_u + ar * torch.sqrt((self.GAMMA * (p_star + right_p) + p_star - right_p) / (2 * self.GAMMA * right_p))
            mask_rs = right_contact & right_shock & (s > s_r)
            if torch.any(mask_rs):
                rho[mask_rs] = right_rho[mask_rs]
                u[mask_rs] = right_u[mask_rs]
                p[mask_rs] = right_p[mask_rs]

            mask_rs2 = right_contact & right_shock & ~(s > s_r)
            if torch.any(mask_rs2):
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
        E = p / (self.GAMMA - 1) + 0.5 * rho * (u**2 + v**2 + w**2)
        
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
    
    def W_to_U(self, W):
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
        E = p / (self.GAMMA - 1) + 0.5 * rho * (u**2 + v**2 + w**2)
        return torch.stack([rho, u * rho, v * rho, w * rho, E], dim=-1)
    
    def U_to_W(self, U):
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
        p = (self.GAMMA - 1) * (E - 0.5 * rho * (u**2 + v**2 + w**2))
        p = torch.clamp(p, min=1e-10)
        return torch.stack([rho, u, v, w, p], dim=-1)
    
    def cal_dt(self, CELL):
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
        a = torch.sqrt(self.GAMMA * CELL[:, :, :, 4] / CELL[:, :, :, 0])  # sound speed
        u = CELL[:, :, :, 1]  # x-velocity
        v = CELL[:, :, :, 2]  # y-velocity
        w = CELL[:, :, :, 3]  # z-velocity
        u_max = torch.max(u.abs() + a)
        v_max = torch.max(v.abs() + a)
        w_max = torch.max(w.abs() + a)
        dt_x = self.cfl_coefficient * self.dx / u_max
        dt_y = self.cfl_coefficient * self.dy / v_max
        dt_z = self.cfl_coefficient * self.dz / w_max
        return torch.min(torch.min(dt_x, dt_y), dt_z)
    
    def sweep_x(self, CELL, dt):
        """
        Perform x-direction sweep.
        
        Parameters:
        -----------
        CELL : torch.Tensor
            Cell data, shape (Nz+2, Ny+2, Nx+2, 5) - [rho, u, v, w, p]
        dt : float
            Time step
        
        Returns:
        --------
        CELL : torch.Tensor
            Updated cell data
        """
        # X-direction sweep: solve Riemann problems along x-direction for each y
        flux_x = self.riemann_flux(CELL[:, :, :-1, :], CELL[:, :, 1:, :], normal='x')

        U_cell = self.W_to_U(CELL[:, :, 1:-1, :])
        
        # X-direction update
        U_new = U_cell + dt/self.dx * (flux_x[:, :, :-1, :] - flux_x[:, :, 1:, :])
        
        # Convert back to primitive
        CELL[:, :, 1:-1, :] = self.U_to_W(U_new)
        
        # Apply boundary conditions in x-direction
        CELL[:, :, 0, :] = CELL[:, :, 1, :]
        CELL[:, :, -1, :] = CELL[:, :, -2, :]

        return CELL
    
    def sweep_y(self, CELL, dt):
        """
        Perform y-direction sweep.
        
        Parameters:
        -----------
        CELL : torch.Tensor
            Cell data, shape (Nz+2, Ny+2, Nx+2, 5) - [rho, u, v, w, p]
        dt : float
            Time step
        
        Returns:
        --------
        CELL : torch.Tensor
            Updated cell data
        """
        # Y-direction sweep: solve Riemann problems along y-direction for each x
        flux_y = self.riemann_flux(CELL[:, :-1, :, :], CELL[:, 1:, :, :], normal='y')
        
        # Update in y-direction
        U_cell = self.W_to_U(CELL[:, 1:-1, :, :])
        
        # Y-direction update
        U_new = U_cell + dt/self.dy * (flux_y[:, :-1, :, :] - flux_y[:, 1:, :, :])
        
        # Final update
        CELL[:, 1:-1, :, :] = self.U_to_W(U_new)
        
        # Apply boundary conditions in y-direction
        CELL[:, 0, :, :] = CELL[:, 1, :, :]
        CELL[:, -1, :, :] = CELL[:, -2, :, :]
        
        return CELL
    
    def sweep_z(self, CELL, dt):
        """
        Perform z-direction sweep.
        
        Parameters:
        -----------
        CELL : torch.Tensor
            Cell data, shape (Nz+2, Ny+2, Nx+2, 5) - [rho, u, v, w, p]
        dt : float
            Time step
        
        Returns:
        --------
        CELL : torch.Tensor
            Updated cell data
        """
        # Z-direction sweep: solve Riemann problems along z-direction for each x
        flux_z = self.riemann_flux(CELL[:-1, :, :, :], CELL[1:, :, :, :], normal='z')
        
        # Update in z-direction
        U_cell = self.W_to_U(CELL[1:-1, :, :, :])
        
        # Z-direction update
        U_new = U_cell + dt/self.dz * (flux_z[:-1, :, :, :] - flux_z[1:, :, :, :])
        
        # Final update
        CELL[1:-1, :, :, :] = self.U_to_W(U_new)
        
        # Apply boundary conditions in z-direction
        CELL[0, :, :, :] = CELL[1, :, :, :]
        CELL[-1, :, :, :] = CELL[-2, :, :, :]
        
        return CELL
    
    def update(self, CELL):
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
        dt = self.cal_dt(CELL).item()

        sweep_order = [self.sweep_x, self.sweep_y, self.sweep_z]
        random.shuffle(sweep_order)

        CELL = sweep_order[0](CELL, dt * 0.5)
        CELL = sweep_order[1](CELL, dt * 0.5)
        CELL = sweep_order[2](CELL, dt)
        CELL = sweep_order[1](CELL, dt * 0.5)
        CELL = sweep_order[0](CELL, dt * 0.5)
        
        return CELL, dt
    
    def create_explosion_initial_condition(self, rho_inner=1.0, p_inner=1.0, 
                                          rho_outer=0.125, p_outer=0.1, sigma = 0.1):
        """
        Create explosion initial condition.
        
        Parameters:
        -----------
        diameter : float
            Diameter of the explosion region
        rho_inner, p_inner : float
            Density and pressure inside the explosion region
        rho_outer, p_outer : float
            Density and pressure outside the explosion region
        
        Returns:
        --------
        CELL : torch.Tensor
            Initial cell data, shape (Nz+2, Ny+2, Nx+2, 5) - [rho, u, v, w, p]
        """
        # +2 for cell boundary (ghost cells)
        # Shape: (Nz + 2, Ny + 2, Nx + 2, 5) - [rho, u, v, w, p]
        CELL = torch.zeros((self.num_cells_z + 2, self.num_cells_y + 2, self.num_cells_x + 2, 5), 
                          device=self.device)

        # 중심 좌표 (도메인 중앙)
        center_x = (self.x_domain[0] + self.x_domain[1]) / 2
        center_y = (self.y_domain[0] + self.y_domain[1]) / 2
        center_z = (self.z_domain[0] + self.z_domain[1]) / 2

        # 각 셀의 중심 좌표 계산 (ghost cell 제외한 실제 셀만)
        x_coords = torch.linspace(self.x_domain[0] + self.dx/2, self.x_domain[1] - self.dx/2, 
                                 self.num_cells_x, device=self.device)
        y_coords = torch.linspace(self.y_domain[0] + self.dy/2, self.y_domain[1] - self.dy/2, 
                                 self.num_cells_y, device=self.device)
        z_coords = torch.linspace(self.z_domain[0] + self.dz/2, self.z_domain[1] - self.dz/2, 
                                 self.num_cells_z, device=self.device)
        X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='xy')

        # 중심으로부터의 거리 계산
        distances2 = (X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2
        # === Smooth Gaussian Profile ===
        # exp(-r²/(2σ²)) 형태
        gaussian_profile = torch.exp(-distances2 / (2 * sigma**2))

        # 기본값 설정 (외부 영역) - ghost cell 포함 전체
        CELL[:, :, :, 0] = rho_outer # rho (low density)
        CELL[:, :, :, 4] = p_outer    # p (low pressure)

        # 폭발 영역 설정 (고압, 고밀도) - 실제 셀만 (ghost cell 제외)
        CELL[1:-1, 1:-1, 1:-1, 0] += (rho_inner - rho_outer) * gaussian_profile    # rho (high density)
        CELL[1:-1, 1:-1, 1:-1, 4] += (p_inner - p_outer) * gaussian_profile     # p (high pressure)

        return CELL
