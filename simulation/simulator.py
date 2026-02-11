import torch

class Simulator:

    def __init__(
        self,
        dx,
        dy,
        dz,
        riemann_solver,
        reconstruction_method,
        boundary_function,
        update_method,
        dimension,
        cfl_coefficient=0.8,
        GAMMA=1.4,
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
        
        # Grid spacing
        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.riemann_solver = riemann_solver
        self.reconstruction_method = reconstruction_method
        self.boundary_function = boundary_function
        self.update_method = update_method

        self.dimension = dimension
        
        # Physical constants
        self.cfl_coefficient = cfl_coefficient
        self.GAMMA = GAMMA

    def update(self, CELL):

        CELL, dt = self.update_method(CELL, self.cfl_coefficient,\
                                        self.dx, self.dy, self.dz,\
                                        self.reconstruction_method,\
                                        self.riemann_solver,\
                                        self.boundary_function,
                                        self.GAMMA, dimension=self.dimension)
        return CELL, dt