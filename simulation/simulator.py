import torch
from CFD import gradient_scalar_field

class Simulator:

    def __init__(
        self,
        cell,
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
        visualizers=[],
        solid_cell = None
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
        self.cell = cell

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
        
        self.solid_cell = solid_cell
        self.visualizers = visualizers

    @torch.no_grad()
    def update(self):

        self.cell, dt = self.update_method(self.cell, self.cfl_coefficient,\
                                        self.dx, self.dy, self.dz,\
                                        self.reconstruction_method,\
                                        self.riemann_solver,\
                                        self.boundary_function,
                                        self.GAMMA, dimension=self.dimension, 
                                        solid_cell = self.solid_cell)
        for visualizer in self.visualizers:
            visualizer.update(self.cell, dt, self.dx, self.dy, self.dz)
        return dt

    def get_images_num(self):
        # return the number of images to be displayed
        # 1 for density
        # len(self.visualizers) for visualizers
        return 2 + len(self.visualizers)

    def get_images(self):
        images = []
        
        rho = self.cell[..., 0]
        rho_image = torch.sum(rho, dim=0)
        rho_image /= torch.max(rho_image)
        rho_image = rho_image.detach().cpu().numpy()
        images.append(rho_image)

        p = self.cell[..., 4]
        grad_p = gradient_scalar_field(p, self.dx, self.dy, self.dz)
        dpdx = grad_p[0]
        dpdy = grad_p[1]
        dpdz = grad_p[2]

        grad_mag = torch.sqrt(dpdx**2 + dpdy**2 + dpdz**2)
        grad_mag_image = torch.sum(grad_mag, dim=0)
        grad_mag_image /= torch.max(grad_mag_image)
        grad_mag_image = grad_mag_image.detach().cpu().numpy()
        images.append(grad_mag_image)

        for visualizer in self.visualizers:
            images.append(visualizer.get_image())

        return images


        