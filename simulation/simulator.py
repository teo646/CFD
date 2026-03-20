import torch
from .util import volume_render_y

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
        
        #Don't visualize ghost cell
        rho = self.cell[2:-2, 2:-2, 2:-2, 0]
        rho_image, _ = volume_render_y(
                        rho,
                        dy=self.dy,
                        density_scale=6.0,
                        grad_scale=3.0,
                        tone_mapper="log1p",
                        tone_param=12.0,
                        use_gradient=False,
                        density_weight=1.0,
                        grad_weight=0.4,
                        percentile_clip=(0.01, 0.995),
                        white_background=True,
                    )
        
        images.append(rho_image.cpu().numpy())
        

        #Don't visualize ghost cell
        p = self.cell[2:-2, 2:-2, 2:-2, 4]
        p_image, _ = volume_render_y(
                        p,
                        dy=self.dy,
                        density_scale=6.0,
                        grad_scale=3.0,
                        tone_mapper="log1p",
                        tone_param=12.0,
                        use_gradient=True,
                        density_weight=1.0,
                        grad_weight=0.4,
                        percentile_clip=(0.01, 0.995),
                        white_background=False,
                    )
        images.append(p_image.cpu().numpy())

        for visualizer in self.visualizers:
            images.append(visualizer.get_image())

        return images


        