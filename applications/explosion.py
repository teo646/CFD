"""
Explosion simulation using GodunovEuler3D and LinesOnField visualization.
"""
import sys
from pathlib import Path

# Add parent directory to path to enable imports
# This allows importing from CFD package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import GodunovEuler3D from the CFD package
from CFD.Riemann_Solvers_and_Numerical_Methods_for_Fluid_Dynamics_by_TORO.chapter16.godunov_euler import GodunovEuler3D

# Import LinesOnField (assuming it's installed or in path)
try:
    from LinesOnField import Lines
except ImportError:
    # If LinesOnField is not installed, add it to path
    linesonfield_path = Path(__file__).parent.parent.parent / "LinesOnField"
    if linesonfield_path.exists():
        sys.path.insert(0, str(linesonfield_path))
        from LinesOnField import Lines
    else:
        raise ImportError("LinesOnField not found. Please install it or add to path.")


def main():
    """
    Main function to run explosion simulation and visualize with lines.
    """
    # Initialize the solver
    solver = GodunovEuler3D(
        num_cells_x=100,
        num_cells_y=100,
        num_cells_z=100,
        x_domain=[0, 1],
        y_domain=[0, 1],
        z_domain=[0, 1],
        cfl_coefficient=0.8,
        gamma=1.4
    )
    
    # Create explosion initial condition
    CELL = solver.create_explosion_initial_condition(diameter=0.2)
    
    # Run simulation for a few time steps
    t = 0.0
    t_end = 0.1
    
    while t < t_end:
        CELL, dt = solver.update(CELL)
        t += dt
        print(f"t = {t:.4f}, dt = {dt:.6f}")
    
    # Extract field data for visualization
    # CELL shape: (Nz+2, Ny+2, Nx+2, 5) - [rho, u, v, w, p]
    # Remove ghost cells: CELL[1:-1, 1:-1, 1:-1, :]
    field_data = CELL[1:-1, 1:-1, 1:-1, :].cpu().numpy()
    
    # TODO: Use LinesOnField to visualize the field
    # Example:
    # lines = Lines(field_data)
    # lines.visualize()
    
    print("Simulation completed!")
    print(f"Field shape: {field_data.shape}")


if __name__ == "__main__":
    main()
