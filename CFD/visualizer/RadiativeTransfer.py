import torch
from .visualizer import Visualizer
from .AdvectingField import AdvectingField


class RadiativeTransfer(Visualizer):
    def __init__(self, shape, k_a, k_s, R_star, T_ref, soot_mask, device=None):
        super().__init__()

        nz, ny, nx = shape

        self.k_a = k_a
        self.k_s = k_s
        self.R_star = R_star
        self.T_ref = T_ref

        self.device = device

        self.radiative_cell = torch.zeros((nz, ny, nx, 6), device=device)
        self.tmp_for_transfer = torch.zeros_like(self.radiative_cell)

        self.soot_mask = soot_mask
        soot_field = torch.zeros(shape, device=device)
        soot_field[self.soot_mask] = 1

        self.soot = AdvectingField(soot_field)
        self.image = torch.zeros((ny, nx), device=device)
        self.cell = None
        self.dt = None
        self.dx = None
        self.dy = None
        self.dz = None

    def compute_transfer_weights(self):
        mu_a = self.k_a * self.soot.field
        mu_s = self.k_s * self.soot.field
        mu_t = mu_a + mu_s

        scatter_ratio = mu_s / (mu_t + 1e-8)

        wt_x = torch.exp(-mu_t * self.dx)
        wt_y = torch.exp(-mu_t * self.dy)
        wt_z = torch.exp(-mu_t * self.dz)

        ws_x = scatter_ratio * (1.0 - wt_x) / 4.0
        ws_y = scatter_ratio * (1.0 - wt_y) / 4.0
        ws_z = scatter_ratio * (1.0 - wt_z) / 4.0

        return wt_x, ws_x, wt_y, ws_y, wt_z, ws_z

    def transfer(self, wt_x, ws_x, wt_y, ws_y, wt_z, ws_z):
        self.tmp_for_transfer.zero_()

        # +x direction transfer
        input_v = self.radiative_cell[:, :, :-1, 0]
        self.tmp_for_transfer[:, :, 1:, 0] += input_v * wt_x[:, :, 1:]
        self.tmp_for_transfer[:, :, 1:, 2:] += (input_v * ws_x[:, :, 1:]).unsqueeze(-1)

        # -x direction transfer
        input_v = self.radiative_cell[:, :, 1:, 1]
        self.tmp_for_transfer[:, :, :-1, 1] += input_v * wt_x[:, :, :-1]
        self.tmp_for_transfer[:, :, :-1, 2:] += (input_v * ws_x[:, :, :-1]).unsqueeze(-1)

        # +y direction transfer
        input_v = self.radiative_cell[:, :-1, :, 2]
        self.tmp_for_transfer[:, 1:, :, 2] += input_v * wt_y[:, 1:, :]
        self.tmp_for_transfer[:, 1:, :, [0, 1, 4, 5]] += (input_v * ws_y[:, 1:, :]).unsqueeze(-1)

        # -y direction transfer
        input_v = self.radiative_cell[:, 1:, :, 3]
        self.tmp_for_transfer[:, :-1, :, 3] += input_v * wt_y[:, :-1, :]
        self.tmp_for_transfer[:, :-1, :, [0, 1, 4, 5]] += (input_v * ws_y[:, :-1, :]).unsqueeze(-1)

        # +z direction transfer
        input_v = self.radiative_cell[:-1, :, :, 4]
        self.tmp_for_transfer[1:, :, :, 4] += input_v * wt_z[1:, :, :]
        self.tmp_for_transfer[1:, :, :, :4] += (input_v * ws_z[1:, :, :]).unsqueeze(-1)

        # -z direction transfer
        input_v = self.radiative_cell[1:, :, :, 5]
        self.tmp_for_transfer[:-1, :, :, 5] += input_v * wt_z[:-1, :, :]
        self.tmp_for_transfer[:-1, :, :, :4] += (input_v * ws_z[:-1, :, :]).unsqueeze(-1)

        self.image += self.radiative_cell[-1, :, :, 4]
        self.radiative_cell.copy_(self.tmp_for_transfer)

    @torch.no_grad()
    def update(self, cell, dt, dx, dy, dz):
        self.soot.update(cell, dt, dx, dy, dz)
        self.soot.field[self.soot_mask] = 1

        #save parameters for later use.
        self.cell, self.dt, self.dx, self.dy, self.dz = cell, dt, dx, dy, dz

    @torch.no_grad()
    def get_image(self, scale=10):

        rho_safe = torch.clamp(self.cell[..., 0], min=1e-4)
        p = self.cell[..., 4]

        # 이상기체식이면 보통 T = p / (rho * R)
        T = p / (rho_safe * self.R_star)

        # Stefan-Boltzmann-like emission
        cell_volume = self.dx * self.dy * self.dz
        emission = (T.clamp_min(0.0) ** 4) * cell_volume * (T > self.T_ref)
        source = emission.unsqueeze(-1).expand(-1, -1, -1, 6)
        initial_max = torch.max(source)

        self.image.zero_()
        self.radiative_cell.copy_(source)

        wt_x, ws_x, wt_y, ws_y, wt_z, ws_z = self.compute_transfer_weights()

        nz, ny, nx, _ = self.radiative_cell.shape

        for _ in range(nz + ny + nx):
            self.transfer(wt_x, ws_x, wt_y, ws_y, wt_z, ws_z)

            if torch.max(self.radiative_cell) < initial_max * 1e-3:
                break
                
        return self.image.cpu()