import torch
import torch.nn.functional as F
from .visualizer import Visualizer

class Dye(Visualizer):
    def __init__(self, dye_field: torch.Tensor):
        self.dye_field = dye_field
        device = dye_field.device
        dtype = dye_field.dtype

        # indices: (nz, ny, nx, 3) with (z, y, x) in last dim
        coord_dtype = torch.float32  # 좌표는 fp32 권장
        z = torch.arange(dye_field.shape[0], device=device, dtype=coord_dtype)
        y = torch.arange(dye_field.shape[1], device=device, dtype=coord_dtype)
        x = torch.arange(dye_field.shape[2], device=device, dtype=coord_dtype)
        Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
        self.indices = torch.stack([Z, Y, X], dim=-1)  # fp32

    @torch.no_grad()
    def advect_dye_field(
        self,
        velocity: torch.Tensor,   # (nz, ny, nx, 3) with (u, v, w) = (x,y,z) components
        dt: float,
        dx: float,
        dy: float,
        dz: float,
        filter_epsilon=1e-2,
        mode="bilinear",
    ):
        nz, ny, nx = self.dye_field.shape

        # indices components (z,y,x)
        X = self.indices[..., 2]; Y = self.indices[..., 1]; Z = self.indices[..., 0]

        U = velocity[..., 0].to(torch.float32)
        V = velocity[..., 1].to(torch.float32)
        W = velocity[..., 2].to(torch.float32)

        Xs = X - U * (dt / dx)
        Ys = Y - V * (dt / dy)
        Zs = Z - W * (dt / dz)

        # nx,ny,nz==1 분기/보호 (최소한 clamp)
        denx = max(nx - 1, 1)
        deny = max(ny - 1, 1)
        denz = max(nz - 1, 1)

        x_norm = 2.0 * Xs / denx - 1.0
        y_norm = 2.0 * Ys / deny - 1.0
        z_norm = torch.full_like(x_norm, -1.0) if nz == 1 else (2.0 * Zs / denz - 1.0)

        grid = torch.stack([x_norm, y_norm, z_norm], dim=-1).unsqueeze(0)  # fp32

        field_in = self.dye_field.unsqueeze(0).unsqueeze(0)
        advected = F.grid_sample(field_in, grid.to(field_in.dtype),
                                align_corners=True, mode=mode, padding_mode="border"
        ).squeeze(0).squeeze(0)

        # dt-의존 필터로 바꾸고 싶으면:
        # eps = torch.exp(torch.tensor(-dt/tau, device=advected.device, dtype=advected.dtype))
        # self.dye_field = advected * (1 - eps) + self.dye_field * eps
        self.dye_field = advected * (1 - filter_epsilon) + self.dye_field * filter_epsilon

    def update(self, cell, dt, dx, dy, dz):
        # cell[..., 1:4] assumed (u,v,w) = (x,y,z)
        self.advect_dye_field(cell[..., 1:4], dt, dx, dy, dz)

    def get_image(self):
        image = torch.sum(self.dye_field, dim=0)
        m = image.max().clamp_min(1e-12)
        image = (image / m).detach().cpu().numpy()
        return image
