import torch
import math
from .visualizer import Visualizer
import cv2
import numpy as np

class LinesOnVolume(Visualizer):
    def __init__(self, x_resolution, y_resolution, lines, max_length=100):
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.lines = lines
        self.max_length = max_length
        self.device = lines.device

    def redistribute_points(self):
        """
        각 polyline의 점 수를 유지하면서 점들의 위치를 일정 간격으로 재조정.
        w는 그대로 1로 유지됨.
        """
        if self.lines.numel() == 0:
            return
        
        num_polylines, NUM_POINTS, _ = self.lines.shape

        # 실제 공간 좌표만 사용 (x,y,z)
        xyz = self.lines[:, :, :3]

        # segment 벡터 (B, N-1, 3)
        diffs = xyz[:, 1:, :] - xyz[:, :-1, :]

        # segment 길이 (B, N-1)
        dists = torch.linalg.norm(diffs, dim=2)

        # 누적 아크 길이 (B, N)
        cumdist = torch.cat(
            [torch.zeros((num_polylines, 1), device=self.device), torch.cumsum(dists, dim=1)],
            dim=1
        )

        total_len = cumdist[:, -1:]  # (B,1)

        new_len = torch.minimum(total_len, torch.tensor(self.max_length, device=self.device))

        # 새로운 점들의 위치 (등간격) (B, N)
        new_pos = torch.linspace(0, 1, NUM_POINTS, device=self.device).unsqueeze(0) * new_len

        # 각 new_pos가 속한 segment 찾기 (B, N)
        idx = torch.searchsorted(cumdist, new_pos, right=True) - 1
        idx = torch.clamp(idx, 0, NUM_POINTS - 2)

        # seg 시작점 (B, N, 3)
        seg_start = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

        # seg 벡터 (B, N, 3)
        seg_vec = torch.gather(diffs, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

        # seg 길이 (B, N)
        seg_len = torch.gather(dists, 1, idx)

        seg_offset = new_pos - torch.gather(cumdist, 1, idx)  # (B, N)
        ratio = (seg_offset / (seg_len + 1e-8)).unsqueeze(-1)  # (B, N, 1)

        # 보간 결과 (B, N, 3)
        resampled_xyz = seg_start + seg_vec * ratio

        # 마지막 좌표 w=1 추가
        self.lines[..., 0:3] = resampled_xyz

    def update(self, vector_field: torch.Tensor, dt: float, dx: float, dy: float, dz: float):
        """
        vector_field: (nz, ny, nx, 3), vector_field[..., 0] = u, [..., 1] = v, [..., 2] = w
        dt: float, 시간 간격
        dx: float, x 방향 grid 간격
        dy: float, y 방향 grid 간격
        dz: float, z 방향 grid 간격
        """
        if self.lines.numel() == 0:
            return

        nz, ny, nx = vector_field.shape[:3]

        # 실제 공간 좌표 (물리 좌표)
        x = self.lines[..., 0]
        y = self.lines[..., 1]
        z = self.lines[..., 2]

        # 물리 좌표를 grid index로 변환
        x_idx = x / dx
        y_idx = y / dy
        z_idx = z / dz

        # periodic index 적용
        x0 = torch.floor(x_idx).long() % nx
        y0 = torch.floor(y_idx).long() % ny
        z0 = torch.floor(z_idx).long() % nz
        x1 = (x0 + 1) % nx
        y1 = (y0 + 1) % ny
        z1 = (z0 + 1) % nz

        # 보간 가중치 (grid index 내에서의 위치)
        sx = x_idx - torch.floor(x_idx)
        sy = y_idx - torch.floor(y_idx)
        sz = z_idx - torch.floor(z_idx)

        # ===== trilinear interpolation =====
        def trilerp(field):
            c000 = field[z0, y0, x0]
            c100 = field[z0, y0, x1]
            c010 = field[z0, y1, x0]
            c110 = field[z0, y1, x1]
            c001 = field[z1, y0, x0]
            c101 = field[z1, y0, x1]
            c011 = field[z1, y1, x0]
            c111 = field[z1, y1, x1]

            c00 = c000 * (1 - sx) + c100 * sx
            c01 = c001 * (1 - sx) + c101 * sx
            c10 = c010 * (1 - sx) + c110 * sx
            c11 = c011 * (1 - sx) + c111 * sx

            c0 = c00 * (1 - sy) + c10 * sy
            c1 = c01 * (1 - sy) + c11 * sy

            return c0 * (1 - sz) + c1 * sz

        u = trilerp(vector_field[..., 0])
        v = trilerp(vector_field[..., 1])
        w = trilerp(vector_field[..., 2])

        # 좌표 업데이트 (물리 좌표로)
        self.lines[:, 1:, 0] = (x + u * dt)[:, 1:]
        self.lines[:, 1:, 1] = (y + v * dt)[:, 1:]
        self.lines[:, 1:, 2] = (z + w * dt)[:, 1:]

        self.redistribute_points()

    def get_image(self):
        """
        Returns
        -------
        img : np.ndarray, uint8, shape (H, W, 3)
            Polyline들을 그린 이미지
        """
        H = int(self.y_resolution)
        W = int(self.x_resolution)

        # (L, P, 2) in pixel coords (x, y)
        projected_lines = torch.stack(
            [self.lines[:, :, 0] * W, self.lines[:, :, 1] * H],
            dim=-1
        ).detach().cpu()

        # numpy int32
        pts = np.round(projected_lines.numpy()).astype(np.int32)

        # (선택) clip: 화면 밖 좌표가 많을 때 안전/성능에 도움
        pts[..., 0] = np.clip(pts[..., 0], 0, W - 1)  # x
        pts[..., 1] = np.clip(pts[..., 1], 0, H - 1)  # y

        # blank image
        img = np.zeros((H, W, 3), dtype=np.uint8)

        # OpenCV polylines expects list of (P,1,2)
        pts_list = [p.reshape(-1, 1, 2) for p in pts]

        cv2.polylines(
            img,
            pts_list,
            isClosed=False,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_8,  # 빠름 (더 예쁘게는 cv2.LINE_AA)
        )

        return img

def create_uniform_sphere_points(radius, num_polylines, num_points, x_domain, y_domain, z_domain, device):
    center_x = (x_domain[0] + x_domain[1]) / 2
    center_y = (y_domain[0] + y_domain[1]) / 2
    center_z = (z_domain[0] + z_domain[1]) / 2

    # 구면 좌표계를 사용하여 구 표면의 점들 생성
    # 균등하게 분포된 점들을 생성하기 위해 균등 격자 사용
    # NUM_POLYLINE개의 점을 생성
    num_theta = int(math.sqrt(num_polylines * math.pi))  # azimuth 방향 격자 수
    num_phi = int(num_polylines / num_theta)  # polar 방향 격자 수

    # 균등하게 분포된 구면 좌표 생성
    theta = torch.linspace(0, 2*math.pi, num_theta, device=device)  # azimuth angle (0 to 2π)
    phi = torch.linspace(0, math.pi, num_phi, device=device)  # polar angle (0 to π)

    # 격자 생성
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
    theta_flat = theta_grid.flatten()[:num_polylines]  # NUM_POLYLINE개만 선택
    phi_flat = phi_grid.flatten()[:num_polylines]

    # 구면 좌표를 직교 좌표로 변환
    # x = r * sin(φ) * cos(θ) + center_x
    # y = r * sin(φ) * sin(θ) + center_y
    # z = r * cos(φ) + center_z
    x = radius * torch.sin(phi_flat) * torch.cos(theta_flat) + center_x
    y = radius * torch.sin(phi_flat) * torch.sin(theta_flat) + center_y
    z = radius * torch.cos(phi_flat) + center_z

    # (NUM_POLYLINE, 3) 형태로 결합
    sphere_points = torch.stack([x, y, z], dim=1)  # (NUM_POLYLINE, 3)

    # w 좌표 추가 (homogeneous coordinate)
    w = torch.ones((sphere_points.shape[0], 1), device=device)  # (NUM_POLYLINE, 1)
    sphere_points_3d = torch.cat([sphere_points, w], dim=1)  # (NUM_POLYLINE, 4)

    # 각 polyline에 대해 NUM_POINTS개의 동일한 좌표를 가진 점들 생성
    # (NUM_POLYLINE, 1, 4)로 unsqueeze 후 repeat
    sphere_points_3d = sphere_points_3d.unsqueeze(1)  # (NUM_POLYLINE, 1, 4)
    polylines = sphere_points_3d.repeat(1, num_points, 1)  # (NUM_POLYLINE, NUM_POINTS, 4)
    return polylines

def create_random_sphere_points(radius, num_polylines, num_points, x_domain, y_domain, z_domain, device):
    center_x = (x_domain[0] + x_domain[1]) / 2
    center_y = (y_domain[0] + y_domain[1]) / 2
    center_z = (z_domain[0] + z_domain[1]) / 2

    # u, v ~ Uniform(0, 1)
    u = torch.rand(num_polylines, device=device)
    v = torch.rand(num_polylines, device=device)

    theta = 2 * torch.pi * u
    phi = torch.acos(2 * v - 1)

    x = center_x + radius * torch.sin(phi) * torch.cos(theta)
    y = center_y + radius * torch.sin(phi) * torch.sin(theta)
    z = center_z + radius * torch.cos(phi)
    w = torch.ones((num_polylines), device=device)

    sphere_points_3d = torch.stack([x, y, z, w], dim=1) # (NUM_POLYLINE, 4)

    sphere_points_3d = sphere_points_3d.unsqueeze(1)  # (NUM_POLYLINE, 1, 4)
    polylines = sphere_points_3d.repeat(1, num_points, 1)  # (NUM_POLYLINE, NUM_POINTS, 4)
    return polylines

def create_random_circle_points(radius, num_polylines, num_points, x_domain, y_domain, device):
    center_x = (x_domain[0] + x_domain[1]) / 2
    center_y = (y_domain[0] + y_domain[1]) / 2

    # u, v ~ Uniform(0, 1)
    u = torch.rand(num_polylines, device=device)

    theta = 2 * torch.pi * u

    x = center_x + radius * torch.cos(theta)
    y = center_y + radius * torch.sin(theta)
    z = torch.zeros_like(x)
    w = torch.ones((num_polylines), device=device)

    circle_points_3d = torch.stack([x, y, z, w], dim=1) # (NUM_POLYLINE, 4)

    circle_points_3d = circle_points_3d.unsqueeze(1)  # (NUM_POLYLINE, 1, 4)
    polylines = circle_points_3d.repeat(1, num_points, 1)  # (NUM_POLYLINE, NUM_POINTS, 4)
    return polylines
