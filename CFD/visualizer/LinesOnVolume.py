import torch
import math
from .visualizer import Visualizer
import cv2
import numpy as np
from noise import pnoise2
from skimage import measure
import random
import cv2
from shapely.geometry import Polygon, MultiPolygon

class LinesOnVolume(Visualizer):
    def __init__(self, x_resolution, y_resolution, lines, x_domain, y_domain, z_domain, max_length=1, anchor = True):
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.lines = lines
        self.max_length = max_length
        self.device = lines.device
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.z_domain = z_domain

        self.anchor = anchor

    @torch.no_grad()
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

    @torch.no_grad()
    def update(self, cell: torch.Tensor, dt: float, dx: float, dy: float, dz: float):
        """
        vector_field: (nz, ny, nx, 3), vector_field[..., 0] = u, [..., 1] = v, [..., 2] = w
        dt: float, 시간 간격
        dx: float, x 방향 grid 간격
        dy: float, y 방향 grid 간격
        dz: float, z 방향 grid 간격
        """

        velocity = cell[..., 1:4]
        if self.lines.numel() == 0:
            return

        nz, ny, nx = velocity.shape[:3]

        # 실제 공간 좌표 (물리 좌표)
        x = self.lines[..., 0]
        y = self.lines[..., 1]
        z = self.lines[..., 2]

        x_min, x_max = self.x_domain
        y_min, y_max = self.y_domain
        z_min, z_max = self.z_domain
        
        x_idx = (x - x_min) / dx
        y_idx = (y - y_min) / dy
        z_idx = (z - z_min) / dz

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

        u = trilerp(velocity[..., 0])
        v = trilerp(velocity[..., 1])
        w = trilerp(velocity[..., 2])

        # 좌표 업데이트 (물리 좌표로)
        if(self.anchor):
            self.lines[:, 1:, 0] = (x + u * dt)[:, 1:]
            self.lines[:, 1:, 1] = (y + v * dt)[:, 1:]
            self.lines[:, 1:, 2] = (z + w * dt)[:, 1:]
        else:
            self.lines[..., 0] = (x + u * dt)
            self.lines[..., 1] = (y + v * dt)
            self.lines[..., 2] = (z + w * dt)

        self.redistribute_points()

    def get_image(self, scale=10):
        """
        Returns
        -------
        img : np.ndarray, uint8, shape (H, W, 3)
            Polyline들을 그린 이미지
        """
        H = int(self.y_resolution * scale)
        W = int(self.x_resolution * scale)

        x_min, x_max = self.x_domain
        y_min, y_max = self.y_domain
        
        px = (self.lines[:, :, 0] - x_min) / (x_max - x_min) * (W - 1)
        py = (self.lines[:, :, 1] - y_min) / (y_max - y_min) * (H - 1)
        # (L, P, 2) in pixel coords (x, y)
        projected_lines = torch.stack(
            [px, py],
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

        if(len(pts_list) == 1):
            cv2.circle(img, tuple(pts_list[0][0]), radius=1, color=(255, 255, 255), thickness=-1)
        else:

            cv2.polylines(
                img,
                pts_list,
                isClosed=False,
                color=(255, 255, 255),
                thickness=7,
                lineType=cv2.LINE_AA,
            )

        if scale != 1:
            img = cv2.resize(img, None, fx=1/scale, fy=1/scale)

        return img

    def rotate(self, theta, phi, center = (0.0, 0.0, 0.0)):
        """
        theta: float, Z축 회전 각도 (yaw)
        phi: float, Y축 회전 각도 (pitch)
        center: (cx, cy, cz) 회전 중심
        """
        device = self.lines.device
        dtype = self.lines.dtype

        th = torch.tensor(theta, device=device, dtype=dtype)
        ph = torch.tensor(phi,   device=device, dtype=dtype)

        cth, sth = torch.cos(th), torch.sin(th)
        cph, sph = torch.cos(ph), torch.sin(ph)

        cx, cy, cz = center

        # --- Rotation matrices ---
        Rz = torch.tensor([
            [ cth, -sth, 0.0, 0.0],
            [ sth,  cth, 0.0, 0.0],
            [ 0.0,  0.0, 1.0, 0.0],
            [ 0.0,  0.0, 0.0, 1.0],
        ], device=device, dtype=dtype)

        Ry = torch.tensor([
            [ cph, 0.0,  sph, 0.0],
            [ 0.0, 1.0,  0.0, 0.0],
            [-sph, 0.0,  cph, 0.0],
            [ 0.0, 0.0,  0.0, 1.0],
        ], device=device, dtype=dtype)

        R = Ry @ Rz  # (4,4)

        # --- Translation matrices ---
        T_neg = torch.tensor([
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 1.0, -cz],
            [0.0, 0.0, 0.0,  1.0],
        ], device=device, dtype=dtype)

        T_pos = torch.tensor([
            [1.0, 0.0, 0.0, cx],
            [0.0, 1.0, 0.0, cy],
            [0.0, 0.0, 1.0, cz],
            [0.0, 0.0, 0.0, 1.0],
        ], device=device, dtype=dtype)

        # 최종 변환 행렬
        M = T_pos @ R @ T_neg  # (4,4)

        # row vector이므로 transpose
        self.lines = self.lines @ M.T

    def add_lines(self, new_lines):
        # if the number of points doesn't match.
        if(not new_lines.shape[1] == self.lines.shape[1]):
            raise IndexError("The number of points should match.")
        self.lines = torch.cat([self.lines, new_lines.to(self.device)], dim=0)


def create_uniform_sphere_points(radius, num_polylines, num_points, x_domain, y_domain, z_domain, center=None, device=None):
    if center is None:
        center_x = (x_domain[0] + x_domain[1]) / 2
        center_y = (y_domain[0] + y_domain[1]) / 2
        center_z = (z_domain[0] + z_domain[1]) / 2
    else:
        center_x, center_y, center_z = center

    # 1. Fibonacci Sphere 알고리즘을 사용하여 균등 분포 점 생성
    # indices: 0부터 num_polylines-1까지의 인덱스
    indices = torch.arange(0, num_polylines, dtype=torch.float32, device=device) + 0.5
    
    # phi (polar angle): arccos를 사용하여 고르게 분포 (Z축 방향 균등)
    # cos(phi)가 [-1, 1] 사이에서 균등하게 분포해야 표면적이 일정함
    phi = torch.acos(1 - 2 * indices / num_polylines)
    
    # theta (azimuth angle): 황금각(Golden Angle)을 이용하여 회전 배치
    # 약 2.399... 라디안 (math.pi * (3 - math.sqrt(5)))
    golden_angle = math.pi * (3 - math.sqrt(5))
    theta = golden_angle * indices

    # 2. 구면 좌표를 직교 좌표로 변환
    # x = r * sin(phi) * cos(theta)
    # y = r * sin(phi) * sin(theta)
    # z = r * cos(phi)
    x = radius * torch.sin(phi) * torch.cos(theta) + center_x
    y = radius * torch.sin(phi) * torch.sin(theta) + center_y
    z = radius * torch.cos(phi) + center_z
    w = torch.ones_like(x, device=device)

    # 3. 도메인 마스킹 (영역을 벗어나는 점 제거)
    positive_mask = (x > x_domain[0]) & (x < x_domain[1]) & \
                    (y > y_domain[0]) & (y < y_domain[1]) & \
                    (z > z_domain[0]) & (z < z_domain[1])

    x = x[positive_mask]
    y = y[positive_mask]
    z = z[positive_mask]
    w = w[positive_mask]

    # 4. 결과 텐서 구성
    # sphere_points_3d: (N_valid, 4)
    sphere_points_3d = torch.stack([x, y, z, w], dim=1) 

    # (N_valid, 1, 4) -> (N_valid, num_points, 4)로 확장
    sphere_points_3d = sphere_points_3d.unsqueeze(1)
    polylines = sphere_points_3d.repeat(1, num_points, 1)
    
    return polylines

def create_random_sphere_points(radius, num_polylines, num_points, x_domain, y_domain, z_domain, center = None, device = None):
    if(center is None):
        center_x = (x_domain[0] + x_domain[1]) / 2
        center_y = (y_domain[0] + y_domain[1]) / 2
        center_z = (z_domain[0] + z_domain[1]) / 2
    else:
        center_x, center_y, center_z = center

    # u, v ~ Uniform(0, 1)
    u = torch.rand(num_polylines, device=device)
    v = torch.rand(num_polylines, device=device)

    theta = 2 * torch.pi * u
    phi = torch.acos(2 * v - 1)

    x = center_x + radius * torch.sin(phi) * torch.cos(theta)
    y = center_y + radius * torch.sin(phi) * torch.sin(theta)
    z = center_z + radius * torch.cos(phi)
    w = torch.ones((num_polylines), device=device)

    positive_mask = (x > x_domain[0]) & (y > y_domain[0]) & (z > z_domain[0])\
                    & (x < x_domain[1]) & (y < y_domain[1]) & (z < z_domain[1])

    x = x[positive_mask]
    y = y[positive_mask]
    z = z[positive_mask]
    w = w[positive_mask]

    sphere_points_3d = torch.stack([x, y, z, w], dim=1) # (NUM_POLYLINE, 4)

    sphere_points_3d = sphere_points_3d.unsqueeze(1)  # (NUM_POLYLINE, 1, 4)
    polylines = sphere_points_3d.repeat(1, num_points, 1)  # (NUM_POLYLINE, NUM_POINTS, 4)
    return polylines

def create_random_circle_points(radius, num_polylines, num_points, x_domain, y_domain, z_domain, center = None, device = None):
    if(center is None):
        center_x = (x_domain[0] + x_domain[1]) / 2
        center_y = (y_domain[0] + y_domain[1]) / 2
        center_z = (z_domain[0] + z_domain[1]) / 2
    else:
        center_x, center_y, center_z = center

    # u, v ~ Uniform(0, 1)
    u = torch.rand(num_polylines, device=device)

    theta = 2 * torch.pi * u

    x = center_x + radius * torch.cos(theta)
    y = center_y + radius * torch.sin(theta)
    z = torch.full_like(x, center_z)
    w = torch.ones((num_polylines), device=device)

    positive_mask = x > 0 & y > 0 & z > 0

    x = x[positive_mask]
    y = y[positive_mask]
    z = z[positive_mask]
    w = w[positive_mask]
    

    circle_points_3d = torch.stack([x, y, z, w], dim=1) # (NUM_POLYLINE, 4)

    circle_points_3d = circle_points_3d.unsqueeze(1)  # (NUM_POLYLINE, 1, 4)
    polylines = circle_points_3d.repeat(1, num_points, 1)  # (NUM_POLYLINE, NUM_POINTS, 4)
    return polylines

def generate_perlin_noise_2d(shape, scale=50.0, octaves=4, persistence=0.5, lacunarity=2.0, seed=0):
    H, W = shape
    # np.vectorize 대신 직접 계산하여 속도 개선
    noise = np.zeros(shape, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            noise[y, x] = pnoise2(
                x / scale, y / scale,
                octaves=octaves, persistence=persistence,
                lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=seed
            )
    
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise

def extract_closed_contours(noise, threshold=0.5, tol=1e-2):
    contours = measure.find_contours(noise, level=threshold)
    return [c - c.mean() for c in contours if np.linalg.norm(c[0] - c[-1]) < tol]

def normalize_contour(contour, target_size=1.0):
    # 컨투어의 중심을 (0,0)으로 이동
    centroid = contour.mean(axis=0)
    centered_contour = contour - centroid
    
    # 현재 컨투어의 최대 반경(또는 bounding box 크기) 계산
    max_dist = np.linalg.norm(centered_contour, axis=1).max()
    
    # 크기를 1로 맞춘 후 target_size 적용
    normalized_contour = (centered_contour / max_dist) * target_size
    return normalized_contour

def get_contours(num, area_range = (2, 4.5)):
    contour_counter = 0
    contours = []
    while contour_counter < num:
        noise = generate_perlin_noise_2d((256, 256), seed=random.randint(0, 10000))
        target_contours = extract_closed_contours(noise)
        for contour in target_contours:
            if(len(contour) < 30):
                continue
            size = random.uniform(area_range[0], area_range[1])
            contour = normalize_contour(contour, target_size=size)
            contours.append(contour)
            contour_counter += 1
            if(contour_counter == num):
                break
    return contours   

def interp_torch(x, xp, fp):
    idx = torch.searchsorted(xp, x) - 1
    idx = torch.clamp(idx, 0, len(xp) - 2)

    t = (x - xp[idx]) / (xp[idx+1] - xp[idx] + 1e-10)
    return (1 - t) * fp[idx] + t * fp[idx+1]


def resample_polyline(polyline, target_dist):
    diffs = torch.diff(polyline, dim=0)
    seg_dists = torch.norm(diffs, dim=1)

    s = torch.cat([
        torch.zeros(1, device=polyline.device),
        torch.cumsum(seg_dists, dim=0)
    ])

    total_len = s[-1]
    n_points = max(2, int(torch.ceil(total_len / target_dist).item()) + 1)

    new_s = torch.linspace(0, total_len, n_points, device=polyline.device)

    resampled = torch.stack([
        interp_torch(new_s, s, polyline[:, i])
        for i in range(polyline.shape[1])
    ], dim=1)

    return resampled

def offset_contour_2d(points, d, device='cpu'):
    """
    2D points (N, 2)를 입력받아 contour를 d만큼 offset합니다.

    반환:
        List[Tensor(N_i, 2)]  # 여러 polygon 가능

    d > 0 : inward
    d < 0 : outward
    """

    if d == 0:
        return [points]

    pts_np = points.detach().cpu().numpy()
    poly = Polygon(pts_np)

    if not poly.is_valid:
        poly = poly.buffer(0)

    offset_poly = poly.buffer(-d)

    if offset_poly.is_empty:
        return []

    polygons = []

    if isinstance(offset_poly, MultiPolygon):
        geoms = offset_poly.geoms
    else:
        geoms = [offset_poly]

    for g in geoms:
        coords = g.exterior.coords
        polygons.append(
            torch.tensor(coords, device=device, dtype=points.dtype)
        )

    return polygons

def resize_contour_2d(points, ratio, device='cpu'):
    points[:, :2] *= ratio 

    return [points]
    
def map_points_to_sphere(points, center, radius, normal=None, ratio=1, device='cpu'):
    points = torch.as_tensor(points, dtype=torch.float32, device=device)
    center = torch.as_tensor(center, dtype=torch.float32, device=device)

    contours = resize_contour_2d(points, ratio, device=device)

    if len(contours) == 0:
        return []

    # normal 설정
    if normal is None:
        normal = torch.randn(3, device=device)
    normal = normal / (torch.norm(normal) + 1e-8)

    tmp = torch.tensor([1.0, 0.0, 0.0], device=device)
    if torch.abs(normal[0]) >= 0.9:
        tmp = torch.tensor([0.0, 1.0, 0.0], device=device)

    t1 = torch.linalg.cross(normal, tmp)
    t1 = t1 / (torch.norm(t1) + 1e-8)
    t2 = torch.linalg.cross(normal, t1)

    results = []

    for pts in contours:
        if pts.shape[0] < 4:
            continue

        pts = pts.to(device=device, dtype=torch.float32)

        # 2D → 3D 평면 위
        points_3d = (
            center
            + radius * normal
            + (pts[:, 0:1] * t1 + pts[:, 1:2] * t2)
        )

        # 구 표면으로 projection
        vec = points_3d - center
        vec = vec / (torch.norm(vec, dim=1, keepdim=True) + 1e-8)

        mapped = center + radius * vec
        results.append(mapped)

    return results

def get_inner_points(contour, dist):
    """
    contour: (N, 2) numpy array [[y, x], ...] (음수 포함 가능)
    dist: 점 사이의 간격 (float)
    """
    # 1. 컨투어의 최소/최대 좌표 계산 (Bounding Box)
    min_coords = np.min(contour, axis=0)
    max_coords = np.max(contour, axis=0)
    
    # 2. 음수 좌표 해결을 위한 이동(Offset) 및 간격(dist)에 따른 스케일링
    # 컨투어의 모든 좌표를 (0, 0) 기준으로 옮기고 dist로 나눕니다.
    shifted_contour = (contour - min_coords) / dist
    
    # 3. 마스크 크기 결정
    # 스케일링된 좌표의 최대값만큼의 크기를 가진 빈 이미지를 만듭니다.
    h_scaled, w_scaled = np.ceil(np.max(shifted_contour, axis=0)).astype(int) + 2
    mask = np.zeros((h_scaled, w_scaled), dtype=np.uint8)
    
    # 4. 컨투어 내부 채우기
    # cv2.fillPoly는 (x, y) 순서를 요구하므로 [y, x] -> [x, y]로 뒤집습니다.
    poly = shifted_contour[:, [1, 0]].astype(np.int32)
    cv2.fillPoly(mask, [poly], 255)
    
    # 5. 마스크 내부의 픽셀 좌표(정수) 추출
    # coords는 [y_idx, x_idx] 형태입니다.
    coords = np.argwhere(mask > 0).astype(np.float32)
    
    # 6. 실제 좌표계로 역변환
    # (인덱스 * dist)를 통해 스케일을 복구하고, 최소 좌표를 다시 더해줍니다.
    real_coords = (coords * dist) + min_coords
    
    return torch.from_numpy(real_coords)

def points_to_voxel(points, shape, x_domain, y_domain, z_domain, device=None):
    
    voxel = torch.zeros(shape, dtype=torch.bool, device=device)
    points = clamp_to_domain(points, x_domain, y_domain, z_domain)
    if(points.numel() == 0):
        return voxel

    # 간격 계산
    dx = (x_domain[1] - x_domain[0]) / shape[2]
    dy = (y_domain[1] - y_domain[0]) / shape[1]
    dz = (z_domain[1] - z_domain[0]) / shape[0]
    
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    z_min, z_max = z_domain
    
    ix = ((points[:, 0] - x_min) / dx).long()
    iy = ((points[:, 1] - y_min) / dy).long()
    iz = ((points[:, 2] - z_min) / dz).long()

    voxel[iz, iy, ix] = True
        
    return voxel

def clamp_to_domain(points, x_domain, y_domain, z_domain):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    valid_mask = (x >= x_domain[0]) & (x < x_domain[1]) & \
                 (y >= y_domain[0]) & (y < y_domain[1]) & \
                 (z >= z_domain[0]) & (z < z_domain[1])

    return points[valid_mask]

def get_random_normal_from_voxel(voxel, center, x_domain, y_domain, z_domain, device='cpu'):
    # 1. True인 셀들의 인덱스 찾기 (Shape: [N, 3])
    # updated_boundary_band는 (Z, Y, X) 순서라고 가정
    indices = torch.where(voxel)
    coords = torch.stack(indices, dim=1) # [[z, y, x], ...]
    
    if coords.shape[0] == 0:
        return None # True인 셀이 없을 경우 처리
    
    # 2. 임의의 인덱스 하나 선택
    random_idx = torch.randint(0, coords.shape[0], (1,)).item()
    selected_indices = coords[random_idx] # [z_idx, y_idx, x_idx]
    
    # 3. 인덱스를 실제 도메인 좌표로 변환
    # 도메인 변수는 (min, max) 형태의 튜플 또는 리스트라고 가정합니다.
    # 인덱스 위치를 도메인 범위에 매핑 (Grid Size로 나누어 비율 계산)
    z_size, y_size, x_size = voxel.shape
    
    # 각 차원별 좌표 계산 (중심점 기준 매핑)
    target_z = z_domain[0] + (selected_indices[0].float() / (z_size - 1)) * (z_domain[1] - z_domain[0])
    target_y = y_domain[0] + (selected_indices[1].float() / (y_size - 1)) * (y_domain[1] - y_domain[0])
    target_x = x_domain[0] + (selected_indices[2].float() / (x_size - 1)) * (x_domain[1] - x_domain[0])
    
    target_pos = torch.tensor([target_x, target_y, target_z], device=device)
    
    # 5. Normal Vector 계산 (Center -> Target)
    normal = target_pos - torch.tensor(center, device=device)
    normal_magnitude = torch.norm(normal)
    
    if normal_magnitude > 0:
        normal = normal / normal_magnitude
    return normal

# 사용 예시:
# x_domain = (0.0, 10.0), y_domain = (0.0, 10.0), z_domain = (0.0, 10.0)
# normal, pos = get_random_normal_from_boundary(updated_boundary_band, x_domain, y_domain, z_domain)

def create_boundary_band_with_contours(
    shape, radius, num_holes, hole_range, polyline_dist, 
    x_domain, y_domain, z_domain, center=None, 
    device='cuda', contour_ratios = [1]
):
    if(center is None):
        center = ((x_domain[1] + x_domain[0]) / 2, (y_domain[1] + y_domain[0]) / 2, (z_domain[1] + z_domain[0]) / 2)
        
    sphere_points = create_uniform_sphere_points(radius, 100000, 1, x_domain, y_domain, z_domain, center=center, device=device)
    sphere_points = sphere_points[:, 0, :3]
    
    boundary_band = points_to_voxel(sphere_points, shape, x_domain, y_domain, z_domain, device=device)
    
    updated_boundary_band = boundary_band.clone()

    sphere_contours = []    
    contours = get_contours(num_holes, area_range = hole_range)
    for contour in contours:
        hole_points = get_inner_points(contour, dist=0.0004)
        
        max_normal_trials = 10
        for counter in range(max_normal_trials):
            normal = get_random_normal_from_voxel(updated_boundary_band, center, x_domain, y_domain, z_domain, device=device)
            if normal is None:
                return sphere_contours, updated_boundary_band
                
            hole_pts3d = map_points_to_sphere(
                hole_points, center=center, radius=radius,
                device=device, normal=normal
            )[0]
            voxel_hole = points_to_voxel(
                hole_pts3d, shape, x_domain, y_domain, z_domain, device=device
            )
            if ((not torch.any(voxel_hole & (boundary_band & ~updated_boundary_band))) and torch.any(voxel_hole) ): 
                break
                
        if counter == max_normal_trials - 1:
            continue
        
        updated_boundary_band = updated_boundary_band & ~voxel_hole

        for contour_ratio in contour_ratios:
            
            contour_pts3ds = map_points_to_sphere(
                contour, center=center, radius=radius,
                ratio=contour_ratio, device=device, normal=normal
            )

            for contour_pts3d in contour_pts3ds:
                
                contour_pts3d = resample_polyline(contour_pts3d, polyline_dist)
                contour_pts3d = clamp_to_domain(
                    contour_pts3d, x_domain, y_domain, z_domain
                )
        
                sphere_contours.append(contour_pts3d)

    return sphere_contours, updated_boundary_band # 필요한 경우 points도 반환
    
    


    


    
    
    
