import os
import time
from pathlib import Path
import yaml
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from jax_fem.solver import solver, ad_wrapper
# 导入最新版 JAX-FEM 的核心组件
from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh
from jax_fem.problem import Problem
import meshio
from jax_fem.utils import save_sol
from jax_fem.mma import MMA
import jax.scipy.ndimage as jndimage
import jax.scipy.signal as jsignal
from jax_fem.generate_mesh import rectangle_mesh, get_meshio_cell_type, Mesh
from dataclasses import dataclass, field
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def apply_density_filter(theta_2d, mask, sigma=1.0):
    """JAX 原生实现：带边缘修正的 2D 高斯滤波器"""
    radius = int(3.0 * sigma + 0.5)
    x = jnp.arange(-radius, radius + 1)
    
    kernel_1d = jnp.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / jnp.sum(kernel_1d)
    
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    
    # 分子：只对 mask 内部有意义的值进行卷积
    filtered_theta = jsignal.convolve(theta_2d * mask, kernel_2d, mode='same')
    
    # 🔥 分母：用 mask 替代 ones_2d，只在设计域内归一化
    edge_weights = jsignal.convolve(mask, kernel_2d, mode='same')
    
    # 防止除以 0（空域的 edge_weights 可能是 0）
    edge_weights = jnp.clip(edge_weights, a_min=1e-6)
    
    return filtered_theta / edge_weights

def build_l_shape_mask(Nx, Ny, Lx, Ly):
    """
    生成 L 型设计域 mask
    1 = 设计域
    0 = 空域（禁止材料）
    """
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)  # 注意 shape = (Ny, Nx)

    mask = np.ones_like(X)

    # 👉 这里定义 L 型（你可以随便改）
    # 切掉右上角
    mask[(X > 0.5) & (Y > 0.5)] = 0

    return mask.astype(np.float32)

def build_density2feature_input_image(filtered_theta_flat, Nx, Ny):
    """
    将 filtered_theta 转成与旧版 pt 握手完全一致的 density2feature 输入图像。
    返回 shape = [Ny, Nx] 的 clean_image
    """
    final_density_jax = jnp.reshape(filtered_theta_flat, (Nx, Ny))
    image_2d = np.array(final_density_jax).T

    # 与旧版握手保持一致
    min_val = image_2d.min()
    max_val = image_2d.max()
    clean_image = (image_2d - min_val) / (max_val - min_val + 1e-8)

    lower_bound = 0.0
    upper_bound = 0.9
    clean_image = np.clip(
        (clean_image - lower_bound) / (upper_bound - lower_bound),
        0.0,
        1.0,
    )

    return clean_image



def save_density2feature_pt(
    clean_image_filtered: np.ndarray,
    clean_image_prior: np.ndarray,
    mask_np: np.ndarray,          # 🔥 新增参数：接收 mask
    pt_path: str,
    Nx: int,
    Ny: int,
):
    import torch
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)

    image_tensor = torch.from_numpy(clean_image_filtered).float().unsqueeze(0)
    image_prior_tensor = torch.from_numpy(clean_image_prior).float().unsqueeze(0)
    
    # 🔥 新增：把 mask 也转成 tensor 存进去
    mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)

    save_dict = {
        "image": image_tensor,
        "image_prior": image_prior_tensor,
        "design_mask": mask_tensor,    # 🔥 写入大礼包
        "nelx": Nx,
        "nely": Ny,
        "is_real_test": True,
    }
    torch.save(save_dict, pt_path)

@dataclass
class SwitchConfig:
    # 至少跑这么多步之后才允许切换，避免过早切
    min_iter: int = 25

    # 统计窗口长度
    window: int = 4

    # compliance 最近几步平均相对改进率阈值
    improve_thresh: float = 0.008

    # binaryness 阈值：越大越接近 0/1
    binary_thresh: float = 0.72

    # raw/filter density 差异阈值
    gap_thresh: float = 0.06

    # 连续满足多少次条件才触发切换
    patience: int = 2

    # 是否打印调试信息
    verbose: bool = True
    
    

class SwitchMonitor:
    def __init__(self, cfg: SwitchConfig):
        self.cfg = cfg
        self.obj_hist = deque(maxlen=cfg.window + 1)
        self.binary_hist = deque(maxlen=cfg.window)
        self.gap_hist = deque(maxlen=cfg.window)
        self.hit_count = 0

    @staticmethod
    def compute_binaryness(filtered_theta_flat):
        rho = np.asarray(filtered_theta_flat, dtype=np.float64)
        # 1 - 4*rho*(1-rho) 的均值，范围大致在 [0,1]
        return float(1.0 - 4.0 * np.mean(rho * (1.0 - rho)))

    @staticmethod
    def compute_filter_gap(theta_flat, filtered_theta_flat):
        x = np.asarray(theta_flat, dtype=np.float64).reshape(-1)
        y = np.asarray(filtered_theta_flat, dtype=np.float64).reshape(-1)
        return float(np.mean(np.abs(x - y)))

    def _mean_relative_improve(self):
        if len(self.obj_hist) < 2:
            return None
        vals = list(self.obj_hist)
        improves = []
        for a, b in zip(vals[:-1], vals[1:]):
            denom = abs(a) + 1e-8
            improves.append((a - b) / denom)
        return float(np.mean(improves))

    def update(self, iter_idx, obj, theta_flat, filtered_theta_flat):
        obj = float(np.asarray(obj))
        binaryness = self.compute_binaryness(filtered_theta_flat)
        gap = self.compute_filter_gap(theta_flat, filtered_theta_flat)

        self.obj_hist.append(obj)
        self.binary_hist.append(binaryness)
        self.gap_hist.append(gap)

        mean_improve = self._mean_relative_improve()

        ready = (
            iter_idx >= self.cfg.min_iter and
            mean_improve is not None and
            len(self.binary_hist) >= self.cfg.window and
            len(self.gap_hist) >= self.cfg.window
        )

        should_switch = False
        cond_progress = False
        cond_binary = False
        cond_gap = False

        if ready:
            mean_binary = float(np.mean(self.binary_hist))
            mean_gap = float(np.mean(self.gap_hist))

            cond_progress = mean_improve < self.cfg.improve_thresh
            cond_binary = mean_binary > self.cfg.binary_thresh
            cond_gap = mean_gap < self.cfg.gap_thresh

            if cond_progress and cond_binary and cond_gap:
                self.hit_count += 1
            else:
                self.hit_count = 0

            if self.hit_count >= self.cfg.patience:
                should_switch = True
        else:
            mean_binary = None
            mean_gap = None

        stats = {
            "obj": obj,
            "mean_improve": mean_improve,
            "binaryness": binaryness,
            "gap": gap,
            "mean_binary": mean_binary,
            "mean_gap": mean_gap,
            "cond_progress": cond_progress,
            "cond_binary": cond_binary,
            "cond_gap": cond_gap,
            "hit_count": self.hit_count,
            "should_switch": should_switch,
        }

        return should_switch, stats

from typing import List, Union, Tuple, Callable
from dataclasses import dataclass

@dataclass
class ProblemConfig:
    name: str
    meshfile: str

    # 几何尺寸与离散
    Lx: float
    Ly: float
    Nx: int
    Ny: int

    # FMTO 风格边界条件
    # fixed_x: [[x_const, [y0, y1]], ...] 或 [[x, y], ...]
    # fixed_y: [[x_const, [y0, y1]], ...] 或 [[x, y], ...]
    # fixed_xy: [[x_const, [y0, y1]], ...] 或 [[x, y], ...]
    fixed_x: list
    fixed_y: list
    fixed_xy: list

    # loads: [ [[x, y], [fx, fy]], ... ]
    loads: list

    # 优化参数
    vol_frac: float = 0.5
    filter_sigma: float = 1.0

    # 将点载荷近似成一个小区域边载荷时的容差
    load_region_eps_x: float = 0.03
    load_region_eps_y: float = 0.03

    # density2feature / backend 握手尺寸
    d2f_Lx: float = 0.6
    d2f_Ly: float = 0.2

def load_problem_config(yaml_path: str) -> ProblemConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ProblemConfig(
        name=data["name"],
        meshfile=data.get("meshfile", ""),

        Lx=float(data["Lx"]),
        Ly=float(data["Ly"]),
        Nx=int(data["Nx"]),
        Ny=int(data["Ny"]),

        fixed_x=data.get("fixed_x", []),
        fixed_y=data.get("fixed_y", []),
        fixed_xy=data.get("fixed_xy", []),

        loads=data.get("loads", []),

        vol_frac=float(data.get("vol_frac", 0.5)),
        filter_sigma=float(data.get("filter_sigma", 1.0)),

        load_region_eps_x=float(data.get("load_region_eps_x", 0.03)),
        load_region_eps_y=float(data.get("load_region_eps_y", 0.03)),

        d2f_Lx=float(data.get("d2f_Lx", 0.6)),
        d2f_Ly=float(data.get("d2f_Ly", 0.2)),
    )



def _is_scalar(v):
    return isinstance(v, (int, float, np.floating))

def make_point_selector(x0, y0, atol=1e-6):
    def fn(point):
        return jnp.logical_and(
            jnp.isclose(point[0], x0, atol=atol),
            jnp.isclose(point[1], y0, atol=atol),
        )
    return fn

def make_vertical_segment_selector(x_const, y_range, atol=1e-6):
    y0, y1 = float(y_range[0]), float(y_range[1])
    y_min, y_max = min(y0, y1), max(y0, y1)

    def fn(point):
        return jnp.logical_and(
            jnp.isclose(point[0], x_const, atol=atol),
            jnp.logical_and(point[1] >= y_min - atol, point[1] <= y_max + atol),
        )
    return fn

def make_horizontal_segment_selector(y_const, x_range, atol=1e-6):
    x0, x1 = float(x_range[0]), float(x_range[1])
    x_min, x_max = min(x0, x1), max(x0, x1)

    def fn(point):
        return jnp.logical_and(
            jnp.isclose(point[1], y_const, atol=atol),
            jnp.logical_and(point[0] >= x_min - atol, point[0] <= x_max + atol),
        )
    return fn

def parse_constraint_entry(entry, prefer_axis="x", atol=1e-6):
    """
    支持三类格式：
    1. [x_const, [y0, y1]]   -> 竖直边上一段
    2. [x, y]                -> 单点
    3. [[x0, x1], y_const]   -> 水平边上一段
    """
    if len(entry) != 2:
        raise ValueError(f"Invalid constraint entry: {entry}")

    a, b = entry[0], entry[1]

    # 情况 1: a 是标量, b 是区间 (通常为竖直边段)
    if _is_scalar(a) and isinstance(b, (list, tuple)) and len(b) == 2:
        if prefer_axis == "x":
            return make_vertical_segment_selector(float(a), b, atol=atol)
        elif prefer_axis == "y":
            return make_horizontal_segment_selector(float(a), b, atol=atol)
        else:
            raise ValueError(f"Unknown prefer_axis={prefer_axis}")

    # 🔥 情况 2: a 是区间, b 是标量 (新增：水平边段)
    if isinstance(a, (list, tuple)) and len(a) == 2 and _is_scalar(b):
        # 此时 a 是 x_range, b 是 y_const
        return make_horizontal_segment_selector(float(b), a, atol=atol)

    # 情况 3: a 和 b 都是标量 (单点约束)
    if _is_scalar(a) and _is_scalar(b):
        return make_point_selector(float(a), float(b), atol=atol)

    raise ValueError(f"Unsupported constraint format: {entry}")

def zero_val(point):
    return 0.0

def build_dirichlet_bc_info(cfg: ProblemConfig, atol=1e-6):
    location_fns = []
    vec_ids = []
    val_fns = []

    # fixed_x -> 约束 x 位移
    for entry in cfg.fixed_x:
        location_fns.append(parse_constraint_entry(entry, prefer_axis="x", atol=atol))
        vec_ids.append(0)
        val_fns.append(zero_val)

    # fixed_y -> 约束 y 位移
    for entry in cfg.fixed_y:
        # 如果是 [x, y] 点约束，也能正常识别
        # 如果以后出现 [y_const, [x0, x1]] 的水平边约束，可把格式单独扩一下
        location_fns.append(parse_constraint_entry(entry, prefer_axis="x", atol=atol))
        vec_ids.append(1)
        val_fns.append(zero_val)

    # fixed_xy -> 同时约束 x,y
    for entry in cfg.fixed_xy:
        loc_fn = parse_constraint_entry(entry, prefer_axis="x", atol=atol)
        location_fns.append(loc_fn)
        vec_ids.append(0)
        val_fns.append(zero_val)

        location_fns.append(loc_fn)
        vec_ids.append(1)
        val_fns.append(zero_val)

    return [location_fns, vec_ids, val_fns]

def make_segment_load_selector(region_desc, cfg: ProblemConfig, atol=1e-6):
    """
    支持两种分布载荷格式：

    1. [[x0, x1], y_const]
       水平边界上一段：x in [x0, x1], y = y_const

    2. [x_const, [y0, y1]]
       竖直边界上一段：x = x_const, y in [y0, y1]
    """
    a, b = region_desc

    # 水平边界区间：[[x0, x1], y_const]
    if isinstance(a, (list, tuple)) and len(a) == 2 and _is_scalar(b):
        x0, x1 = float(a[0]), float(a[1])
        y_const = float(b)
        x_min, x_max = min(x0, x1), max(x0, x1)

        def fn(point):
            return jnp.logical_and(
                jnp.isclose(point[1], y_const, atol=atol),
                jnp.logical_and(point[0] >= x_min - atol, point[0] <= x_max + atol),
            )
        return fn

    # 竖直边界区间：[x_const, [y0, y1]]
    if _is_scalar(a) and isinstance(b, (list, tuple)) and len(b) == 2:
        x_const = float(a)
        y0, y1 = float(b[0]), float(b[1])
        y_min, y_max = min(y0, y1), max(y0, y1)

        def fn(point):
            return jnp.logical_and(
                jnp.isclose(point[0], x_const, atol=atol),
                jnp.logical_and(point[1] >= y_min - atol, point[1] <= y_max + atol),
            )
        return fn

    raise ValueError(f"Unsupported distributed load region format: {region_desc}")

def make_load_region_selector(load_point, cfg: ProblemConfig):
    x0, y0 = float(load_point[0]), float(load_point[1])
    ex, ey = cfg.load_region_eps_x, cfg.load_region_eps_y

    # 如果载荷点在上边界/下边界/左边界/右边界附近，
    # 就把它近似成该边界上一小段 Neumann 区域
    if np.isclose(y0, cfg.Ly):
        def fn(point):
            return jnp.logical_and(
                jnp.isclose(point[1], cfg.Ly, atol=1e-6),
                jnp.logical_and(point[0] >= x0 - ex, point[0] <= x0 + ex),
            )
        return fn

    if np.isclose(y0, 0.0):
        def fn(point):
            return jnp.logical_and(
                jnp.isclose(point[1], 0.0, atol=1e-6),
                jnp.logical_and(point[0] >= x0 - ex, point[0] <= x0 + ex),
            )
        return fn

    if np.isclose(x0, 0.0):
        def fn(point):
            return jnp.logical_and(
                jnp.isclose(point[0], 0.0, atol=1e-6),
                jnp.logical_and(point[1] >= y0 - ey, point[1] <= y0 + ey),
            )
        return fn

    if np.isclose(x0, cfg.Lx):
        def fn(point):
            return jnp.logical_and(
                jnp.isclose(point[0], cfg.Lx, atol=1e-6),
                jnp.logical_and(point[1] >= y0 - ey, point[1] <= y0 + ey),
            )
        return fn

    raise ValueError(
        f"Current SIMP front-end only supports boundary loads. "
        f"Got load at interior point {load_point}."
    )

def build_loads(cfg: ProblemConfig):
    location_fns = []
    traction_fns = []

    for load in cfg.loads:
        region_desc, load_vec = load[0], load[1]
        fx, fy = float(load_vec[0]), float(load_vec[1])

        # 情况 1：点载荷格式 [[x, y], [fx, fy]]
        if (
            isinstance(region_desc, (list, tuple))
            and len(region_desc) == 2
            and _is_scalar(region_desc[0])
            and _is_scalar(region_desc[1])
        ):
            loc_fn = make_load_region_selector(region_desc, cfg)

        # 情况 2：分布载荷区间格式
        #   [[x0, x1], y_const]
        #   [x_const, [y0, y1]]
        elif (
            isinstance(region_desc, (list, tuple))
            and len(region_desc) == 2
        ):
            loc_fn = make_segment_load_selector(region_desc, cfg)

        else:
            raise ValueError(f"Unsupported load format: {load}")

        def traction_fn(point, fx=fx, fy=fy):
            return jnp.array([fx, fy])

        location_fns.append(loc_fn)
        traction_fns.append(traction_fn)

    return location_fns, traction_fns
# ==============================================================================
# 1. 定义物理模型 (严格遵循教程规范)
# ==============================================================================
GLOBAL_TRACTION_FNS = []

class CantileverSIMP(Problem):
    def get_tensor_map(self):
        def stress(u_grad, theta):
            E_0, E_min, p = 1.0, 1e-3, 3.0
            E = E_min + (theta[0] ** p) * (E_0 - E_min)
            nu = 0.3

            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / (1.0 - nu**2)

            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * jnp.trace(epsilon) * jnp.eye(self.dim) + 2 * mu * epsilon
            return sigma
        return stress

    def get_surface_maps(self):
        surface_maps = []

        for traction_fn in GLOBAL_TRACTION_FNS:
            def neumann_bc(u, x, traction_fn=traction_fn):
                return traction_fn(x)
            surface_maps.append(neumann_bc)

        return surface_maps

    def compute_compliance(self, sol):
        total_compliance = 0.0
        boundary_inds_list = self.fes[0].get_boundary_conditions_inds(self.location_fns)

        for load_idx, traction_fn in enumerate(GLOBAL_TRACTION_FNS):
            boundary_inds = boundary_inds_list[load_idx]
            if boundary_inds.shape[0] == 0:
                continue

            _, nanson_scale = self.fes[0].get_face_shape_grads(boundary_inds)

            selected_cell_sols = sol[self.fes[0].cells][boundary_inds[:, 0]]
            selected_face_shape_vals = self.fes[0].face_shape_vals[boundary_inds[:, 1]]

            u_face = jnp.sum(
                selected_cell_sols[:, None, :, :] * selected_face_shape_vals[:, :, :, None],
                axis=2
            )

            physical_surface_quad_points = self.fes[0].get_physical_surface_quad_points(boundary_inds)
            traction = jax.vmap(jax.vmap(traction_fn))(physical_surface_quad_points)

            total_compliance = total_compliance + jnp.sum(
                traction * u_face * nanson_scale[:, :, None]
            )

        return total_compliance

    def set_params(self, params):
        theta = params[0]
        self.internal_vars = [theta]

# ==============================================================================
# 2. 网格与边界条件初始化 (修改为完整 MBB 梁)
# ==============================================================================
# ==============================================================================
# 2. problem 配置驱动的网格与边界条件初始化
# ==============================================================================
problem_yaml_path = os.environ.get(
    "FMTO_SIMP_CONFIG",
    str(PROJECT_ROOT / "configs" / "simp" / "Canti_beam_distrl.yaml"),
)
cfg = load_problem_config(problem_yaml_path)

print(f"Generating 2D mesh for problem: {cfg.name}")

Lx, Ly = cfg.Lx, cfg.Ly
Nx, Ny = cfg.Nx, cfg.Ny

# =========================
# 🔥 新增：设计域 mask
# =========================
mask_np = build_l_shape_mask(Nx, Ny, Lx, Ly)
mask = jnp.array(mask_np)

data_dir = os.path.join(os.getcwd(), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)

meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
points_2d = meshio_mesh.points[:, :2]
cells_2d = meshio_mesh.cells_dict['quad']

mesh = Mesh(points_2d, cells_2d)
mesh.cell_type = 'quad'
num_cells = mesh.cells.shape[0]

dirichlet_bc_info = build_dirichlet_bc_info(cfg)
location_fns, traction_fns = build_loads(cfg)

GLOBAL_TRACTION_FNS = traction_fns

problem = CantileverSIMP(
    mesh,
    vec=2,
    dim=2,
    ele_type='QUAD4',
    dirichlet_bc_info=dirichlet_bc_info,
    location_fns=location_fns,
)
print("num dirichlet funcs:", len(dirichlet_bc_info[0]))
print("num load location_fns:", len(location_fns))
print("num traction_fns:", len(traction_fns))


# ==============================================================================
# 3. 定义优化目标 (Loss Function) 与自动微分
# ==============================================================================

num_cells = mesh.cells.shape[0]
vol_frac = cfg.vol_frac
theta_init = np.ones(num_cells) * vol_frac

theta_init_2d = theta_init.reshape(Ny, Nx)
# 设计域为 vol_frac，非设计域为 0.001（与 MMA 的 min bound 保持绝对一致）
theta_init_2d = jnp.where(mask_np > 0.5, theta_init_2d, 0.001)
theta_init = np.array(theta_init_2d).flatten()

fwd_pred = ad_wrapper(problem, solver_options={'petsc_solver': {}})

def compute_objective(theta_flat):
    theta_2d = jnp.reshape(theta_flat, (Ny, Nx))

    # =========================
    # 🔥 1. mask 原始变量
    # =========================
    theta_2d = theta_2d * mask

    # =========================
    # 🔥 2. filter
    # =========================
    filtered_theta_2d = apply_density_filter(theta_2d, mask, sigma=cfg.filter_sigma)

    # =========================
    # 🔥 3. 再 mask（防止扩散）
    # =========================
    filtered_theta_2d = filtered_theta_2d * mask

    filtered_theta_flat = jnp.reshape(filtered_theta_2d, (-1,))

    num_quads = problem.fes[0].num_quads
    theta_in = jnp.reshape(filtered_theta_flat, (num_cells, 1, 1))
    theta_in = jnp.repeat(theta_in, num_quads, axis=1)

    sol_list = fwd_pred([theta_in])
    sol = sol_list[0]

    compliance = problem.compute_compliance(sol)
    return jnp.abs(compliance), (sol, filtered_theta_flat)
print("Building JAX AutoDiff graph...")
obj_and_grad = jax.value_and_grad(compute_objective, has_aux=True)
def compute_volume(theta_flat):
    theta_2d = jnp.reshape(theta_flat, (Ny, Nx))

    # mask
    theta_2d = theta_2d * mask

    filtered_theta_2d = apply_density_filter(theta_2d, mask, sigma=cfg.filter_sigma)

    # 再 mask
    filtered_theta_2d = filtered_theta_2d * mask

    # ⚠️ 只统计设计域
    vol = jnp.sum(filtered_theta_2d) / jnp.sum(mask)

    return vol - vol_frac

vol_and_grad = jax.value_and_grad(compute_volume)

# ==============================================================================
# 4. 执行约束优化 (使用纯血 MMA 引擎)
# ==============================================================================
print("Starting Topology Optimization with MMA (No Filter)...")
start_time = time.time()

n = num_cells  # 变量总数 (8000)
m = 1          # 约束总数 (只有1个体积约束)

# 初始化 MMA 引擎
mma = MMA()
mma.setNumConstraints(m)
mma.setNumDesignVariables(n)
# =========================
# 🔥 修改：锁定非设计区域（Passive Elements）
# =========================
mma.setMinandMaxBoundsForDesignVariables(0.001 * np.ones((n, 1)), np.ones((n, 1)))
mma.setLowerAndUpperAsymptotes(np.ones((n, 1)), np.ones((n, 1)))
mma.setScalingParams(1.0, np.zeros((m, 1)), 10000 * np.ones((m, 1)), np.zeros((m, 1)))
mma.setMoveLimit(0.2)

# MMA 历史状态初始化
xval = theta_init.reshape(-1, 1)
xold1 = xval.copy()
xold2 = xval.copy()
mma.registerMMAIter(xval, xold1, xold2)

max_iters = 35

theta = theta_init.copy()
switch_cfg = SwitchConfig(
    min_iter=30,
    window=4,
    improve_thresh=0.01,
    binary_thresh=0.6,
    gap_thresh=0.11,
    patience=3,
    verbose=True,
)
switch_monitor = SwitchMonitor(switch_cfg)

switch_iter = None
switch_snapshot = None

yaml_dir = str(PROJECT_ROOT / "configs" / "fmto")
template_yaml_path = os.path.join(yaml_dir, f"{cfg.name}.yaml")
generated_yaml_dir = PROJECT_ROOT / "artifacts" / "generated_configs"
generated_yaml_dir.mkdir(parents=True, exist_ok=True)
generated_yaml_path = str(generated_yaml_dir / f"{cfg.name}.yaml")

density2feature_out_dir = os.path.join(data_dir, f"density2feature_switch_{cfg.name}")
temp_pt_path = os.path.join(density2feature_out_dir, f"{cfg.name}_switch_input.pt")
for i in range(max_iters):
    iter_start = time.time()

    # 1. 计算目标函数(柔度)及其梯度
    (obj, (sol, filtered_theta)), grad = obj_and_grad(theta)

    # 2. 计算约束(体积)及其梯度
    vol_val, vol_grad = vol_and_grad(theta)

    # =========================
    # Adaptive switch monitor
    # =========================
    should_switch, switch_stats = switch_monitor.update(
        iter_idx=i,
        obj=obj,
        theta_flat=theta,
        filtered_theta_flat=filtered_theta,
    )

    # 3. 保存结果
    vtu_path = os.path.join(data_dir, f'vtk/sol_{i:03d}.vtu')

    sol_3d = np.pad(np.array(sol), ((0, 0), (0, 1)), mode='constant')

    out_mesh = meshio.Mesh(
        points=np.array(meshio_mesh.points),
        cells=[("quad", np.array(mesh.cells))],
        point_data={"displacement": sol_3d},
        cell_data={"density": [np.array(filtered_theta)]}
    )
    out_mesh.write(vtu_path)

    iter_time = time.time() - iter_start
    print(f"Iter {i:03d} | Compliance: {obj:.6f} | Vol Error: {vol_val:.4f} | Time: {iter_time:.2f}s")

    if switch_cfg.verbose:
        mi = switch_stats["mean_improve"]
        mb = switch_stats["mean_binary"]
        mg = switch_stats["mean_gap"]

        mi_str = f"{mi:.5f}" if mi is not None else "None"
        mb_str = f"{mb:.5f}" if mb is not None else "None"
        mg_str = f"{mg:.5f}" if mg is not None else "None"

        print(
            f"  [SwitchMonitor] "
            f"mean_improve={mi_str} | "
            f"mean_binary={mb_str} | "
            f"mean_gap={mg_str} | "
            f"hits={switch_stats['hit_count']} | "
            f"conds=({int(switch_stats['cond_progress'])},"
            f"{int(switch_stats['cond_binary'])},"
            f"{int(switch_stats['cond_gap'])})"
        )

    if should_switch:
        from bridge.density2feature_mask import FitConfig, export_density2feature_and_yaml_from_pt

        d2f_cfg = FitConfig(
            nelx=Nx,
            nely=Ny,
            Lx=cfg.d2f_Lx,
            Ly=cfg.d2f_Ly,
            threshold=0.40,
            opening_iters=1,
            closing_iters=1,
            min_material_area_pixels=8,
            min_void_area_pixels=5,
            maxiter=250,
            max_holes=8,
            near_material_iters=3,
            local_bbox_expand=8,
            seed_min_area_pixels=3,
            candidate_opening_iters=0,
            candidate_closing_iters=1,
            duplicate_iou_threshold=0.35,
        )

        switch_iter = i
        switch_snapshot = {
            "theta": np.array(theta).copy(),
            "filtered_theta": np.array(filtered_theta).copy(),
            "sol": np.array(sol).copy(),
            "obj": float(np.asarray(obj)),
            "vol_val": float(np.asarray(vol_val)),
            "switch_stats": switch_stats,
        }

        print(f"\n>>> Adaptive switch triggered at iter {i:03d}")
        print(
            f">>> reason: plateau={switch_stats['cond_progress']}, "
            f"binary={switch_stats['cond_binary']}, gap={switch_stats['cond_gap']}"
        )

        export_filtered_theta = switch_snapshot["filtered_theta"]
        export_raw_theta = switch_snapshot["theta"]

        clean_image_filtered = build_density2feature_input_image(export_filtered_theta, Nx, Ny)
        clean_image_prior = build_density2feature_input_image(export_raw_theta, Nx, Ny)

        save_density2feature_pt(
            clean_image_filtered=clean_image_filtered,
            clean_image_prior=clean_image_prior,
            mask_np=mask_np,
            pt_path=temp_pt_path,
            Nx=Nx,
            Ny=Ny,
        )
        
        print(f">>> temp pt saved: {temp_pt_path}")
        print(f">>> temp pt exists: {os.path.exists(temp_pt_path)}")

        d2f_result = export_density2feature_and_yaml_from_pt(
            pt_path=temp_pt_path,
            out_dir=density2feature_out_dir,
            cfg=d2f_cfg,
            template_yaml_path=template_yaml_path,
            output_yaml_path=generated_yaml_path,
        )

        print(">>> density2feature finished.")
        print(f">>> num_primitives: {d2f_result['num_primitives']}")
        print(f">>> mean_iou: {d2f_result['mean_iou']:.6f}")
        print(f">>> generated yaml: {d2f_result['output_yaml_path']}")
        print(f">>> temp pt: {temp_pt_path}")
        break
    
    # 4. 整理形状，喂给 MMA 算法
    J_mma = np.array(obj)
    # 转换为 numpy 数组以便修改
    grad_np = np.array(grad)
    vol_grad_np = np.array(vol_grad)

    # 🔥 核心修复：物理阻断！把 Mask 区域的灵敏度彻底清零
    passive_indices = np.where(mask_np.flatten() == 0)[0]
    grad_np[passive_indices] = 0.0
    vol_grad_np[passive_indices] = 0.0
    dJ_mma = np.array(grad).reshape(-1, 1)
    vc_mma = np.array([vol_val])[:, None]
    dvc_mma = np.array(vol_grad).reshape(1, -1)

    mma.setObjectiveWithGradient(J_mma, dJ_mma)
    mma.setConstraintWithGradient(vc_mma, dvc_mma)

    mma.mmasub(xval)
    xmma, _, _ = mma.getOptimalValues()

    xold2 = xold1.copy()
    xold1 = xval.copy()
    xval = xmma.copy()
    mma.registerMMAIter(xval, xold1, xold2)

    theta = xval.flatten()

    

if switch_snapshot is None:
    from bridge.density2feature_mask import FitConfig, export_density2feature_and_yaml_from_pt

    print("Adaptive switch not triggered; using final iteration result for export.")

    d2f_cfg = FitConfig(
        nelx=Nx,
        nely=Ny,
        Lx=cfg.d2f_Lx,
        Ly=cfg.d2f_Ly,
        threshold=0.15,
        opening_iters=0,
        closing_iters=1,
        min_material_area_pixels=8,
        min_void_area_pixels=5,
        maxiter=250,
        max_holes=8,
        near_material_iters=3,
        local_bbox_expand=8,
        seed_min_area_pixels=3,
        candidate_opening_iters=0,
        candidate_closing_iters=1,
        duplicate_iou_threshold=0.35,
    )

    export_filtered_theta = np.array(filtered_theta)
    export_raw_theta = np.array(theta)

    clean_image_filtered = build_density2feature_input_image(export_filtered_theta, Nx, Ny)
    clean_image_prior = build_density2feature_input_image(export_raw_theta, Nx, Ny)

    save_density2feature_pt(
        clean_image_filtered=clean_image_filtered,
        clean_image_prior=clean_image_prior,
        mask_np=mask_np,
        pt_path=temp_pt_path,
        Nx=Nx,
        Ny=Ny,
    )

    d2f_result = export_density2feature_and_yaml_from_pt(
        pt_path=temp_pt_path,
        out_dir=density2feature_out_dir,
        cfg=d2f_cfg,
        template_yaml_path=template_yaml_path,
        output_yaml_path=generated_yaml_path,
    )

    print(">>> density2feature finished from final iteration.")
    print(f">>> num_primitives: {d2f_result['num_primitives']}")
    print(f">>> mean_iou: {d2f_result['mean_iou']:.6f}")
    print(f">>> updated yaml: {d2f_result['output_yaml_path']}")
    print(f">>> temp pt: {temp_pt_path}")

print(f"\nOptimization Finished in {time.time() - start_time:.2f} seconds!")

# ==============================================================================
# 5. 跨界握手：清洗物理场并导出为 DETR 标准格式
# ==============================================================================
import torch

print("\n💾 正在对 JAX-FEM 物理场进行 [信号清洗] 与 [降维打包]...")

# 1. 提取最后一次迭代的过滤密度场
if switch_snapshot is not None:
    export_filtered_theta = switch_snapshot["filtered_theta"]
else:
    export_filtered_theta = np.array(filtered_theta)

export_filtered_theta = export_filtered_theta * mask_np.flatten()
final_density_jax = jnp.reshape(export_filtered_theta, (Nx, Ny))
image_2d = np.array(final_density_jax).T

# ---------------------------------------------------------
# 🔥 【核心新增】：Sim-to-Real 信号清洗滤波器
# ---------------------------------------------------------
# A. 拔除 E_min 底噪，强制归一化到绝对的 [0, 1]
min_val = image_2d.min()
max_val = image_2d.max()
clean_image = (image_2d - min_val) / (max_val - min_val + 1e-8)

# B. 对比度截断拉伸 (去除漫反射灰度，增强主干)
# 低于 0.15 的直接归零，高于 0.8 的直接拉满 1.0，中间部分线性映射
lower_bound = 0.0
upper_bound = 0.9
clean_image = np.clip((clean_image - lower_bound) / (upper_bound - lower_bound), 0.0, 1.0)
# ---------------------------------------------------------

# 2. 升维打包成 DETR 认识的 [Channel, Height, Width]
image_tensor = torch.from_numpy(clean_image).float().unsqueeze(0)

# 3. 组装测试大礼包
jax_export_data = {
    "image": image_tensor,       
    "nelx": Nx,                  
    "nely": Ny,                  
    "is_real_test": True         
}

# 4. 独立落盘保存
export_dir = str(PROJECT_ROOT / "artifacts" / "simp_exports")
os.makedirs(export_dir, exist_ok=True)
export_path = os.path.join(export_dir, "jax_mbb_beam.pt")

torch.save(jax_export_data, export_path)

print(f"✅ 跨界数据包已生成！(已完成底噪剔除与灰度锐化)")
print(f"📂 文件位置: {os.path.abspath(export_path)}")
