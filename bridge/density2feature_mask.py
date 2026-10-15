import argparse
import math
import os
from pathlib import Path as FilePath
from dataclasses import dataclass
from typing import Dict, List, Tuple
import copy
import yaml
import numpy as np
import torch
from matplotlib.path import Path
from scipy import ndimage
from scipy.optimize import minimize

PROJECT_ROOT = FilePath(__file__).resolve().parents[1]


@dataclass
class FitConfig:
    """整体配置。"""
    nelx: int = 90
    nely: int = 30
    Lx: float = 0.6
    Ly: float = 0.2
    threshold: float = 0.50
    opening_iters: int = 1
    closing_iters: int = 1
    min_material_area_pixels: int = 8
    min_void_area_pixels: int = 5
    maxiter: int = 250
    max_holes: int = 8
    verbose: bool = True
    # design-box based void extraction
    near_material_iters: int = 3
    local_bbox_expand: int = 8
    seed_min_area_pixels: int = 3
    candidate_opening_iters: int = 0
    candidate_closing_iters: int = 1
    duplicate_iou_threshold: float = 0.35
    boundary_outer_expand: int = 1      # 朝 design-box 外边界方向最多扩多少
    boundary_inner_bonus: int = 4       # 朝材料内部方向额外多扩多少
    near_material_score_weight: float = 0.45
    outward_penalty_weight: float = 0.60

    # ===== 新增：拟合阶段的越界惩罚 =====
    vertex_outside_weight: float = 120.0
    reg_weight: float = 0.10

# ============================================================
# 1. 数据读取
# ============================================================
def load_simp_image(pt_path: str) -> Tuple[np.ndarray, Dict]:
    """从 .pt 文件中读取 image。"""
    data = torch.load(pt_path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("读取的 .pt 内容不是 dict。")
    if "image" not in data:
        raise KeyError(".pt 文件中没有找到键 'image'。")

    image = data["image"]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)

    if image.ndim == 3:
        if image.shape[0] != 1:
            raise ValueError(f"image 形状为 {image.shape}，仅支持 [1,H,W] 或 [H,W]。")
        image = image[0]
    elif image.ndim != 2:
        raise ValueError(f"image 形状为 {image.shape}，仅支持 2D 图像。")

    image = image.astype(np.float64)
    image = np.clip(image, 0.0, 1.0)
    return image, dict(data)

def load_simp_images(pt_path: str):
    data = torch.load(pt_path, map_location="cpu")

    image = data["image"]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim == 3:
        image = image[0]
    image = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)

    if "image_prior" in data:
        image_prior = data["image_prior"]
        if isinstance(image_prior, torch.Tensor):
            image_prior = image_prior.detach().cpu().numpy()
        if image_prior.ndim == 3:
            image_prior = image_prior[0]
        image_prior = np.clip(np.asarray(image_prior, dtype=np.float64), 0.0, 1.0)
    else:
        # 兼容旧pt：没有image_prior时就退化为与image相同
        image_prior = image.copy()

    # 🔥 新增提取 design_mask
    design_mask = None
    if "design_mask" in data:
        dm = data["design_mask"]
        if isinstance(dm, torch.Tensor):
            dm = dm.detach().cpu().numpy()
        if dm.ndim == 3:
            dm = dm[0]
        design_mask = np.asarray(dm, dtype=bool)

    meta = dict(data) 
    # 🔥 返回值增加 design_mask
    return image, image_prior, design_mask, meta

# ============================================================
# 2. 图像处理
# ============================================================
def keep_largest_component(mask: np.ndarray, min_area_pixels: int) -> np.ndarray:
    """保留面积最大的连通域。"""
    labeled, num = ndimage.label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)

    ids = np.arange(1, num + 1)
    sizes = np.asarray(ndimage.sum(mask, labeled, index=ids), dtype=np.float64)
    valid = ids[sizes >= min_area_pixels]
    if len(valid) == 0:
        return np.zeros_like(mask, dtype=bool)

    valid_sizes = sizes[valid - 1]
    largest_id = valid[np.argmax(valid_sizes)]
    return labeled == largest_id


def preprocess_material_mask(image: np.ndarray, cfg: FitConfig) -> np.ndarray:
    """将 SIMP 灰度图转成主材料区域 mask。"""
    material = image >= cfg.threshold
    if cfg.opening_iters > 0:
        material = ndimage.binary_opening(material, iterations=cfg.opening_iters)
    if cfg.closing_iters > 0:
        material = ndimage.binary_closing(material, iterations=cfg.closing_iters)
    material = keep_largest_component(material, min_area_pixels=cfg.min_material_area_pixels)
    return material.astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float((inter + 1e-6) / (union + 1e-6))


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return 0, 0, -1, -1
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def expand_bbox(r0: int, r1: int, c0: int, c1: int, shape: Tuple[int, int], margin: int) -> Tuple[int, int, int, int]:
    H, W = shape
    return max(0, r0 - margin), min(H - 1, r1 + margin), max(0, c0 - margin), min(W - 1, c1 + margin)


def polygon_mask_grid(points_cr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    points_cr = np.asarray(points_cr, dtype=np.float64)
    cols = np.clip(points_cr[:, 0], 0, shape[1] - 1)
    rows = np.clip(points_cr[:, 1], 0, shape[0] - 1)
    rr_cc = np.stack([rows, cols], axis=1)
    path = Path(np.stack([cols, rows], axis=1))
    ccg, rrg = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    query = np.stack([ccg.ravel(), rrg.ravel()], axis=1)
    inside = path.contains_points(query, radius=0.25)
    return inside.reshape(shape)


def touches_domain_boundary(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def build_design_box_mask(shape: Tuple[int, int]) -> np.ndarray:
    return np.ones(shape, dtype=bool)

def boundary_touch_flags(mask: np.ndarray) -> Dict[str, bool]:
    return {
        "top": bool(mask[0, :].any()),
        "bottom": bool(mask[-1, :].any()),
        "left": bool(mask[:, 0].any()),
        "right": bool(mask[:, -1].any()),
    }

def refine_candidate_from_seed(
    seed_mask: np.ndarray,
    full_void_mask: np.ndarray,
    material_mask: np.ndarray,
    cfg: FitConfig
) -> np.ndarray:
    """
    对边界孔做“向材料内部偏置”的 candidate 恢复。
    核心思想：
      - 仍然以 design_box - material 作为母区域
      - 但如果 seed 接触了 design-box 边界，则 bbox 不对称扩张：
          * 朝外边界方向少扩
          * 朝材料内部方向多扩
      - 在候选连通域打分时：
          * 奖励与 seed overlap
          * 奖励靠近材料边界
          * 惩罚落在 design-box 外侧边缘带中的比例
    """
    r0, r1, c0, c1 = bbox_from_mask(seed_mask)
    if r1 < r0:
        return np.zeros_like(seed_mask, dtype=bool)

    H, W = seed_mask.shape
    flags = boundary_touch_flags(seed_mask)

    # 先给一个基础扩张
    top_margin = cfg.local_bbox_expand
    bottom_margin = cfg.local_bbox_expand
    left_margin = cfg.local_bbox_expand
    right_margin = cfg.local_bbox_expand

    # ===== 关键：边界孔不对称扩张，向内多扩、向外少扩 =====
    if flags["left"]:
        left_margin = cfg.boundary_outer_expand
        right_margin = cfg.local_bbox_expand + cfg.boundary_inner_bonus
    if flags["right"]:
        right_margin = cfg.boundary_outer_expand
        left_margin = cfg.local_bbox_expand + cfg.boundary_inner_bonus
    if flags["top"]:
        top_margin = cfg.boundary_outer_expand
        bottom_margin = cfg.local_bbox_expand + cfg.boundary_inner_bonus
    if flags["bottom"]:
        bottom_margin = cfg.boundary_outer_expand
        top_margin = cfg.local_bbox_expand + cfg.boundary_inner_bonus

    r0e = max(0, r0 - top_margin)
    r1e = min(H - 1, r1 + bottom_margin)
    c0e = max(0, c0 - left_margin)
    c1e = min(W - 1, c1 + right_margin)

    seed_local = seed_mask[r0e:r1e + 1, c0e:c1e + 1]
    void_local = full_void_mask[r0e:r1e + 1, c0e:c1e + 1]
    material_local = material_mask[r0e:r1e + 1, c0e:c1e + 1]

    labeled, num = ndimage.label(void_local)
    if num == 0:
        return np.zeros_like(seed_mask, dtype=bool)

    # 靠近材料边界的局部带状区：奖励“朝材料内部”的 candidate
    near_material_local = ndimage.binary_dilation(material_local, iterations=max(1, cfg.near_material_iters))

    best_comp = None
    best_score = -1e18

    for idx in range(1, num + 1):
        comp = (labeled == idx)
        overlap = float(np.logical_and(comp, seed_local).sum())
        if overlap <= 0:
            continue

        area = float(comp.sum())
        if area < 1.0:
            continue

        # 1) 基础：和当前 seed 的重叠越大越好
        score = overlap / math.sqrt(area + 1e-6)

        # 2) 奖励：candidate 更贴近材料边界（说明更像局部孔，而不是远处背景）
        near_ratio = float(np.logical_and(comp, near_material_local).sum()) / (area + 1e-6)
        score += cfg.near_material_score_weight * near_ratio

        # 3) 惩罚：如果 seed 接触边界，candidate 在“外侧边缘带”占比太大，说明它被往 design box 外拉了
        outward_ratio = 0.0
        if flags["left"]:
            outward_ratio += float(comp[:, :max(1, cfg.boundary_outer_expand)].sum()) / (area + 1e-6)
        if flags["right"]:
            outward_ratio += float(comp[:, -max(1, cfg.boundary_outer_expand):].sum()) / (area + 1e-6)
        if flags["top"]:
            outward_ratio += float(comp[:max(1, cfg.boundary_outer_expand), :].sum()) / (area + 1e-6)
        if flags["bottom"]:
            outward_ratio += float(comp[-max(1, cfg.boundary_outer_expand):, :].sum()) / (area + 1e-6)

        score -= cfg.outward_penalty_weight * outward_ratio

        if score > best_score:
            best_score = score
            best_comp = comp

    if best_comp is None:
        out = np.zeros_like(seed_mask, dtype=bool)
        out[r0e:r1e + 1, c0e:c1e + 1] = seed_local
        return out

    cand_local = best_comp.copy()
    if cfg.candidate_opening_iters > 0:
        cand_local = ndimage.binary_opening(cand_local, iterations=cfg.candidate_opening_iters)
    if cfg.candidate_closing_iters > 0:
        cand_local = ndimage.binary_closing(cand_local, iterations=cfg.candidate_closing_iters)

    out = np.zeros_like(seed_mask, dtype=bool)
    out[r0e:r1e + 1, c0e:c1e + 1] = cand_local
    return out


def extract_void_primitives(image: np.ndarray, material_mask: np.ndarray, cfg: FitConfig, design_mask: np.ndarray = None) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    按设计域 design box 提取 void primitives。

    关键改动：
      - 母区域不再是 hull(material)，而是整个 design box。
      - primitive 定义为 design_box 内的非材料区域中的几何单元。
      - 使用 "靠近材料边界的 void seed" 来避免把整个外部背景都当成 primitive。

    输出：
      masks          : 每个 primitive 候选 mask 列表
      void_union     : 所有 primitive 候选并集
      full_void_mask : design_box - material_mask
      seed_union     : 靠近材料边界的 seed 并集（便于调试）
    """
    if design_mask is not None:
        design_box = design_mask.astype(bool)
    else:
        design_box = build_design_box_mask(material_mask.shape)
    full_void_mask = np.logical_and(design_box, np.logical_not(material_mask))

    # 只取靠近材料边界的非材料区域作为 seed。
    # 这一步会自然包含：
    #   - 内部孔洞附近的 void
    #   - 贴设计域边界的开口孔
    # 同时尽量不把远离材料的大背景整块拿进来。
    near_material_band = ndimage.binary_dilation(material_mask, iterations=cfg.near_material_iters)
    seed_mask = np.logical_and(full_void_mask, near_material_band)

    if cfg.candidate_opening_iters > 0:
        seed_mask = ndimage.binary_opening(seed_mask, iterations=cfg.candidate_opening_iters)
    if cfg.candidate_closing_iters > 0:
        seed_mask = ndimage.binary_closing(seed_mask, iterations=cfg.candidate_closing_iters)

    labeled, num = ndimage.label(seed_mask)
    masks: List[np.ndarray] = []
    seed_union = np.zeros_like(material_mask, dtype=bool)
    void_union = np.zeros_like(material_mask, dtype=bool)

    for idx in range(1, num + 1):
        seed_comp = labeled == idx
        area_seed = int(seed_comp.sum())
        if area_seed < cfg.seed_min_area_pixels:
            continue

        seed_union = np.logical_or(seed_union, seed_comp)
        cand = refine_candidate_from_seed(seed_comp, full_void_mask, material_mask, cfg)
        area_cand = int(cand.sum())
        if area_cand < cfg.min_void_area_pixels:
            continue

        duplicated = False
        for m in masks:
            if mask_iou(cand, m) > cfg.duplicate_iou_threshold:
                duplicated = True
                break
        if duplicated:
            continue

        masks.append(cand)
        void_union = np.logical_or(void_union, cand)

    return masks, void_union, full_void_mask, seed_union


# ============================================================
# 3. 坐标变换：网格 <-> 物理坐标
# ============================================================
def grid_to_phys_xy(col: np.ndarray, row: np.ndarray, cfg: FitConfig) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(col, dtype=np.float64) / (cfg.nelx - 1) * (2.0 * cfg.Lx) - cfg.Lx
    y = np.asarray(row, dtype=np.float64) / (cfg.nely - 1) * (2.0 * cfg.Ly) - cfg.Ly
    return x, y


def phys_xy_to_grid(x: np.ndarray, y: np.ndarray, cfg: FitConfig) -> Tuple[np.ndarray, np.ndarray]:
    col = (np.asarray(x, dtype=np.float64) + cfg.Lx) / (2.0 * cfg.Lx) * (cfg.nelx - 1)
    row = (np.asarray(y, dtype=np.float64) + cfg.Ly) / (2.0 * cfg.Ly) * (cfg.nely - 1)
    return col, row


# ============================================================
# 4. 几何参数化与渲染
# ============================================================
def params8_to_polygon(params8: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2, offsetx, offsety, rotation, scale = params8
    pts = np.array([
        [-0.5, -0.5],
        [0.5, -0.5],
        [x1, y1],
        [x2, y2],
    ], dtype=np.float64)
    pts = pts * scale
    c, s = math.cos(rotation), math.sin(rotation)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    pts = pts @ R.T
    pts[:, 0] += offsetx
    pts[:, 1] += offsety
    return pts


def polygon_to_mask(pts_phys: np.ndarray, cfg: FitConfig) -> np.ndarray:
    cols = np.arange(cfg.nelx)
    rows = np.arange(cfg.nely)
    cc, rr = np.meshgrid(cols, rows)
    xx, yy = grid_to_phys_xy(cc, rr, cfg)
    query_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    path = Path(pts_phys)
    inside = path.contains_points(query_points, radius=1e-12)
    return inside.reshape(cfg.nely, cfg.nelx)


def render_from_params8(params8: np.ndarray, cfg: FitConfig) -> np.ndarray:
    return polygon_to_mask(params8_to_polygon(params8), cfg)


def params8_to_geom12(params8: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2, offsetx, offsety, rotation, scale = params8
    return np.array([x1, y1, x2, y2, 0.0, 0.0, 0.0, 0.0, offsetx, offsety, rotation, scale], dtype=np.float64)

def normalize_rotation_half_pi(theta: float) -> float:
    """
    将角度规范到 [-pi/2, pi/2]。
    因为四边形 primitive 的朝向存在 180° 等价性，
    所以可以把 theta 折叠到这个区间。
    """
    theta = float(theta)
    while theta > math.pi / 2:
        theta -= math.pi
    while theta < -math.pi / 2:
        theta += math.pi
    return theta
# ============================================================
# 5. 单个 primitive 的初值估计
# ============================================================
def estimate_rotation_pca(mask: np.ndarray, cfg: FitConfig) -> float:
    rows, cols = np.nonzero(mask)
    if len(rows) < 2:
        return 0.0
    x, y = grid_to_phys_xy(cols, rows, cfg)
    pts = np.stack([x, y], axis=1)
    pts_centered = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    theta = math.atan2(major[1], major[0])
    return normalize_rotation_half_pi(theta)

def _safe_normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def _project_void_points_to_major_axis(void_mask: np.ndarray, cfg: FitConfig):
    rows, cols = np.nonzero(void_mask)
    if len(rows) < 3:
        return None

    x, y = grid_to_phys_xy(cols, rows, cfg)
    pts = np.stack([x, y], axis=1)
    center = pts.mean(axis=0)

    pts_centered = pts - center[None, :]
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = eigvecs[:, 0]
    major = _safe_normalize_vec(major)

    proj = pts_centered @ major
    return {
        "pts": pts,
        "center": center,
        "proj": proj,
        "major": major,
        "eigvals": eigvals,
    }

def _compute_bimodal_score_from_projection(proj: np.ndarray, num_bins: int = 21) -> float:
    if proj is None or len(proj) < 6:
        return 0.0

    hist, bin_edges = np.histogram(proj, bins=num_bins)
    hist = hist.astype(np.float64)

    if hist.sum() <= 0:
        return 0.0

    # 轻微平滑
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    hist_s = np.convolve(hist, kernel, mode="same")

    # 寻找局部峰
    peak_ids = []
    for i in range(1, len(hist_s) - 1):
        if hist_s[i] >= hist_s[i - 1] and hist_s[i] >= hist_s[i + 1]:
            peak_ids.append(i)

    if len(peak_ids) < 2:
        return 0.0

    # 取两个最高峰
    peak_ids = sorted(peak_ids, key=lambda i: hist_s[i], reverse=True)[:2]
    peak_ids = sorted(peak_ids)

    left_peak = hist_s[peak_ids[0]]
    right_peak = hist_s[peak_ids[1]]
    valley = hist_s[peak_ids[0]:peak_ids[1] + 1].min()

    denom = max(1e-8, 0.5 * (left_peak + right_peak))
    score = 1.0 - valley / denom
    return float(np.clip(score, 0.0, 1.0))

def _compute_weighted_profile_along_major_axis(
    image: np.ndarray,
    void_mask: np.ndarray,
    major: np.ndarray,
    cfg: FitConfig,
    num_bins: int = 25,
):
    r0, r1, c0, c1 = bbox_from_mask(void_mask)
    if r1 < r0:
        return None

    margin = max(2, cfg.local_bbox_expand // 2)
    r0, r1, c0, c1 = expand_bbox(r0, r1, c0, c1, image.shape, margin)

    patch = image[r0:r1 + 1, c0:c1 + 1]
    H, W = patch.shape

    rr, cc = np.meshgrid(
        np.arange(r0, r1 + 1),
        np.arange(c0, c1 + 1),
        indexing="ij",
    )
    x, y = grid_to_phys_xy(cc, rr, cfg)
    pts = np.stack([x.ravel(), y.ravel()], axis=1)

    rows_v, cols_v = np.nonzero(void_mask)
    if len(rows_v) < 3:
        return None

    xv, yv = grid_to_phys_xy(cols_v, rows_v, cfg)
    center = np.stack([xv, yv], axis=1).mean(axis=0)

    pts_centered = pts - center[None, :]
    proj = pts_centered @ major

    # void-like 权重：越接近空洞（低密度）越大
    weights = (1.0 - patch).ravel()

    hist_w, bin_edges = np.histogram(proj, bins=num_bins, weights=weights)
    hist_c, _ = np.histogram(proj, bins=num_bins)

    valid = hist_c > 0
    profile = np.zeros_like(hist_w, dtype=np.float64)
    profile[valid] = hist_w[valid] / hist_c[valid]

    # 平滑
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    profile = np.convolve(profile, kernel, mode="same")

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, profile


def _compute_bimodal_score_from_image_profile(
    image: np.ndarray,
    void_mask: np.ndarray,
    major: np.ndarray,
    cfg: FitConfig,
) -> Tuple[float, float]:
    out = _compute_weighted_profile_along_major_axis(image, void_mask, major, cfg, num_bins=25)
    if out is None:
        return 0.0, 0.0

    centers, profile = out
    if len(profile) < 5 or np.max(profile) < 1e-8:
        return 0.0, 0.0

    peak_ids = []
    for i in range(1, len(profile) - 1):
        if profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]:
            peak_ids.append(i)

    if len(peak_ids) < 2:
        return 0.0, 0.0

    # 取两个最高峰
    peak_ids = sorted(peak_ids, key=lambda i: profile[i], reverse=True)[:2]
    peak_ids = sorted(peak_ids)

    p1, p2 = peak_ids
    left_peak = profile[p1]
    right_peak = profile[p2]
    valley = profile[p1:p2 + 1].min()

    peak_mean = max(1e-8, 0.5 * (left_peak + right_peak))
    contrast_score = 1.0 - valley / peak_mean
    contrast_score = float(np.clip(contrast_score, 0.0, 1.0))

    peak_dist = abs(centers[p2] - centers[p1])
    total_span = max(1e-8, centers[-1] - centers[0])
    # 更宽松的距离分数
    dist_score = np.clip(peak_dist / (0.20 * total_span), 0.0, 1.0)

    # 用加法，不用乘法，避免弱双峰被压到接近0
    score = 0.7 * contrast_score + 0.3 * dist_score
    return float(np.clip(score, 0.0, 1.0)), float(peak_dist)


def _compute_gray_score(image: np.ndarray, void_mask: np.ndarray, cfg: FitConfig) -> float:
    vals = image[void_mask]
    print("vals.size =", vals.size)
    print("vals.min =", float(vals.min()) if vals.size > 0 else None)
    print("vals.max =", float(vals.max()) if vals.size > 0 else None)
    print("low_density_pixels =", int(np.sum(vals < 0.1)))
    print("low_density_ratio =", float(np.mean(vals < 0.1)) if vals.size > 0 else None)
    if vals.size == 0:
        return 0.0

    total_pixels = vals.size
    low_density_pixels = np.sum(vals < 0.01)

    ratio = low_density_pixels / total_pixels
    score = 0.5 - ratio

    return float(np.clip(score, 0.0, 1.0))

def _compute_anisotropy_score(eigvals: np.ndarray) -> float:
    lam1 = float(max(eigvals[0], 1e-8))
    lam2 = float(max(eigvals[1], 1e-8)) if len(eigvals) > 1 else 1e-8
    ratio = lam1 / lam2
    return float(np.clip((ratio - 2.0) / (6.0 - 2.0), 0.0, 1.0))

def _compute_neck_score(void_mask: np.ndarray, cfg: FitConfig) -> float:
    info = _project_void_points_to_major_axis(void_mask, cfg)
    if info is None:
        return 0.0

    pts = info["pts"]
    center = info["center"]
    major = info["major"]
    minor = np.array([-major[1], major[0]], dtype=np.float64)

    pts_centered = pts - center[None, :]
    major_proj = pts_centered @ major
    minor_proj = pts_centered @ minor

    lo = float(major_proj.min())
    hi = float(major_proj.max())
    if hi - lo < 1e-8:
        return 0.0

    bins = np.linspace(lo, hi, 11)
    widths = []
    centers = []

    for i in range(len(bins) - 1):
        mask = (major_proj >= bins[i]) & (major_proj < bins[i + 1])
        centers.append(0.5 * (bins[i] + bins[i + 1]))
        if mask.sum() < 3:
            widths.append(np.nan)
            continue
        w = float(minor_proj[mask].max() - minor_proj[mask].min())
        widths.append(w)

    widths = np.asarray(widths, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    valid = ~np.isnan(widths)
    if valid.sum() < 4:
        return 0.0

    widths_valid = widths[valid]
    centers_valid = centers[valid]

    # 只在中间 60% 区域找最窄处，避免被边缘干扰
    c_lo = np.quantile(centers_valid, 0.2)
    c_hi = np.quantile(centers_valid, 0.8)
    mid_mask = (centers_valid >= c_lo) & (centers_valid <= c_hi)

    if mid_mask.sum() < 2:
        return 0.0

    w_mid_min = float(np.min(widths_valid[mid_mask]))
    w_peak = float(np.max(widths_valid))

    if w_peak < 1e-8:
        return 0.0

    score = 1.0 - w_mid_min / w_peak
    return float(np.clip(score, 0.0, 1.0))


def estimate_split_prior_for_void(
    image: np.ndarray,
    void_mask: np.ndarray,
    cfg: FitConfig,
) -> Dict[str, float]:
    info = _project_void_points_to_major_axis(void_mask, cfg)

    if info is None:
        return {
            "split_confidence": 0.0,
            "split_direction_x": 1.0,
            "split_direction_y": 0.0,
            "split_distance_prior": 0.0,
            "split_bimodal_score": 0.0,
            "split_neck_score": 0.0,
            "split_anisotropy_score": 0.0,
            "split_gray_score": 0.0,
        }

    major = info["major"]
    eigvals = info["eigvals"]

    proj = info["proj"]

    bimodal_score = _compute_bimodal_score_from_projection(proj)

    neck_score = _compute_neck_score(void_mask, cfg)
    anisotropy_score = _compute_anisotropy_score(eigvals)
    gray_score = _compute_gray_score(image, void_mask, cfg)

    # 主证据
    core_score = 0.4 * bimodal_score + 0.30 * neck_score

    # 辅助证据
    aux_score = 0.05 * anisotropy_score + 0.25 * gray_score

    split_confidence = core_score + aux_score

    # 如果主证据很弱，整体压低，避免“只是细长/只是灰”就高分
    if bimodal_score < 0.05 and neck_score < 0.05:
        split_confidence *= 0.7

    split_confidence = float(np.clip(split_confidence, 0.0, 1.0))

    # 用主轴尺度的一部分作为建议分裂距离
    spread = float(np.std(proj)) if len(proj) > 1 else 0.0
    split_distance_prior = float(np.clip(1.25 * spread, 0.0, 0.25 * max(cfg.Lx, cfg.Ly)))

    return {
        "split_confidence": split_confidence,
        "split_direction_x": float(major[0]),
        "split_direction_y": float(major[1]),
        "split_distance_prior": split_distance_prior,
        "split_bimodal_score": float(bimodal_score),
        "split_neck_score": float(neck_score),
        "split_anisotropy_score": float(anisotropy_score),
        "split_gray_score": float(gray_score),
    }
def inverse_transform_points_phys(x: np.ndarray, y: np.ndarray, offsetx: float, offsety: float, rotation: float, scale: float) -> np.ndarray:
    pts = np.stack([x - offsetx, y - offsety], axis=1)
    c, s = math.cos(rotation), math.sin(rotation)
    R_inv = np.array([[c, s], [-s, c]], dtype=np.float64)
    pts_local = pts @ R_inv.T
    pts_local = pts_local / max(scale, 1e-8)
    return pts_local


def order_polygon_vertices(pts: np.ndarray) -> np.ndarray:
    ctr = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0])
    return pts[np.argsort(ang)]


def estimate_quad_from_mask(mask: np.ndarray, cfg: FitConfig) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    x, y = grid_to_phys_xy(cols, rows, cfg)
    pts = np.stack([x, y], axis=1)

    offsetx, offsety = float(np.mean(x)), float(np.mean(y))
    rotation = estimate_rotation_pca(mask, cfg)

    c, s = math.cos(rotation), math.sin(rotation)
    R_inv = np.array([[c, s], [-s, c]], dtype=np.float64)
    pts_local = (pts - np.array([[offsetx, offsety]], dtype=np.float64)) @ R_inv.T

    dirs = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    quad_local = []
    for d in dirs:
        score = pts_local @ d
        quad_local.append(pts_local[np.argmax(score)])
    quad_local = np.stack(quad_local, axis=0)

    uniq = np.unique(np.round(quad_local, decimals=6), axis=0)
    if len(uniq) < 4:
        minx, miny = pts_local.min(axis=0)
        maxx, maxy = pts_local.max(axis=0)
        quad_local = np.array([[maxx, maxy], [maxx, miny], [minx, miny], [minx, maxy]], dtype=np.float64)

    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    quad_phys = quad_local @ R.T + np.array([[offsetx, offsety]], dtype=np.float64)
    quad_phys = order_polygon_vertices(quad_phys)
    return quad_phys


def params8_from_quad_and_edge(quad_phys: np.ndarray, edge_idx: int, reverse_edge: bool = False, cfg: FitConfig = None) -> np.ndarray:
    i0 = edge_idx
    i1 = (edge_idx + 1) % 4
    a = quad_phys[i0]
    b = quad_phys[i1]
    if reverse_edge:
        a, b = b, a

    edge_vec = b - a
    scale = float(np.linalg.norm(edge_vec))
    if scale < 1e-8:
        scale = 1e-3

    rotation = normalize_rotation_half_pi(math.atan2(edge_vec[1], edge_vec[0]))

    edge_mid = 0.5 * (a + b)
    c, s = math.cos(rotation), math.sin(rotation)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    offset = edge_mid - (np.array([0.0, -0.5], dtype=np.float64) * scale) @ R.T
    offsetx, offsety = float(offset[0]), float(offset[1])

    local = inverse_transform_points_phys(quad_phys[:, 0], quad_phys[:, 1], offsetx, offsety, rotation, scale)
    top_ids = np.argsort(local[:, 1])[-2:]
    top_pts = local[top_ids]
    order = np.argsort(top_pts[:, 0])
    left_top = top_pts[order[0]]
    right_top = top_pts[order[1]]

    x1, y1 = float(right_top[0]), float(right_top[1])
    x2, y2 = float(left_top[0]), float(left_top[1])

    x1 = float(np.clip(x1, -1.8, 1.8))
    y1 = float(np.clip(y1, -1.8, 2.2))
    x2 = float(np.clip(x2, -1.8, 1.8))
    y2 = float(np.clip(y2, -1.8, 2.2))
    if cfg is not None:
        scale = float(np.clip(scale, 0.01, 2.0 * max(cfg.Lx, cfg.Ly)))

    return np.array([x1, y1, x2, y2, offsetx, offsety, rotation, scale], dtype=np.float64)


# ============================================================
# 6. 单 primitive 数值拟合
# ============================================================
def params_bounds(cfg: FitConfig):
    return [
        (-1.25, 1.25),   # x1
        (-1.20, 1.80),   # y1
        (-1.25, 1.25),   # x2
        (-1.20, 1.80),   # y2
        (-cfg.Lx, cfg.Lx),   # offsetx
        (-cfg.Ly, cfg.Ly),   # offsety
        (-math.pi/2, math.pi/2),  # rotation
        (0.02, 1.35 * max(cfg.Lx, cfg.Ly)),  # scale
    ]


def regularization_penalty(params8: np.ndarray, cfg: FitConfig) -> float:
    x1, y1, x2, y2, offsetx, offsety, _rotation, scale = params8
    penalty = 0.0

    # 1) 保持右上点在左上点右侧（避免上下点顺序混乱）
    if x1 < x2:
        penalty += 30.0 * (x2 - x1 + 1e-3) ** 2

    # 2) 不希望 scale 无限制变大
    penalty += 0.12 * (scale / max(cfg.Lx, cfg.Ly)) ** 2

    # 3) offset 超出 design box 的惩罚
    if abs(offsetx) > cfg.Lx:
        penalty += 80.0 * (abs(offsetx) - cfg.Lx) ** 2
    if abs(offsety) > cfg.Ly:
        penalty += 80.0 * (abs(offsety) - cfg.Ly) ** 2

    # 4) 顶点超出 design box 的强惩罚（这是这次最关键的新项）
    pts = params8_to_polygon(params8)
    excess_x = np.maximum(np.abs(pts[:, 0]) - cfg.Lx, 0.0)
    excess_y = np.maximum(np.abs(pts[:, 1]) - cfg.Ly, 0.0)
    penalty += cfg.vertex_outside_weight * float(np.sum(excess_x ** 2) + np.sum(excess_y ** 2))

    # 5) 上顶点不要塌得太低
    if y1 < -0.7:
        penalty += 8.0 * (-0.7 - y1) ** 2
    if y2 < -0.7:
        penalty += 8.0 * (-0.7 - y2) ** 2

    return penalty


def chamfer_like_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    ba = np.logical_and(mask_a, np.logical_not(ndimage.binary_erosion(mask_a)))
    bb = np.logical_and(mask_b, np.logical_not(ndimage.binary_erosion(mask_b)))
    if ba.sum() == 0 or bb.sum() == 0:
        return 1.0
    dist_to_b = ndimage.distance_transform_edt(~bb)
    dist_to_a = ndimage.distance_transform_edt(~ba)
    da = dist_to_b[ba].mean() if ba.any() else 0.0
    db = dist_to_a[bb].mean() if bb.any() else 0.0
    norm = max(mask_a.shape)
    return float((da + db) / max(norm, 1))


# def objective_params8(params8: np.ndarray, target_mask: np.ndarray, cfg: FitConfig) -> float:
#     pred_mask = render_from_params8(params8, cfg)
#     tgt = target_mask.astype(bool)
#     pred_f = pred_mask.astype(np.float64)
#     tgt_f = tgt.astype(np.float64)
#     mse = float(np.mean((pred_f - tgt_f) ** 2))
#     inter = np.logical_and(pred_mask, tgt).sum()
#     union = np.logical_or(pred_mask, tgt).sum()
#     iou_loss = 1.0 - float((inter + 1e-6) / (union + 1e-6))
#     boundary_loss = chamfer_like_distance(pred_mask, tgt)
#     reg = regularization_penalty(params8, cfg)
#     return 0.42 * mse + 0.30 * iou_loss + 0.18 * boundary_loss + cfg.reg_weight * reg

def objective_params8(params8: np.ndarray, target_mask: np.ndarray, cfg: FitConfig) -> float:
    pred_mask = render_from_params8(params8, cfg)
    tgt = target_mask.astype(bool)

    pred_f = pred_mask.astype(np.float64)
    tgt_f = tgt.astype(np.float64)

    mse = float(np.mean((pred_f - tgt_f) ** 2))

    inter = np.logical_and(pred_mask, tgt).sum()
    union = np.logical_or(pred_mask, tgt).sum()
    iou_loss = 1.0 - float((inter + 1e-6) / (union + 1e-6))

    boundary_loss = chamfer_like_distance(pred_mask, tgt)

    pred_area = float(pred_mask.sum())
    tgt_area = float(tgt.sum())
    area_loss = abs(pred_area - tgt_area) / (tgt_area + 1e-6)

    reg = regularization_penalty(params8, cfg)

    return (
        0.12 * mse
        + 0.15 * iou_loss
        + 0.50 * boundary_loss
        + 0.18 * area_loss
        + cfg.reg_weight * reg
    )

# def fit_single_void_params8(target_mask: np.ndarray, params0: np.ndarray, cfg: FitConfig) -> np.ndarray:
#     res = minimize(
#         objective_params8,
#         x0=params0,
#         args=(target_mask, cfg),
#         method="Powell",
#         bounds=params_bounds(cfg),
#         options={"maxiter": cfg.maxiter, "disp": False},
#     )
#     return np.asarray(res.x, dtype=np.float64)
def fit_single_void_params8(target_mask: np.ndarray, params0: np.ndarray, cfg: FitConfig) -> np.ndarray:
    bounds = params_bounds(cfg)

    params0 = np.asarray(params0, dtype=np.float64).copy()
    for i, (lb, ub) in enumerate(bounds):
        params0[i] = np.clip(params0[i], lb, ub)

    res = minimize(
        objective_params8,
        x0=params0,
        args=(target_mask, cfg),
        method="Powell",
        bounds=bounds,
        options={"maxiter": cfg.maxiter, "disp": False},
    )
    return np.asarray(res.x, dtype=np.float64)

def fit_single_void_with_edge_candidates(target_mask: np.ndarray, cfg: FitConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    quad_phys = estimate_quad_from_mask(target_mask, cfg)
    best_params = None
    best_loss = float("inf")
    best_meta: Dict[str, float] = {}

    for edge_idx in range(4):
        for reverse_edge in [False, True]:
            params0 = params8_from_quad_and_edge(quad_phys, edge_idx=edge_idx, reverse_edge=reverse_edge, cfg=cfg)
            params_opt = fit_single_void_params8(target_mask, params0, cfg)
            loss = objective_params8(params_opt, target_mask, cfg)
            if loss < best_loss:
                best_loss = float(loss)
                best_params = params_opt
                best_meta = {
                    "best_edge_idx": float(edge_idx),
                    "best_edge_reversed": float(int(reverse_edge)),
                    "best_loss": float(loss),
                }

    if best_params is None:
        raise RuntimeError("未能为该 primitive 得到有效拟合结果。")
    return best_params, best_meta


# ============================================================
# 7. 多 primitive 提取、拟合与排序
# ============================================================
def primitive_sort_key(geom12: np.ndarray) -> Tuple[float, float]:
    return float(geom12[8]), float(geom12[9])


def fit_all_primitives(void_masks: List[np.ndarray], cfg: FitConfig) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]], List[int]]:
    geom_list: List[np.ndarray] = []
    pred_union = np.zeros((cfg.nely, cfg.nelx), dtype=bool)
    metrics_list: List[Dict[str, float]] = []

    for idx, void_mask in enumerate(void_masks):
        params_opt, meta = fit_single_void_with_edge_candidates(void_mask, cfg)
        geom12 = params8_to_geom12(params_opt)
        pred_mask = render_from_params8(params_opt, cfg)
        pred_union = np.logical_or(pred_union, pred_mask)

        inter = np.logical_and(pred_mask, void_mask).sum()
        union = np.logical_or(pred_mask, void_mask).sum()
        iou = float((inter + 1e-6) / (union + 1e-6))
        mse = float(np.mean((pred_mask.astype(np.float64) - void_mask.astype(np.float64)) ** 2))
        boundary = float(chamfer_like_distance(pred_mask, void_mask))

        geom_list.append(geom12)
        metrics_list.append({
            "primitive_index": float(idx),
            "area_pixels": float(void_mask.sum()),
            "iou": iou,
            "mse": mse,
            "boundary_loss": boundary,
            **meta,
        })

    if len(geom_list) == 0:
        return np.zeros((0, 12), dtype=np.float64), pred_union, metrics_list, []

    order = sorted(range(len(geom_list)), key=lambda i: primitive_sort_key(geom_list[i]))
    geom_sorted = np.stack([geom_list[i] for i in order], axis=0)
    metrics_sorted = [metrics_list[i] for i in order]
    return geom_sorted, pred_union, metrics_sorted, order

def estimate_split_priors(
    image: np.ndarray,
    void_masks: List[np.ndarray],
    geom12_all: np.ndarray,
    cfg: FitConfig,
    order: List[int],
) -> List[Dict[str, float]]:
    priors: List[Dict[str, float]] = []

    if len(order) == 0:
        return priors

    # 关键：先把void_masks按几何排序后的顺序重排
    void_masks_sorted = [void_masks[i] for i in order]

    for idx, void_mask in enumerate(void_masks_sorted):
        prior = estimate_split_prior_for_void(image, void_mask, cfg)
        prior["primitive_index"] = float(idx)

        if idx < len(geom12_all):
            prior["offset_x"] = float(geom12_all[idx, 8])
            prior["offset_y"] = float(geom12_all[idx, 9])
            prior["rotation"] = float(geom12_all[idx, 10])
            prior["scale"] = float(geom12_all[idx, 11])

        priors.append(prior)

    return priors

def pad_geom_vectors(geom12_all: np.ndarray, max_holes: int) -> Tuple[np.ndarray, np.ndarray]:
    K = geom12_all.shape[0]
    padded = np.zeros((max_holes, 12), dtype=np.float64)
    valid = np.zeros((max_holes,), dtype=np.int64)
    keep = min(K, max_holes)
    if keep > 0:
        padded[:keep] = geom12_all[:keep]
        valid[:keep] = 1
    return padded, valid

from pathlib import Path as FilePath
import numpy as np

def format_tensor_2d(name: str, arr: np.ndarray, comment_prefix: str = "形状") -> str:
    lines = [f"{name} = torch.tensor(["]
    for i, row in enumerate(arr):
        row_str = ", ".join([f"{x:.6e}" for x in row])
        if i == len(arr) - 1:
            lines.append(f"    [{row_str}]  # {comment_prefix} {i+1}")
        else:
            lines.append(f"    [{row_str}],  # {comment_prefix} {i+1}")
    lines.append("], dtype=torch.float32)")
    return "\n".join(lines)

def format_tensor_1d(name: str, arr: np.ndarray, comment_prefix: str = "形状") -> str:
    lines = [f"{name} = torch.tensor(["]
    for i, x in enumerate(arr):
        if i == len(arr) - 1:
            lines.append(f"    {x:.6e}  # {comment_prefix} {i+1}")
        else:
            lines.append(f"    {x:.6e},  # {comment_prefix} {i+1}")
    lines.append("], dtype=torch.float32)")
    return "\n".join(lines)

def save_pytorch_style_result_txt(out_dir, geom12_all: np.ndarray, cfg: FitConfig):
    out_dir  = FilePath(out_dir)   # 关键修复：把 str 转成 Path

    fixed_chis = geom12_all[:, 0:8].copy()
    fixed_offsets = geom12_all[:, 8:10]
    fixed_rotations = geom12_all[:, 10]
    fixed_scales = geom12_all[:, 11]

    # 如果你想固定 r1-r4 = 0.05
    fixed_chis[:, 4:8] = 0.05
    fixed_offsets_shifted = fixed_offsets + np.array([cfg.Lx, cfg.Ly], dtype=np.float64)
    text_blocks = [
        "import torch",
        "",
        format_tensor_2d("fixed_chis", fixed_chis),
        "",
        format_tensor_2d("fixed_offsets", fixed_offsets_shifted),
        "",
        format_tensor_1d("fixed_scales", fixed_scales),
        "",
        format_tensor_1d("fixed_rotations", fixed_rotations*-180/math.pi),
        "",
    ]

    result_txt = "\n".join(text_blocks)

    out_path = out_dir / "result.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result_txt)

def geom12_to_backend_payload(geom12_all: np.ndarray, cfg: FitConfig) -> Dict:
    """
    将 geom12_all 转成后端 YAML 所需的初始化字段。
    对齐当前 save_pytorch_style_result_txt 的约定：
      - chis: geom12[:, 0:8]，其中 r1-r4 强制设为 0.05
      - offsets: geom12[:, 8:10] 后再加 [cfg.Lx, cfg.Ly]
      - scales: geom12[:, 11]
      - rotations: -geom12[:, 10] * 180 / pi  （角度制）
    """
    geom12_all = np.asarray(geom12_all, dtype=np.float64)
    K = int(geom12_all.shape[0])

    if K == 0:
        return {
            "N_g": 0,
            "N_g_x": 0,
            "N_g_y": 1,
            "init_centers_x": [],
            "init_centers_y": [],
            "init_scale": [],
            "init_rotation": [],
            "init_chis": [],
        }

    fixed_chis = geom12_all[:, 0:8].copy()
    fixed_chis[:, 4:8] = 0.05

    fixed_offsets = geom12_all[:, 8:10].copy()
    fixed_offsets = fixed_offsets + np.array([cfg.Lx, cfg.Ly], dtype=np.float64)

    fixed_scales = geom12_all[:, 11].copy()
    fixed_rotations = -geom12_all[:, 10].copy() * 180.0 / math.pi

    payload = {
        "N_g": K,
        "N_g_x": K,
        "N_g_y": 1,
        "init_centers_x": fixed_offsets[:, 0].tolist(),
        "init_centers_y": fixed_offsets[:, 1].tolist(),
        "init_scale": fixed_scales.tolist(),
        "init_rotation": fixed_rotations.tolist(),
        "init_chis": fixed_chis.tolist(),
    }
    return payload

def split_priors_to_backend_payload(split_priors: List[Dict[str, float]]) -> Dict:
    K = len(split_priors)

    conf = np.zeros((K,), dtype=np.float64)
    dir_x = np.zeros((K,), dtype=np.float64)
    dir_y = np.zeros((K,), dtype=np.float64)
    dist = np.zeros((K,), dtype=np.float64)

    bimodal = np.zeros((K,), dtype=np.float64)
    neck = np.zeros((K,), dtype=np.float64)
    anisotropy = np.zeros((K,), dtype=np.float64)
    gray = np.zeros((K,), dtype=np.float64)

    for i, p in enumerate(split_priors):
        conf[i] = p.get("split_confidence", 0.0)
        dir_x[i] = p.get("split_direction_x", 1.0)
        dir_y[i] = p.get("split_direction_y", 0.0)
        dist[i] = p.get("split_distance_prior", 0.0)
        bimodal[i] = p.get("split_bimodal_score", 0.0)
        neck[i] = p.get("split_neck_score", 0.0)
        anisotropy[i] = p.get("split_anisotropy_score", 0.0)
        gray[i] = p.get("split_gray_score", 0.0)

    return {
        "init_split_confidence": conf.tolist(),
        "init_split_direction_x": dir_x.tolist(),
        "init_split_direction_y": dir_y.tolist(),
        "init_split_distance_prior": dist.tolist(),
        "init_split_bimodal_score": bimodal.tolist(),
        "init_split_neck_score": neck.tolist(),
        "init_split_anisotropy_score": anisotropy.tolist(),
        "init_split_gray_score": gray.tolist(),
    }

def update_yaml_template_with_payload(
    template_yaml_path: str,
    output_yaml_path: str,
    backend_payload: Dict,
) -> Dict:
    """
    将 backend_payload 写入 YAML 模板中的 params 字段。
    除原有初始化字段外，也写入 split prior 字段。
    """
    with open(template_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    if "params" not in data or data["params"] is None:
        data["params"] = {}

    params = data["params"]

    # ===== 原有字段 =====
    params["N_g"] = int(backend_payload["N_g"])
    params["N_g_x"] = int(backend_payload["N_g_x"])
    params["N_g_y"] = int(backend_payload["N_g_y"])
    params["init_centers_x"] = [float(x) for x in backend_payload["init_centers_x"]]
    params["init_centers_y"] = [float(y) for y in backend_payload["init_centers_y"]]
    params["init_scale"] = [float(s) for s in backend_payload["init_scale"]]
    params["init_rotation"] = [float(r) for r in backend_payload["init_rotation"]]
    params["init_chis"] = [
        [float(v) for v in row] for row in backend_payload["init_chis"]
    ]

    # ===== 新增 split prior 字段 =====
    if "init_split_confidence" in backend_payload:
        params["init_split_confidence"] = [
            float(x) for x in backend_payload["init_split_confidence"]
        ]

    if "init_split_direction_x" in backend_payload:
        params["init_split_direction_x"] = [
            float(x) for x in backend_payload["init_split_direction_x"]
        ]

    if "init_split_direction_y" in backend_payload:
        params["init_split_direction_y"] = [
            float(y) for y in backend_payload["init_split_direction_y"]
        ]

    if "init_split_distance_prior" in backend_payload:
        params["init_split_distance_prior"] = [
            float(x) for x in backend_payload["init_split_distance_prior"]
        ]

    if "init_split_bimodal_score" in backend_payload:
        params["init_split_bimodal_score"] = [
            float(x) for x in backend_payload["init_split_bimodal_score"]
        ]

    if "init_split_neck_score" in backend_payload:
        params["init_split_neck_score"] = [
            float(x) for x in backend_payload["init_split_neck_score"]
        ]

    if "init_split_anisotropy_score" in backend_payload:
        params["init_split_anisotropy_score"] = [
            float(x) for x in backend_payload["init_split_anisotropy_score"]
        ]

    if "init_split_gray_score" in backend_payload:
        params["init_split_gray_score"] = [
            float(x) for x in backend_payload["init_split_gray_score"]
        ]

    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

    return data


def export_density2feature_and_yaml(
    image: np.ndarray,
    out_dir: str,
    cfg: FitConfig,
    template_yaml_path: str,
    output_yaml_path: str,
    image_prior: np.ndarray = None,
    design_mask: np.ndarray = None,   # 🔥 新增参数
) -> Dict:
    """
    一次性完成：
      1. density2feature 提取与保存
      2. 生成 backend_payload
      3. 将 payload 写入新的 YAML
    """
    result = export_density2feature_from_array(
        image=image,
        out_dir=out_dir,
        cfg=cfg,
        image_prior=image_prior,
        design_mask=design_mask,      # 🔥 关键透传
    )

    backend_payload = result["backend_payload"]
    yaml_data = update_yaml_template_with_payload(
        template_yaml_path=template_yaml_path,
        output_yaml_path=output_yaml_path,
        backend_payload=backend_payload,
    )

    result["yaml_data"] = yaml_data
    result["output_yaml_path"] = output_yaml_path
    return result
# ============================================================
# 8. 保存
# ============================================================
def save_outputs(
    out_dir: str,
    image: np.ndarray,
    material_mask: np.ndarray,
    design_void_mask: np.ndarray,
    seed_union: np.ndarray,
    voids_gt_union: np.ndarray,
    voids_pred_union: np.ndarray,
    geom12_all: np.ndarray,
    geom12_padded: np.ndarray,
    valid_mask: np.ndarray,
    metrics_list: List[Dict[str, float]],
    cfg: FitConfig,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "simp_image.npy"), image)
    np.save(os.path.join(out_dir, "material_mask.npy"), material_mask.astype(np.uint8))
    np.save(os.path.join(out_dir, "design_void_mask.npy"), design_void_mask.astype(np.uint8))
    np.save(os.path.join(out_dir, "seed_union.npy"), seed_union.astype(np.uint8))
    np.save(os.path.join(out_dir, "voids_gt_union.npy"), voids_gt_union.astype(np.uint8))
    np.save(os.path.join(out_dir, "voids_pred_union.npy"), voids_pred_union.astype(np.uint8))
    np.save(os.path.join(out_dir, "geom12_all.npy"), geom12_all)
    np.save(os.path.join(out_dir, "geom12_padded.npy"), geom12_padded)
    np.save(os.path.join(out_dir, "valid_mask.npy"), valid_mask)
    save_pytorch_style_result_txt(out_dir, geom12_all, cfg)
    # with open(os.path.join(out_dir, "result.txt"), "w", encoding="utf-8") as f:
    #     f.write("=== Number of fitted primitives ===\n")
    #     f.write(f"{geom12_all.shape[0]}\n\n")
    #     f.write("=== geom12_all (K x 12) ===\n")
    #     f.write(np.array2string(geom12_all, precision=6, suppress_small=False))
    #     f.write("\n\n")
    #     f.write("=== geom12_padded (max_holes x 12) ===\n")
    #     f.write(np.array2string(geom12_padded, precision=6, suppress_small=False))
    #     f.write("\n\n")
    #     f.write("=== valid_mask ===\n")
    #     f.write(np.array2string(valid_mask))
    #     f.write("\n\n")
    #     f.write("=== Per-primitive metrics ===\n")
    #     for item in metrics_list:
    #         f.write(str(item) + "\n")
  

def run_density2feature_core(
    image_geom: np.ndarray,
    cfg: FitConfig,
    image_prior: np.ndarray = None,
    design_mask: np.ndarray = None,
) -> Dict:
    image_geom = np.asarray(image_geom, dtype=np.float64)
    if image_geom.ndim != 2:
        raise ValueError(f"image 应为 2D 数组，当前 shape={image_geom.shape}")
    image_geom = np.clip(image_geom, 0.0, 1.0)

    if image_geom.shape != (cfg.nely, cfg.nelx):
        raise ValueError(
            f"image shape={image_geom.shape} 与 cfg.nely/cfg.nelx=({cfg.nely}, {cfg.nelx}) 不一致"
        )
    if image_prior is None:
        image_prior = image_geom.copy()
    else:
        image_prior = np.asarray(image_prior, dtype=np.float64)
        if image_prior.ndim != 2:
            raise ValueError(f"image_prior 应为 2D 数组，当前 shape={image_prior.shape}")
        image_prior = np.clip(image_prior, 0.0, 1.0)

        if image_prior.shape != (cfg.nely, cfg.nelx):
            raise ValueError(
                f"image_prior shape={image_prior.shape} 与 cfg.nely/cfg.nelx=({cfg.nely}, {cfg.nelx}) 不一致"
            )

    material_mask = preprocess_material_mask(image_geom, cfg)
    if material_mask.sum() == 0:
        raise RuntimeError("阈值化后材料区域为空，请尝试降低 threshold。")

    void_masks, voids_gt_union, design_void_mask, seed_union = extract_void_primitives(
        image=image_geom,
        material_mask=material_mask,
        cfg=cfg,
        design_mask=design_mask,  # 关键透传点
    )

    if len(void_masks) == 0:
        geom12_all = np.zeros((0, 12), dtype=np.float64)
        voids_pred_union = np.zeros_like(material_mask, dtype=bool)
        metrics_list = []
        order = []
    else:
        geom12_all, voids_pred_union, metrics_list, order = fit_all_primitives(void_masks, cfg)

    geom12_padded, valid_mask = pad_geom_vectors(geom12_all, cfg.max_holes)

    if len(metrics_list) > 0:
        mean_iou = float(np.mean([m["iou"] for m in metrics_list]))
        mean_mse = float(np.mean([m["mse"] for m in metrics_list]))
        mean_boundary_loss = float(np.mean([m["boundary_loss"] for m in metrics_list]))
    else:
        mean_iou = 0.0
        mean_mse = 1.0
        mean_boundary_loss = 1.0

    coverage = mask_iou(voids_pred_union, voids_gt_union)
    backend_payload = geom12_to_backend_payload(geom12_all, cfg)
    if len(void_masks) > 0 and len(geom12_all) > 0:
        split_priors = estimate_split_priors(image_prior, void_masks, geom12_all, cfg, order)
    else:
        split_priors = []

    split_prior_payload = split_priors_to_backend_payload(split_priors)

    backend_payload.update(split_prior_payload)

    return {
        "image": image_geom,
        "image_prior": image_prior,
        "material_mask": material_mask,
        "design_void_mask": design_void_mask,
        "seed_union": seed_union,
        "voids_gt_union": voids_gt_union,
        "voids_pred_union": voids_pred_union,
        "void_masks": void_masks,
        "geom12_all": geom12_all,
        "geom12_padded": geom12_padded,
        "valid_mask": valid_mask,
        "metrics_list": metrics_list,
        "num_primitives": int(geom12_all.shape[0]),
        "mean_iou": mean_iou,
        "mean_mse": mean_mse,
        "mean_boundary_loss": mean_boundary_loss,
        "coverage": coverage,
        "split_priors": split_priors,
        "backend_payload": backend_payload,
    }

def probe_density2feature_from_array(
    image: np.ndarray,
    cfg: FitConfig,
    min_primitives: int = 1,
    min_mean_iou: float = 0.45,
) -> Dict:
    """
    在线 probe：
    - 输入当前 SIMP 的 image_2d
    - 返回 primitives 和质量指标
    - 不落盘
    """
    result = run_density2feature_core(image, cfg)

    probe_ok = (
        result["num_primitives"] >= min_primitives
        and result["mean_iou"] >= min_mean_iou
    )

    result["probe_ok"] = bool(probe_ok)
    return result

def export_density2feature_from_array(
    image: np.ndarray,
    out_dir: str,
    cfg: FitConfig,
    image_prior: np.ndarray = None,
    design_mask: np.ndarray = None,
) -> Dict:
    """
    正式导出接口：
    - 输入二维 image
    - 运行 primitive 提取与拟合
    - 保存所有文件
    - 返回结果字典
    """
    result = run_density2feature_core(
        image, 
        cfg, 
        image_prior=image_prior, 
        design_mask=design_mask        # 🔥 关键透传
    )

    save_outputs(
        out_dir=out_dir,
        image=result["image"],
        material_mask=result["material_mask"],
        design_void_mask=result["design_void_mask"],
        seed_union=result["seed_union"],
        voids_gt_union=result["voids_gt_union"],
        voids_pred_union=result["voids_pred_union"],
        geom12_all=result["geom12_all"],
        geom12_padded=result["geom12_padded"],
        valid_mask=result["valid_mask"],
        metrics_list=result["metrics_list"],
        cfg=cfg,
    )

    return result

def pretty_print_backend_payload(payload: Dict) -> None:
    print("params:")
    print(f"  N_g: {payload['N_g']}")
    print(f"  N_g_x: {payload['N_g_x']}")
    print(f"  N_g_y: {payload['N_g_y']}")
    print(f"  init_centers_x: {payload['init_centers_x']}")
    print(f"  init_centers_y: {payload['init_centers_y']}")
    print(f"  init_scale: {payload['init_scale']}")
    print(f"  init_rotation: {payload['init_rotation']}")
    print(f"  init_chis: {payload['init_chis']}")

    if "init_split_confidence" in payload:
        print(f"  init_split_confidence: {payload['init_split_confidence']}")
    if "init_split_direction_x" in payload:
        print(f"  init_split_direction_x: {payload['init_split_direction_x']}")
    if "init_split_direction_y" in payload:
        print(f"  init_split_direction_y: {payload['init_split_direction_y']}")
    if "init_split_distance_prior" in payload:
        print(f"  init_split_distance_prior: {payload['init_split_distance_prior']}")

def export_density2feature_and_yaml_from_pt(
    pt_path: str,
    out_dir: str,
    cfg: FitConfig,
    template_yaml_path: str,
    output_yaml_path: str,
    ) -> Dict:
    """
    与 standalone main 完全一致的调用路径：
      1. 从 .pt 读取 image
      2. 调 export_density2feature_and_yaml
    """
    image, image_prior, design_mask, _meta = load_simp_images(pt_path)

    if image.shape != (cfg.nely, cfg.nelx):
        raise ValueError(
            f"读取到的图像尺寸为 {image.shape}，但配置的 (nely, nelx)=({cfg.nely}, {cfg.nelx}) 不一致。"
        )
    if image_prior.shape != (cfg.nely, cfg.nelx):
        raise ValueError(
            f"读取到的先验图像尺寸为 {image_prior.shape}，但配置的 (nely, nelx)=({cfg.nely}, {cfg.nelx}) 不一致。"
        )
    result = export_density2feature_and_yaml(
        image=image,
        out_dir=out_dir,
        cfg=cfg,
        template_yaml_path=template_yaml_path,
        output_yaml_path=output_yaml_path,
        image_prior=image_prior,
        design_mask=design_mask,   # 🔥 传下去
    )
    result["input_pt_path"] = pt_path
    return result
# ============================================================
# 9. 主程序
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 SIMP 输出中提取 design-box 内的 void primitives，并为每个 primitive 拟合一个 12 维几何向量。"
    )
    parser.add_argument("--input_pt", type=str, default=str(PROJECT_ROOT / "examples" / "smoke" / "jax_mbb_beam.pt"))
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "artifacts" / "density2feature"), help="输出目录")
    parser.add_argument("--nelx", type=int, default=90)
    parser.add_argument("--nely", type=int, default=30)
    parser.add_argument("--Lx", type=float, default=0.6)
    parser.add_argument("--Ly", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--opening_iters", type=int, default=1)
    parser.add_argument("--closing_iters", type=int, default=1)
    parser.add_argument("--min_material_area_pixels", type=int, default=8)
    parser.add_argument("--min_void_area_pixels", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--max_holes", type=int, default=8)
    parser.add_argument("--near_material_iters", type=int, default=3)
    parser.add_argument("--local_bbox_expand", type=int, default=8)
    parser.add_argument("--seed_min_area_pixels", type=int, default=3)
    parser.add_argument("--candidate_opening_iters", type=int, default=0)
    parser.add_argument("--candidate_closing_iters", type=int, default=1)
    parser.add_argument("--duplicate_iou_threshold", type=float, default=0.35)
    parser.add_argument("--template_yaml", type=str, default=str(PROJECT_ROOT / "configs" / "fmto" / "MBB_beam_half.yaml"), help="后端 YAML 模板路径")
    parser.add_argument("--output_yaml", type=str, default=str(PROJECT_ROOT / "artifacts" / "generated_configs" / "MBB_beam_half.yaml"), help="写入初始化后的 YAML 输出路径")
    args = parser.parse_args()

    cfg = FitConfig(
        nelx=args.nelx,
        nely=args.nely,
        Lx=args.Lx,
        Ly=args.Ly,
        threshold=args.threshold,
        opening_iters=args.opening_iters,
        closing_iters=args.closing_iters,
        min_material_area_pixels=args.min_material_area_pixels,
        min_void_area_pixels=args.min_void_area_pixels,
        maxiter=args.maxiter,
        max_holes=args.max_holes,
        near_material_iters=args.near_material_iters,
        local_bbox_expand=args.local_bbox_expand,
        seed_min_area_pixels=args.seed_min_area_pixels,
        candidate_opening_iters=args.candidate_opening_iters,
        candidate_closing_iters=args.candidate_closing_iters,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
    )

    # 1. 读取 SIMP 导出的图像
    image, _meta = load_simp_image(args.input_pt)
    H, W = image.shape
    if (H, W) != (cfg.nely, cfg.nelx):
        raise ValueError(
            f"读取到的图像尺寸为 {(H, W)}，但配置的 (nely, nelx)=({cfg.nely}, {cfg.nelx}) 不一致。"
        )

    # 2. 调用新的函数化导出入口
    if args.template_yaml is not None and args.output_yaml is not None:
        result = export_density2feature_and_yaml(
            image=image,
            out_dir=args.out_dir,
            cfg=cfg,
            template_yaml_path=args.template_yaml,
            output_yaml_path=args.output_yaml,
        )
    else:
        result = export_density2feature_from_array(
            image=image,
            out_dir=args.out_dir,
            cfg=cfg,
        )

    # 3. 打印摘要信息
    print("==== 提取完成 ====")
    print(f"检测到 primitive 数: {result['num_primitives']}")
    print(f"geom12_all shape: {result['geom12_all'].shape}")
    print(f"geom12_padded shape: {result['geom12_padded'].shape}")
    print(f"valid_mask: {result['valid_mask'].tolist()}")
    print(f"mean_iou: {result['mean_iou']:.6f}")
    print(f"mean_mse: {result['mean_mse']:.6f}")
    print(f"mean_boundary_loss: {result['mean_boundary_loss']:.6f}")
    print(f"coverage: {result['coverage']:.6f}")
    print(f"输出目录: {args.out_dir}")

    # 4. 额外打印后端 payload 摘要，方便检查
    payload = result["backend_payload"]
    print("==== backend payload 摘要 ====")
    pretty_print_backend_payload(payload)
    if "output_yaml_path" in result:
        print(f"已写出 YAML: {result['output_yaml_path']}")

if __name__ == "__main__":
    main()
