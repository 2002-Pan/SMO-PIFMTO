import sys
import os
import re
import csv
import builtins
from datetime import datetime

# ==============================================================================
# 🚨 终极环境修复：动态锚定绝对路径 (无视终端运行位置)
# ==============================================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

work_dir = os.path.join(project_root, 'NN_TopOpt')
if os.path.exists(work_dir):
    os.chdir(work_dir)

print(f"🔧 [环境修复] 已挂载项目根目录: {project_root}")
print(f"🔧 [环境修复] 已切换工作目录至: {os.getcwd()}")
# ==============================================================================

from NN_TopOpt.NN_TopOpt import FeatureMappingDecSDF, FeatureMappingTopOptMMA, SIMP_basic
from NN_TopOpt.TopOpt import TopOptimizer2D, TopOptimizer2D_Multi
import yaml
import torch

torch.cuda.empty_cache()

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../models'))
sys.path.append(os.path.abspath('../dataset_generation'))

methods = {
    "fm_nn_heaviside": FeatureMappingTopOptMMA,
}

dataset_type = "quadrangle"
root_dir = "../src/model_comparison/NHSDF_MMA"

if dataset_type == "tripple":
    experiment_dir = "../configs/NN_top_optimization/FM_NN_Heaviside_MMA/VAE_DeepSDF"
elif dataset_type == "quadrangle":
    experiment_dir = os.path.join(project_root, "configs", "fmto")
    root_dir += "_quad"
elif dataset_type == "triangle":
    experiment_dir = "../configs/NN_top_optimization/FM_NN_Heaviside_MMA/VAE_DeepSDF_triangle"
    root_dir += "_triangle"

config_name = "Cant_beam_distrL"
config_path = os.environ.get("FMTO_CONFIG", f"{experiment_dir}/{config_name}.yaml")

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

params = config['params']

# ==============================================================================
# 📊 实时 CSV 日志：拦截内部 print，解析每轮 compliance 和 volfrac
# ==============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.abspath("./opt_logs")
os.makedirs(log_dir, exist_ok=True)
csv_path = os.path.join(log_dir, f"OptLog_{config_name}_{timestamp}.csv")

# 这个正则匹配你后端每轮打印的格式：
# Iteration: 12 | current volfrac: 0.498123 | compliance: 123.456789
iter_line_pattern = re.compile(
    r"Iteration:\s*(\d+)\s*\|\s*current volfrac:\s*([0-9eE+\-.]+)\s*\|\s*compliance:\s*([0-9eE+\-.]+)"
)

# 可选：记录分裂事件
reopen_pattern = re.compile(
    r"\[Reopen\]\s*parent=(\d+)\s*->\s*children=\((\d+),\s*(\d+)\)"
)

_original_print = builtins.print
split_event_id = {"value": 0}   # 用 dict 是为了在闭包里可修改

# 初始化 CSV 表头
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Iteration",
        "Compliance",
        "VolumeFraction",
        "SplitEventID",
        "RawMessage",
    ])



def patched_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)

    # 先正常打印到终端
    _original_print(*args, **kwargs)

    # 检查是否是分裂事件
    reopen_match = reopen_pattern.search(msg)
    if reopen_match:
        split_event_id["value"] += 1

    # 检查是否是每轮迭代日志
    iter_match = iter_line_pattern.search(msg)
    if iter_match:
        iteration = int(iter_match.group(1))
        volfrac = float(iter_match.group(2))
        compliance = float(iter_match.group(3))

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                compliance,
                volfrac,
                split_event_id["value"],
                msg,
            ])
            f.flush()

# 替换全局 print
builtins.print = patched_print

print(f"📁 实时 CSV 将保存到: {csv_path}")

try:
    TopOptimizer = TopOptimizer2D(methods, params)
    # TopOptimizer = TopOptimizer2D_Multi(methods, params)
    TopOptimizer.optimize(plot_interval=20)
    vf, compliance, max_stress = TopOptimizer.print_metrics()
# ==============================================================================
    # 📊 导出最终特征值 (基础 4 特征 + 解码后的 8 形状特征)
    # ==============================================================================
    print("\n💾 正在提取并解码最终特征数据...")
    try:
        # 1. 定位到内部模型
        # TopOptimizer2D 内部通常存储在 .models 字典中
        # 1. 定位到内部模型 (兼容单模型和多模型)
        
        opt_model = TopOptimizer.method
        
        core = opt_model.gaussian_core
        
        # 2. 获取存活特征的掩码
        mask = core.persistent_mask # [cite: 456]
        
        # 3. 提取原始权重数据
        w_s = core.W_scale.data[mask] # [cite: 452]
        w_z = core.W_shape_var.data[mask] # [cite: 453]
        w_r = core.W_rotation.data[mask] # [cite: 454]
        w_o = core.W_offsets.data[mask] # [cite: 455]

        # 4. 映射到物理值 (复刻 get_x 内部逻辑) 
        if core.sigmoid_limits:
            phys_scale = (core.scale_max - core.scale_min) * torch.sigmoid(w_s) + core.scale_min
            phys_z = (core.shape_var_maxs - core.shape_var_mins) * torch.sigmoid(w_z) + core.shape_var_mins
            phys_rot = core.rotation_min + (core.rotation_max - core.rotation_min) * torch.sigmoid(w_r).view(-1)
            phys_off = core.coord_min + (core.coord_max - core.coord_min) * torch.sigmoid(w_o)
        else:
            phys_scale, phys_z, phys_rot, phys_off = w_s, w_z, w_r, w_o

        # 5. 【核心步骤】将潜变量 z 解码为 8 维 chi 特征
        with torch.no_grad():
            # 调用你提供的 latent_to_chi 接口
            # 注意：如果 phys_z 是 batch，确保 latent_to_chi 处理了 batch
            phys_chi = core.shape_bridge.latent_to_chi(phys_z)
            
        # 6. 转为 Numpy 准备保存
        phys_scale = phys_scale.detach().cpu().numpy()
        phys_rot = phys_rot.detach().cpu().numpy()
        phys_off = phys_off.detach().cpu().numpy()
        phys_chi = phys_chi.detach().cpu().numpy()

        # 7. 写入 CSV
        feature_csv_path = os.path.join(log_dir, f"Features_Decoded_{config_name}_{timestamp}.csv")
        with open(feature_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 表头：ID, 4基础特征, 8形状特征
            header = ["ID", "Scale", "Rotation", "Offset_X", "Offset_Y"]
            header += [f"Chi_{i+1}" for i in range(phys_chi.shape[1])]
            writer.writerow(header)

            for i in range(len(phys_scale)):
                row = [
                    i, 
                    phys_scale[i], 
                    phys_rot[i], 
                    phys_off[i, 0], 
                    phys_off[i, 1]
                ]
                row += phys_chi[i].tolist()
                writer.writerow(row)

        print(f"✅ 成功导出 {len(phys_scale)} 个特征的解码数据至: {feature_csv_path}")

    except Exception as e:
        print(f"❌ 提取特征数据失败: {e}")
finally:
    # 无论成功还是报错，都恢复原始 print
    builtins.print = _original_print

print("\n✅ 优化结束")
print(f"📄 CSV 日志已保存: {csv_path}")