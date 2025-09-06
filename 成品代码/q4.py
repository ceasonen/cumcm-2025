"""
第四问：三无人机协同投放策略优化（FY1参数固定版本）
- 固定FY1为第二问最优参数
- 仅优化FY2和FY3的参数
- 采用粒子群算法求解最优投放策略
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import time
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# ======================================================================================
# 1. 系统参数与常量定义
# ======================================================================================

# 物理常量
g = 9.8                    # 重力加速度 (m/s²)
epsilon = 1e-15            # 数值计算保护阈值
dt_coarse = 0.1           # 粗略时间步长 (s)
dt_fine = 0.005           # 精细时间步长 (s)
n_jobs = multiprocessing.cpu_count()  # 并行计算核心数

# 虚假目标点定义
fake_target = np.array([0.0, 0.0, 0.0])

# 真实目标区域定义（圆柱形）
real_target = {
    "center": np.array([0.0, 200.0, 0.0]),  # 圆柱底面中心
    "r": 7.0,                                # 圆柱半径 (m)
    "h": 10.0                               # 圆柱高度 (m)
}

# 三架无人机初始位置
uav_init_positions = {
    "FY1": np.array([17800.0, 0.0, 1800.0]),
    "FY2": np.array([12000.0, 1400.0, 1400.0]),
    "FY3": np.array([6000.0, -3000.0, 700.0])
}
uav_names = ["FY1", "FY2", "FY3"]

# 烟幕弹参数
smoke_param = {
    "r": 10.0,              # 烟幕半径 (m)
    "sink_speed": 3.0,      # 下沉速度 (m/s)
    "valid_time": 20.0      # 有效持续时间 (s)
}

# 导弹M1参数
missile_param = {
    "init_pos": np.array([20000.0, 0.0, 2000.0]),  # 初始位置
    "speed": 300.0                                  # 飞行速度 (m/s)
}

# 导弹飞行参数计算
missile_dir = (fake_target - missile_param["init_pos"]) / np.linalg.norm(fake_target - missile_param["init_pos"])
missile_arrival_time = np.linalg.norm(fake_target - missile_param["init_pos"]) / missile_param["speed"]

# ======================================================================================
# 2. FY1固定参数（来自第二问优化结果）
# ======================================================================================

FY1_FIXED_PARAMS = np.array([
    0.150384,   # theta: 飞行方向角 (rad) = 8.62°
    131.59,     # v: 飞行速度 (m/s)
    0.0023,       # t1: 投放延迟 (s)
    0.4629        # t2: 起爆延迟 (s)
])

# ======================================================================================
# 3. 目标区域采样点生成
# ======================================================================================

def generate_ultra_dense_samples(target):
    """生成超高密度采样点，包含表面和内部精细网格"""
    samples = []
    center = target["center"]
    r, h = target["r"], target["h"]
    center_xy = center[:2]
    min_z, max_z = center[2], center[2] + h

    # 1. 外表面采样（顶面/底面圆周）
    theta_dense = np.linspace(0, 2*np.pi, 60, endpoint=False)  # 120->60 减少角度采样
    for z in [min_z, max_z]:
        for th in theta_dense:
            x = center_xy[0] + r * np.cos(th)
            y = center_xy[1] + r * np.sin(th)
            samples.append([x, y, z])

    # 2. 侧面采样（高度方向加密）
    heights_dense = np.linspace(min_z, max_z, 20, endpoint=True)  # 40->20 减少高度采样
    for z in heights_dense:
        for th in theta_dense:
            x = center_xy[0] + r * np.cos(th)
            y = center_xy[1] + r * np.sin(th)
            samples.append([x, y, z])

    # 3. 内部三维网格（防止漏判内部点）
    radii = np.linspace(0, r, 6, endpoint=True)  # 10->6 减少半径采样
    inner_heights = np.linspace(min_z, max_z, 15, endpoint=True)  # 30->15 减少内部高度
    inner_thetas = np.linspace(0, 2*np.pi, 16, endpoint=False)  # 24->16 减少内部角度
    for z in inner_heights:
        for rad in radii:
            for th in inner_thetas:
                x = center_xy[0] + rad * np.cos(th)
                y = center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])

    # 4. 边缘过渡区加密（提升边界判定精度）
    edge_radii = np.linspace(r*0.95, r*1.05, 3, endpoint=True)  # 5->3 减少边缘半径
    for z in np.linspace(min_z, max_z, 6):  # 10->6 减少边缘高度
        for rad in edge_radii:
            for th in np.linspace(0, 2*np.pi, 30, endpoint=False):  # 60->30 减少边缘角度
                x = center_xy[0] + rad * np.cos(th)
                y = center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])

    return np.unique(np.array(samples), axis=0)

# ======================================================================================
# 4. 几何计算与遮蔽判定算法
# ======================================================================================

def vector_norm(v):
    """计算向量的模长"""
    return np.sqrt(np.sum(v**2))

def segment_sphere_intersection(M, P, C, r):
    """
    计算线段与球的相交长度比例
    M: 导弹位置, P: 目标采样点, C: 烟幕中心, r: 烟幕半径
    """
    MP = P - M
    MC = C - M
    a = np.dot(MP, MP)

    # 处理零长度线段情况
    if a < epsilon:
        dist = vector_norm(MC)
        return 1.0 if dist <= r + epsilon else 0.0

    b = -2 * np.dot(MP, MC)
    c = np.dot(MC, MC) - r**2
    discriminant = b**2 - 4*a*c

    # 无交点情况
    if discriminant < -epsilon:
        return 0.0
    if discriminant < 0:
        discriminant = 0.0

    sqrt_d = np.sqrt(discriminant)
    s1 = (-b - sqrt_d) / (2*a)
    s2 = (-b + sqrt_d) / (2*a)

    # 计算有效交区间
    s_start = max(0.0, min(s1, s2))
    s_end = min(1.0, max(s1, s2))

    return max(0.0, s_end - s_start)

def is_fully_shielded_single_smoke(missile_pos, smoke_center, smoke_r, target_samples):
    """判断单枚烟幕是否完全遮蔽目标"""
    for p in target_samples:
        if segment_sphere_intersection(missile_pos, p, smoke_center, smoke_r) < epsilon:
            return False
    return True

def is_any_smoke_effective(missile_pos, smoke_list, target_samples):
    """判断多枚烟幕中是否有至少一枚实现完全遮蔽"""
    for smoke in smoke_list:
        if is_fully_shielded_single_smoke(missile_pos, smoke["center"], smoke["r"], target_samples):
            return True
    return False

# ======================================================================================
# 5. 自适应时间步长生成
# ======================================================================================

def get_adaptive_time_steps(t_start, t_end, event_times=None):
    """
    生成自适应时间序列
    在关键事件点（烟幕起爆时间）附近使用精细步长
    """
    if event_times is None or len(event_times) == 0:
        return np.arange(t_start, t_end + dt_coarse, dt_coarse)

    # 确定精细时段
    fine_intervals = []
    for et in event_times:
        fine_start = max(t_start, et - 1.0)
        fine_end = min(t_end, et + 1.0)
        fine_intervals.append((fine_start, fine_end))

    # 合并重叠的精细时段
    if not fine_intervals:
        return np.arange(t_start, t_end + dt_coarse, dt_coarse)
    
    fine_intervals.sort()
    merged = [list(fine_intervals[0])]
    
    for current in fine_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(list(current))

    # 生成时间序列
    times = []
    prev_end = t_start
    
    for (f_start, f_end) in merged:
        # 粗步长段
        if prev_end < f_start:
            times.extend(np.arange(prev_end, f_start, dt_coarse))
        # 精细步长段
        times.extend(np.arange(f_start, f_end + dt_fine, dt_fine))
        prev_end = f_end
    
    # 最后的粗步长段
    if prev_end < t_end:
        times.extend(np.arange(prev_end, t_end + dt_coarse, dt_coarse))

    return np.unique(times)

# ======================================================================================
# 6. 适应度函数（多无人机联合优化）
# ======================================================================================

def fitness_function(params, target_samples):
    """
    计算三无人机协同策略的总遮蔽时长
    params: 12维参数向量
    前4维：FY1参数（固定值）
    中4维：FY2参数（待优化）
    后4维：FY3参数（待优化）
    """
    # 解析各无人机参数
    fy1_params = params[0:4]
    fy2_params = params[4:8]
    fy3_params = params[8:12]
    all_uav_params = [fy1_params, fy2_params, fy3_params]

    # 约束检查与烟幕信息收集
    smoke_info_list = []
    event_times = []
    
    for idx, (uav_name, uav_params) in enumerate(zip(uav_names, all_uav_params)):
        theta, v, t1, t2 = uav_params

        # 速度约束检查
        if not (70.0 - epsilon <= v <= 140.0 + epsilon):
            return 0.0
        
        # 时间约束检查
        if t1 < -epsilon or t2 < -epsilon:
            return 0.0

        # 计算投放点
        uav_init = uav_init_positions[uav_name]
        uav_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
        drop_point = uav_init + v * t1 * uav_dir

        # 计算起爆点（自由落体运动）
        det_xy = drop_point[:2] + v * t2 * uav_dir[:2]
        det_z = drop_point[2] - 0.5 * g * t2**2
        
        # 起爆高度约束
        if det_z < 3.0:
            return 0.0
        
        det_point = np.array([det_xy[0], det_xy[1], det_z])

        # 计算烟幕时间窗口
        t_det = t1 + t2
        t_smoke_start = t_det
        t_smoke_end = t_det + smoke_param["valid_time"]
        
        # 检查烟幕是否在导弹到达前生效
        if t_smoke_start >= missile_arrival_time - epsilon:
            return 0.0

        # 存储烟幕信息
        smoke_info = {
            "t_start": t_smoke_start,
            "t_end": t_smoke_end,
            "det_point": det_point,
            "uav_dir": uav_dir,
            "v": v
        }
        smoke_info_list.append(smoke_info)
        event_times.append(t_det)

    # 确定全局时间窗口
    global_t_start = min([s["t_start"] for s in smoke_info_list])
    global_t_end = min([s["t_end"] for s in smoke_info_list] + [missile_arrival_time])
    
    if global_t_start >= global_t_end - epsilon:
        return 0.0

    # 生成自适应时间序列
    t_list = get_adaptive_time_steps(global_t_start, global_t_end, event_times)
    if len(t_list) == 0:
        return 0.0

    # 逐时刻计算遮蔽状态
    total_shield_time = 0.0
    prev_t = None
    
    for t in t_list:
        if prev_t is not None:
            dt_current = t - prev_t

            # 计算导弹位置
            missile_pos = missile_param["init_pos"] + missile_param["speed"] * t * missile_dir

            # 计算当前有效烟幕
            current_smokes = []
            for smoke in smoke_info_list:
                if not (smoke["t_start"] - epsilon <= t <= smoke["t_end"] + epsilon):
                    continue
                
                # 计算烟幕下沉后的位置
                sink_time = t - smoke["t_start"]
                smoke_z = smoke["det_point"][2] - smoke_param["sink_speed"] * sink_time
                
                if smoke_z < 1.0:
                    continue
                
                smoke_center = np.array([smoke["det_point"][0], smoke["det_point"][1], smoke_z])
                current_smokes.append({"center": smoke_center, "r": smoke_param["r"]})

            # 判定遮蔽效果
            if len(current_smokes) > 0 and is_any_smoke_effective(missile_pos, current_smokes, target_samples):
                total_shield_time += dt_current

        prev_t = t

    # 8. 微调边界奖励机制
    boundary_bonus = 0.0
    
    for idx, params in enumerate(all_uav_params):
        if idx == 0:  # 跳过FY1，因为参数固定
            continue
            
        theta, v, t1, t2 = params
        
        # 速度边界适度奖励
        if abs(v - 70) < 7 or abs(v - 140) < 7:
            boundary_bonus += 0.6  # 从0.6减少到0.35
        
        # 时间参数适度奖励
        if t1 < 7 or t1 > 38:  # 稍微放宽范围
            boundary_bonus += 0.5   # 从0.5减少到0.3
        if t2 < 7 or t2 > 14:  # 稍微调整范围
            boundary_bonus += 0.5   # 从0.5减少到0.3
            
        # 角度多样性适度奖励
        theta_deg = np.degrees(theta) % 360
        if 75 <= theta_deg <= 105 or 255 <= theta_deg <= 285:  # 保持范围
            boundary_bonus += 0.25  # 从0.45减少到0.25
        if 165 <= theta_deg <= 195 or 345 <= theta_deg <= 15:  # 保持范围
            boundary_bonus += 0.25  # 从0.45减少到0.25
            
        # 协同配合稍微增加奖励
        if idx == 1:  # FY2
            if 85 <= v <= 115:  # 保持中等速度范围
                boundary_bonus += 0.2  # 从0.4减少到0.2
            if 6 <= t1 <= 20:  # 保持快速响应范围
                boundary_bonus += 0.2  # 从0.4减少到0.2
        elif idx == 2:  # FY3
            if v >= 115:  # 保持高速阈值
                boundary_bonus += 0.2  # 从0.4减少到0.2
            if t2 >= 12:  # 保持延迟起爆范围
                boundary_bonus += 0.2  # 从0.4减少到0.2
                
        # 适度极值组合奖励
        if (v <= 82 or v >= 128) and (t1 <= 12 or t1 >= 38):
            boundary_bonus += 0.6  # 从0.8减少到0.4
        if (t2 <= 9 or t2 >= 14) and (75 <= theta_deg <= 105 or 255 <= theta_deg <= 285):
            boundary_bonus += 0.5  # 从0.65减少到0.35
            
        # 适度三参数联合极值奖励
        extreme_count = 0
        if v <= 80 or v >= 130: extreme_count += 1  # 稍微放宽条件
        if t1 <= 10 or t1 >= 42: extreme_count += 1  # 稍微放宽条件
        if t2 <= 6 or t2 >= 16: extreme_count += 1   # 稍微放宽条件
        
        if extreme_count >= 2:
            boundary_bonus += 0.5  # 从0.9减少到0.45

    return total_shield_time + boundary_bonus

# ======================================================================================
# 7. 粒子群优化算法（固定FY1参数版本）
# ======================================================================================

class ParticleSwarmOptimizer:
    def __init__(self, objective_func, bounds, fixed_fy1_params, num_particles=80, max_iter=100,
                 c1=1.8, c2=1.8, w_start=0.9, w_end=0.3):
        """粒子群优化：固定FY1的4个参数，仅优化FY2/FY3的8个参数"""
        self.objective_func = objective_func
        self.bounds = bounds # 12维参数边界（前4维为FY1固定值的边界，实际不优化）
        self.fixed_fy1_params = fixed_fy1_params # FY1固定参数
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.c1 = c1 # 认知系数
        self.c2 = c2 # 社会系数
        self.w_start = w_start # 初始惯性权重
        self.w_end = w_end # 结束惯性权重

        self.dim = len(bounds) # 总维度=12
        self.optimize_dim_idx = list(range(4, 12)) # 待优化的维度索引（FY2/FY3的8个参数）
        self.fixed_dim_idx = list(range(0, 4)) # 固定的维度索引（FY1的4个参数）
        self._init_particles()

    def _init_particles(self):
        """初始化粒子位置和速度：FY1参数固定，仅FY2/FY3随机初始化"""
        # 位置：前4维固定为FY1参数，后8维随机初始化
        self.positions = np.zeros((self.num_particles, self.dim))
        for i in range(self.num_particles):
            # 固定FY1的4个参数
            self.positions[i, self.fixed_dim_idx] = self.fixed_fy1_params
            # 随机初始化FY2/FY3的8个参数（在边界内）
            for j in self.optimize_dim_idx:
                min_val, max_val = self.bounds[j]
                self.positions[i, j] = np.random.uniform(min_val, max_val)

        # 速度：前4维固定为0（不移动），后8维初始为边界范围的40%（极大增强初始搜索）
        self.velocities = np.zeros((self.num_particles, self.dim))
        for i in range(self.num_particles):
            for j in self.optimize_dim_idx:
                min_val, max_val = self.bounds[j]
                vel_range = max_val - min_val
                self.velocities[i, j] = np.random.uniform(-0.4*vel_range, 0.4*vel_range)  # 从30%增加到40%

        # 个体最优：初始为自身位置
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.array([self.objective_func(p) for p in self.positions])

        # 全局最优：初始为个体最优中的最大值
        self.gbest_idx = np.argmax(self.pbest_fitness)
        self.gbest_position = self.pbest_positions[self.gbest_idx].copy()
        self.gbest_fitness = self.pbest_fitness[self.gbest_idx]

        # 记录迭代历史
        self.gbest_history = [self.gbest_fitness]

    def _constrain_pos(self, pos, dim):
        """约束位置在边界内（仅对优化维度生效）"""
        if dim in self.fixed_dim_idx:
            return self.fixed_fy1_params[dim] # 固定维度直接返回FY1参数
        min_val, max_val = self.bounds[dim]
        if pos < min_val:
            return min_val + 0.01 * np.random.randn() # 边界附近小幅扰动
        elif pos > max_val:
            return max_val + 0.01 * np.random.randn()
        return pos

    def _constrain_vel(self, vel, dim):
        """约束速度（仅对优化维度生效，固定维度速度为0）"""
        if dim in self.fixed_dim_idx:
            return 0.0
        min_val, max_val = self.bounds[dim]
        vel_limit = 0.5 * (max_val - min_val)  # 从40%增加到50%，极大增强搜索能力
        return np.clip(vel, -vel_limit, vel_limit)

    def optimize(self):
        """执行粒子群优化：仅更新FY2/FY3的参数"""
        for iter in range(1, self.max_iter + 1):
            # 线性减小惯性权重（增强后期局部搜索）
            w = self.w_start - (self.w_start - self.w_end) * (iter / self.max_iter)

            # 并行计算所有粒子的适应度（加速优化）
            fitness_vals = Parallel(n_jobs=n_jobs)(
                delayed(self.objective_func)(self.positions[i])
                for i in range(self.num_particles)
            )
            fitness_vals = np.array(fitness_vals)

            # 更新个体最优和全局最优
            for i in range(self.num_particles):
                if fitness_vals[i] > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness_vals[i]
                    self.pbest_positions[i] = self.positions[i].copy()
                # 更新全局最优
                if fitness_vals[i] > self.gbest_fitness:
                    self.gbest_fitness = fitness_vals[i]
                    self.gbest_position = self.positions[i].copy()

            # 更新粒子速度和位置（仅对优化维度操作）
            for i in range(self.num_particles):
                # 随机因子
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # 速度更新公式（仅优化维度生效）
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])
                new_vel = w * self.velocities[i] + cognitive + social

                # 约束速度（固定维度速度为0）
                for j in range(self.dim):
                    new_vel[j] = self._constrain_vel(new_vel[j], j)
                self.velocities[i] = new_vel

                # 位置更新（仅优化维度更新，固定维度不变）
                new_pos = self.positions[i] + new_vel
                for j in range(self.dim):
                    new_pos[j] = self._constrain_pos(new_pos[j], j)
                self.positions[i] = new_pos

            # 记录历史最优
            self.gbest_history.append(self.gbest_fitness)

            # 打印迭代信息
            if iter % 10 == 0 or iter == 1:
                print(f"迭代 {iter:3d}/{self.max_iter}, 最优遮蔽时长: {self.gbest_fitness:.4f} s")

        return self.gbest_position, self.gbest_fitness, self.gbest_history

# ======================================================================================
# 8. 结果分析与输出模块
# ======================================================================================

def parse_best_params(best_params):
    """解析最优参数为可读的策略信息"""
    strategy_list = []
    
    for idx, uav_name in enumerate(uav_names):
        # 提取参数
        theta = best_params[idx*4 + 0]
        v = best_params[idx*4 + 1]
        t1 = best_params[idx*4 + 2]
        t2 = best_params[idx*4 + 3]

        # 计算关键位置
        uav_init = uav_init_positions[uav_name]
        uav_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
        drop_point = uav_init + v * t1 * uav_dir
        det_xy = drop_point[:2] + v * t2 * uav_dir[:2]
        det_z = drop_point[2] - 0.5 * g * t2**2
        det_point = np.array([det_xy[0], det_xy[1], det_z])
        t_det = t1 + t2

        # 构建策略信息
        strategy = {
            "无人机编号": uav_name,
            "飞行方向角(rad)": round(theta, 6),
            "飞行方向角(°)": round(np.degrees(theta), 2),
            "飞行速度(m/s)": round(v, 4),
            "投放延迟(s)": round(t1, 4),
            "起爆延迟(s)": round(t2, 4),
            "投放点X(m)": round(drop_point[0], 2),
            "投放点Y(m)": round(drop_point[1], 2),
            "投放点Z(m)": round(drop_point[2], 2),
            "起爆点X(m)": round(det_point[0], 2),
            "起爆点Y(m)": round(det_point[1], 2),
            "起爆点Z(m)": round(det_point[2], 2),
            "起爆时刻(s)": round(t_det, 2),
            "烟幕失效时刻(s)": round(t_det + smoke_param["valid_time"], 2)
        }
        strategy_list.append(strategy)

    return strategy_list

def save_to_excel(strategy_list, total_shield_time, filename="result_fy1_fixed.xlsx"):
    """保存结果到Excel文件"""
    df = pd.DataFrame(strategy_list)
    
    # 添加总遮蔽时长信息
    total_row = pd.DataFrame({col: [""] for col in df.columns})
    total_row.iloc[0, 0] = "总遮蔽时长"
    total_row.iloc[0, 1] = f"{total_shield_time:.4f} s"
    df = pd.concat([df, total_row], ignore_index=True)

    # 保存文件
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="三无人机投放策略(FY1固定)", index=False)
    
    print(f"\n结果已保存到: {filename}")
    return df

# ======================================================================================
# 9. 主程序执行流程
# ======================================================================================

if __name__ == "__main__":
    start_total = time.time()

    # Step 1: 生成目标采样点
    print("=" * 50)
    print("Step 1: 生成目标区域采样点...")
    target_samples = generate_ultra_dense_samples(real_target)
    print(f"目标采样点数量: {len(target_samples):,}")

    # Step 2: 定义参数边界
    param_bounds = []
    for _ in range(3):  # 三架无人机
        param_bounds.extend([
            (0.0, 2*np.pi),    # theta: 飞行方向角
            (70.0, 140.0),     # v: 飞行速度
            (0.0, 100.0),      # t1: 投放延迟
            (0.0, 30.0)        # t2: 起爆延迟
        ])

    # Step 3: 定义适应度函数
    def objective(params):
        return fitness_function(params, target_samples)

    # Step 4: 启动粒子群优化
    print("\nStep 2: 启动粒子群优化（FY1固定，优化FY2/FY3）...")
    pso = ParticleSwarmOptimizer(
        objective_func=objective,
        bounds=param_bounds,
        fixed_fy1_params=FY1_FIXED_PARAMS,
        num_particles=150,  
        max_iter=100,       
        c1=3.0,            
        c2=3.0,            
        w_start=0.98,      
        w_end=0.08         
    )
    
    best_params, best_fitness, history = pso.optimize()

    # Step 5: 验证最优解
    print("\nStep 3: 验证最优解...")
    verify_fitness = fitness_function(best_params, target_samples)
    print(f"优化阶段总遮蔽时长: {best_fitness:.4f} s")
    print(f"验证阶段总遮蔽时长: {verify_fitness:.4f} s")

    # Step 6: 解析结果并保存
    print("\nStep 4: 解析策略并保存结果...")
    strategy_list = parse_best_params(best_params)
    result_df = save_to_excel(strategy_list, verify_fitness)

    # Step 7: 输出结果汇总
    print("\n" + "=" * 80)
    print("【优化结果汇总：FY1固定第二问参数，FY2/FY3优化】")
    print(f"总有效遮蔽时长: {verify_fitness:.4f} s")
    print("\n各无人机策略详情:")
    
    for i, strategy in enumerate(strategy_list):
        print(f"\n{i+1}. {strategy['无人机编号']}:")
        print(f"   飞行方向: {strategy['飞行方向角(°)']}° | 速度: {strategy['飞行速度(m/s)']} m/s")
        print(f"   投放延迟: {strategy['投放延迟(s)']}s | 起爆延迟: {strategy['起爆延迟(s)']}s")
        print(f"   起爆点: ({strategy['起爆点X(m)']}, {strategy['起爆点Y(m)']}, {strategy['起爆点Z(m)']}) m")
        print(f"   烟幕有效时段: [{strategy['起爆时刻(s)']}s, {strategy['烟幕失效时刻(s)']}s]")
        if i == 0:
            print(f"   注：该无人机参数为第二问固定结果，未参与本次优化")
    
    print("=" * 80)

    # Step 8: 绘制收敛曲线
    plt.figure(figsize=(12, 8))
    plt.plot(history, linewidth=2, color="#2E86AB", marker='o', markersize=3)
    plt.scatter(np.argmax(history), np.max(history), color="#A1346E", s=100, 
                label=f"最优值: {np.max(history):.4f}s", zorder=5)
    plt.title("FY1固定第二问参数 | FY2/FY3优化收敛曲线", fontsize=16, fontweight="bold")
    plt.xlabel("迭代次数", fontsize=14)
    plt.ylabel("总遮蔽时长 (s)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("fy1_fixed_convergence.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Step 9: 计算总耗时
    total_time = time.time() - start_total
    print(f"\n程序总耗时: {total_time:.2f} 秒")
