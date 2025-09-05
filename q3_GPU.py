#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3: 三枚烟幕弹协同策略 (最终稳定版 - 显存优化 + 结果输出 + 精确复核)
"""

import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import time
import gc

# Matplotlib 中文显示设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 1. 常量与基础参数定义 --------------------------
g = 9.81
epsilon = 1e-7
dt_fine = 0.01
DTYPE = np.float32
CP_DTYPE = cp.float32

fake_target = np.array([0.0, 0.0, 0.0], dtype=DTYPE)
real_target = {"center": np.array([0.0, 200.0, 0.0], dtype=DTYPE), "r": 7.0, "h": 10.0}
fy1_init_pos = np.array([17800.0, 0.0, 1800.0], dtype=DTYPE)
smoke_param = {"r": 10.0, "sink_speed": 3.0, "valid_time": 20.0}
missile_param = {
    "init_pos": np.array([20000.0, 0.0, 2000.0], dtype=DTYPE),
    "speed": 300.0,
    "dir": (fake_target - np.array([20000.0, 0.0, 2000.0], dtype=DTYPE)) / np.linalg.norm(fake_target - np.array([20000.0, 0.0, 2000.0], dtype=DTYPE))
}
missile_arrival_time = np.linalg.norm(fake_target - missile_param["init_pos"]) / missile_param["speed"]

# -------------------------- 2. 真目标高密度采样 (CPU) --------------------------
def generate_target_samples(target, num_theta=60, num_height=20):
    # (代码与原版相同，此处省略)
    samples = []
    center_xy = target["center"][:2]
    min_z = target["center"][2]
    max_z = target["center"][2] + target["h"]
    thetas = np.linspace(0, 2*np.pi, num_theta, endpoint=False, dtype=DTYPE)
    heights = np.linspace(min_z, max_z, num_height, endpoint=True, dtype=DTYPE)
    
    for th in thetas:
        x = center_xy[0] + target["r"] * np.cos(th)
        y = center_xy[1] + target["r"] * np.sin(th)
        samples.append([x, y, min_z])
        samples.append([x, y, max_z])
    
    for z in heights:
        for th in thetas:
            x = center_xy[0] + target["r"] * np.cos(th)
            y = center_xy[1] + target["r"] * np.sin(th)
            samples.append([x, y, z])

    inner_radii = np.linspace(0, target["r"], 5, endpoint=True, dtype=DTYPE)
    inner_heights = np.linspace(min_z, max_z, 10, endpoint=True, dtype=DTYPE)
    inner_thetas = np.linspace(0, 2*np.pi, 20, endpoint=False, dtype=DTYPE)
    for z in inner_heights:
        for rad in inner_radii:
            for th in inner_thetas:
                x = center_xy[0] + rad * np.cos(th)
                y = center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])
                
    return np.unique(np.array(samples, dtype=DTYPE), axis=0)
# -------------------------- 3. GPU 核心计算 (用于PSO优化) --------------------------
def get_shield_durations_batch(bomb_params_batch, target_samples_gpu):
    # (代码与原版相同，此处重命名以示区分)
    num_bombs = bomb_params_batch.shape[0]
    
    theta = bomb_params_batch[:, 0:1]
    v = bomb_params_batch[:, 1:2]
    t1 = bomb_params_batch[:, 2:3]
    t2 = bomb_params_batch[:, 3:4]
    
    uav_dir_batch = cp.concatenate([cp.cos(theta), cp.sin(theta), cp.zeros_like(theta)], axis=1)
    
    drop_point_batch = cp.asarray(fy1_init_pos, dtype=CP_DTYPE) + v * t1 * uav_dir_batch
    
    det_xy_batch = drop_point_batch[:, :2] + (v * t2 * uav_dir_batch[:, :2])
    det_z_batch = drop_point_batch[:, 2] - 0.5 * g * t2.squeeze()**2
    det_point_batch = cp.concatenate([det_xy_batch, det_z_batch.reshape(-1, 1)], axis=1)

    t_det_batch = (t1 + t2).squeeze()
    t_smoke_start_batch = t_det_batch
    t_smoke_end_batch = cp.minimum(t_det_batch + smoke_param["valid_time"], missile_arrival_time)

    max_duration = smoke_param["valid_time"]
    num_timesteps = int(max_duration / dt_fine) + 1
    t_relative = cp.linspace(0, max_duration, num_timesteps, dtype=CP_DTYPE)
    
    t_absolute_batch = t_relative + t_smoke_start_batch[:, None]
    
    valid_time_mask = t_absolute_batch <= t_smoke_end_batch[:, None]
    
    missile_pos_batch = cp.asarray(missile_param["init_pos"], dtype=CP_DTYPE) + \
                        missile_param["speed"] * t_absolute_batch[..., None] * cp.asarray(missile_param["dir"], dtype=CP_DTYPE)
    
    sink_time_batch = t_absolute_batch - t_det_batch[:, None]
    smoke_center_batch = det_point_batch[:, None, :] - \
                         cp.array([0, 0, smoke_param["sink_speed"]], dtype=CP_DTYPE)[None, None, :] * sink_time_batch[..., None]
    
    T, N = num_timesteps, len(target_samples_gpu)
    M = missile_pos_batch[:, :, None, :]
    P = target_samples_gpu[None, None, :, :]
    C = smoke_center_batch[:, :, None, :]

    MP = P - M
    MC = C - M
    a = cp.sum(MP * MP, axis=3)
    b = -2 * cp.sum(MP * MC, axis=3)
    c = cp.sum(MC * MC, axis=3) - smoke_param["r"]**2
    discriminant = b**2 - 4*a*c
    
    intersect = cp.zeros_like(a, dtype=cp.bool_)
    valid_mask = (discriminant >= -epsilon)
    
    a_valid = a[valid_mask]
    b_valid = b[valid_mask]
    a_valid[a_valid < epsilon] = epsilon
    
    sqrt_d = cp.sqrt(cp.maximum(0, discriminant[valid_mask]))
    
    s1 = (-b_valid - sqrt_d) / (2 * a_valid)
    s2 = (-b_valid + sqrt_d) / (2 * a_valid)
    
    intersection_on_segment = (s1 <= 1.0 + epsilon) & (s2 >= -epsilon)
    intersect[valid_mask] = intersection_on_segment
    
    fully_shielded_flags = intersect.all(axis=2)
    
    smoke_on_ground = smoke_center_batch[:, :, 2] < 0
    final_shield_flags = fully_shielded_flags & valid_time_mask & ~smoke_on_ground
    
    shield_durations = cp.sum(final_shield_flags, axis=1) * dt_fine
    
    return shield_durations

def fitness_function_batch(params_batch, target_samples_gpu):
    # (代码与原版相同)
    num_particles = params_batch.shape[0]
    
    v = params_batch[:, 1]
    delta_t2 = params_batch[:, 4]
    delta_t3 = params_batch[:, 6]
    
    valid_mask = (v >= 70.0 - epsilon) & (v <= 140.0 + epsilon) & \
                 (delta_t2 >= 1.0 - epsilon) & (delta_t3 >= 1.0 - epsilon) & \
                 (params_batch[:, [2,3,5,7]] >= -epsilon).all(axis=1)

    theta, v, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3 = params_batch.T
    
    t1_2 = t1_1 + delta_t2
    t1_3 = t1_2 + delta_t3
    
    bomb_params_batch = cp.stack([
        cp.vstack([theta, v, t1_1, t2_1]),
        cp.vstack([theta, v, t1_2, t2_2]),
        cp.vstack([theta, v, t1_3, t2_3])
    ], axis=1).reshape(-1, 4)
    
    all_durations = get_shield_durations_batch(bomb_params_batch, target_samples_gpu)
    
    durations_matrix = all_durations.reshape(num_particles, 3)
    
    # 近似的区间合并逻辑
    total_time_points = int(missile_arrival_time / dt_fine) + 1
    time_axis = cp.zeros((num_particles, total_time_points), dtype=bool)

    t1_all = cp.array([t1_1, t1_2, t1_3], dtype=CP_DTYPE).T
    t2_all = cp.array([t2_1, t2_2, t2_3], dtype=CP_DTYPE).T
    t_det_all = t1_all + t2_all
    
    for i in range(3):
        start_indices = (t_det_all[:, i] / dt_fine).astype(cp.int32)
        duration_indices = (durations_matrix[:, i] / dt_fine).astype(cp.int32)
        
        # 确保索引在范围内
        start_indices = cp.clip(start_indices, 0, total_time_points - 1)
        
        for p in range(num_particles):
            start = start_indices[p]
            end = start + duration_indices[p]
            end = cp.clip(end, 0, total_time_points) # 再次确保
            if end > start:
                time_axis[p, start:end] = True
    
    total_duration_batch = cp.sum(time_axis, axis=1) * dt_fine
    
    final_fitness = cp.where(valid_mask, total_duration_batch, 0.0)
    
    return final_fitness

# -------------------------- 4. PSO 优化器 (GPU) --------------------------------
class PSOOptimizer_GPU:
    # (代码与原版相同)
    def __init__(self, obj_func, bounds, num_particles=50, max_iter=120, batch_size=2):
        self.obj_func = obj_func
        self.bounds = cp.array(bounds, dtype=CP_DTYPE)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.dim = len(bounds)
        self.batch_size = batch_size

        self.pos = cp.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(num_particles, self.dim), dtype=CP_DTYPE
        )
        vel_range = self.bounds[:, 1] - self.bounds[:, 0]
        self.vel = 0.1 * cp.random.uniform(-vel_range, vel_range, size=(num_particles, self.dim), dtype=CP_DTYPE)

        print(f"Initializing particles fitness (batch on GPU, batch_size={self.batch_size})...")
        self.pbest_pos = self.pos.copy()
        self.pbest_fit = cp.zeros(num_particles, dtype=CP_DTYPE)
        for i in range(0, self.num_particles, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_particles)
            pos_batch = self.pos[i:end_idx]
            self.pbest_fit[i:end_idx] = self.obj_func(pos_batch)

        self.gbest_idx = cp.argmax(self.pbest_fit)
        self.gbest_pos = self.pbest_pos[self.gbest_idx].copy()
        self.gbest_fit = self.pbest_fit[self.gbest_idx]
        self.gbest_history = [self.gbest_fit.get()]

    def update(self):
        for iter_num in range(self.max_iter):
            w = 0.9 - 0.5 * (iter_num / self.max_iter)
            c1, c2 = 2.0, 2.0

            fit_values = cp.zeros(self.num_particles, dtype=CP_DTYPE)
            for i in range(0, self.num_particles, self.batch_size):
                end_idx = min(i + self.batch_size, self.num_particles)
                pos_batch = self.pos[i:end_idx]
                fit_values[i:end_idx] = self.obj_func(pos_batch)
            
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()

            improved_mask = fit_values > self.pbest_fit
            self.pbest_fit = cp.where(improved_mask, fit_values, self.pbest_fit)
            self.pbest_pos = cp.where(improved_mask[:, None], self.pos, self.pbest_pos)
            
            current_best_idx = cp.argmax(fit_values)
            if fit_values[current_best_idx] > self.gbest_fit:
                self.gbest_fit = fit_values[current_best_idx]
                self.gbest_pos = self.pos[current_best_idx].copy()
            
            r1 = cp.random.random((self.num_particles, self.dim), dtype=CP_DTYPE)
            r2 = cp.random.random((self.num_particles, self.dim), dtype=CP_DTYPE)
            
            self.vel = w * self.vel + \
                       c1 * r1 * (self.pbest_pos - self.pos) + \
                       c2 * r2 * (self.gbest_pos - self.pos)
            
            self.pos += self.vel
            self.pos = cp.clip(self.pos, self.bounds[:, 0], self.bounds[:, 1])

            self.gbest_history.append(self.gbest_fit.get())
            if (iter_num + 1) % 10 == 0:
                print(f"迭代 {iter_num+1:3d}/{self.max_iter} | 最优适应度(近似值): {self.gbest_fit:.4f}")

        return self.gbest_pos.get(), self.gbest_fit.get(), self.gbest_history

# -------------------------- 5. 新增：CPU 精确复核模块 --------------------------
def segment_sphere_intersect_cpu(M, P, C, r):
    MP = P - M
    MC = C - M
    a = np.dot(MP, MP)
    if a < 1e-12: return np.linalg.norm(MC) <= r + 1e-12
    b = -2 * np.dot(MP, MC)
    c = np.dot(MC, MC) - r**2
    discriminant = b**2 - 4*a*c
    if discriminant < -1e-12: return False
    discriminant = max(discriminant, 0)
    sqrt_d = np.sqrt(discriminant)
    s1 = (-b - sqrt_d) / (2*a)
    s2 = (-b + sqrt_d) / (2*a)
    return (s1 <= 1.0 + 1e-12) and (s2 >= -1e-12)

def get_shield_interval_cpu(bomb_params, target_samples):
    theta, v, t1, t2 = bomb_params
    uav_dir = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=DTYPE)
    drop_point = fy1_init_pos + v * t1 * uav_dir
    det_xy = drop_point[:2] + v * t2 * uav_dir[:2]
    det_z = drop_point[2] - 0.5 * g * t2**2
    if det_z < 0: return []
    det_point = np.array([det_xy[0], det_xy[1], det_z], dtype=DTYPE)

    t_det = t1 + t2
    t_smoke_start = t_det
    t_smoke_end = min(t_det + smoke_param["valid_time"], missile_arrival_time)
    if t_smoke_start >= t_smoke_end: return []

    t_list = np.arange(t_smoke_start, t_smoke_end + dt_fine, dt_fine, dtype=DTYPE)
    shield_intervals = []
    in_shield = False
    interval_start = 0

    for t in t_list:
        missile_pos = missile_param["init_pos"] + missile_param["speed"] * t * missile_param["dir"]
        sink_time = t - t_det
        smoke_center = det_point - np.array([0, 0, smoke_param["sink_speed"] * sink_time], dtype=DTYPE)
        if smoke_center[2] < 0:
            if in_shield: shield_intervals.append([interval_start, t]); in_shield = False
            continue

        fully_shielded = all(segment_sphere_intersect_cpu(missile_pos, p, smoke_center, smoke_param["r"]) for p in target_samples)

        if fully_shielded and not in_shield:
            interval_start = t; in_shield = True
        elif not fully_shielded and in_shield:
            shield_intervals.append([interval_start, t]); in_shield = False

    if in_shield: shield_intervals.append([interval_start, t_smoke_end])
    return shield_intervals

def merge_intervals_cpu(intervals):
    if not intervals: return 0.0, []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1e-12:
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            merged.append(current)
    total = sum([end - start for start, end in merged])
    return total, merged

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    print("生成目标采样点 (on CPU)...")
    target_samples_np = generate_target_samples(real_target)
    target_samples_gpu = cp.asarray(target_samples_np, dtype=CP_DTYPE)
    print(f"采样点数量: {len(target_samples_np)}, 已转移到GPU")

    bounds = [
        (0.0, 2*np.pi), (70.0, 140.0), (0.0, 60.0), (0.0, 20.0),
        (1.0, 30.0), (0.0, 20.0), (1.0, 30.0), (0.0, 20.0)
    ]

    print("\n启动粒子群优化 (Batch on GPU)...")
    pso = PSOOptimizer_GPU(
        obj_func=lambda p_batch: fitness_function_batch(p_batch, target_samples_gpu),
        bounds=bounds, num_particles=50, max_iter=120, batch_size=4
    )
    best_params, best_fitness_approx, gbest_history = pso.update()
    
    # --- 结果解析与精确复核 ---
    print("\n优化完成，正在使用CPU对最优解进行精确复核...")
    theta_opt, v_opt, t1_1_opt, t2_1_opt, delta_t2_opt, t2_2_opt, delta_t3_opt, t2_3_opt = best_params.astype(np.float64) # 提升精度
    t1_2_opt = t1_1_opt + delta_t2_opt
    t1_3_opt = t1_2_opt + delta_t3_opt

    bomb1_params = [theta_opt, v_opt, t1_1_opt, t2_1_opt]
    bomb2_params = [theta_opt, v_opt, t1_2_opt, t2_2_opt]
    bomb3_params = [theta_opt, v_opt, t1_3_opt, t2_3_opt]

    intervals1 = get_shield_interval_cpu(bomb1_params, target_samples_np)
    intervals2 = get_shield_interval_cpu(bomb2_params, target_samples_np)
    intervals3 = get_shield_interval_cpu(bomb3_params, target_samples_np)
    
    total_duration_exact, merged_intervals_exact = merge_intervals_cpu(intervals1 + intervals2 + intervals3)

    end_time = time.time()
    
    # --- 详细数据计算 ---
    uav_dir_opt = np.array([np.cos(theta_opt), np.sin(theta_opt), 0.0])
    drop1 = fy1_init_pos + v_opt * t1_1_opt * uav_dir_opt
    det1 = drop1 + np.array([v_opt * t2_1_opt * uav_dir_opt[0], v_opt * t2_1_opt * uav_dir_opt[1], -0.5 * g * t2_1_opt**2])
    drop2 = fy1_init_pos + v_opt * t1_2_opt * uav_dir_opt
    det2 = drop2 + np.array([v_opt * t2_2_opt * uav_dir_opt[0], v_opt * t2_2_opt * uav_dir_opt[1], -0.5 * g * t2_2_opt**2])
    drop3 = fy1_init_pos + v_opt * t1_3_opt * uav_dir_opt
    det3 = drop3 + np.array([v_opt * t2_3_opt * uav_dir_opt[0], v_opt * t2_3_opt * uav_dir_opt[1], -0.5 * g * t2_3_opt**2])
    
    # --- 结果输出到控制台 ---
    print("\n" + "="*80)
    print("【FY1三枚烟幕弹投放策略优化结果 (GPU加速+CPU精确复核)】")
    print(f"优化总耗时: {end_time - start_time:.2f} s")
    print(f"GPU优化得到的近似最优时长: {best_fitness_approx:.4f} s")
    print(f"CPU复核得到的精确最优时长: {total_duration_exact:.4f} s")
    print(f"无人机固定速度: {v_opt:.4f} m/s")
    print(f"无人机固定航向: {theta_opt:.4f} rad ({np.degrees(theta_opt):.2f}°)")
    print("="*80)
    
    # --- 保存结果到 Excel 文件 ---
    result_df = pd.DataFrame({
        "弹序号": ["第一枚", "第二枚", "第三枚"],
        "无人机航向(rad)": [theta_opt] * 3,
        "无人机速度(m/s)": [v_opt] * 3,
        "投放延迟(s)": [t1_1_opt, t1_2_opt, t1_3_opt],
        "起爆延迟(s)": [t2_1_opt, t2_2_opt, t2_3_opt],
        "投放点X(m)": [drop1[0], drop2[0], drop3[0]],
        "投放点Y(m)": [drop1[1], drop2[1], drop3[1]],
        "投放点Z(m)": [drop1[2], drop2[2], drop3[2]],
        "起爆点X(m)": [det1[0], det2[0], det3[0]],
        "起爆点Y(m)": [det1[1], det2[1], det3[1]],
        "起爆点Z(m)": [det1[2], det2[2], det3[2]],
    })
    result_df.to_excel("result1.xlsx", index=False, float_format="%.4f")
    print("核心结果已保存到 result1.xlsx")

    # --- 保存详细结果到 TXT 文件 ---
    with open("result1_details.txt", "w", encoding="utf-8") as f:
        f.write("="*30 + " 优化摘要 " + "="*30 + "\n")
        f.write(f"总耗时: {end_time - start_time:.2f} s\n")
        f.write(f"精确总遮蔽时长: {total_duration_exact:.6f} s\n")
        f.write(f"无人机速度: {v_opt:.4f} m/s\n")
        f.write(f"无人机航向: {theta_opt:.4f} rad ({np.degrees(theta_opt):.2f}°)\n\n")
        
        f.write("="*30 + " 最优参数 " + "="*30 + "\n")
        f.write(f"theta, v, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3\n")
        f.write(f"{np.array2string(best_params, formatter={'float_kind':lambda x: '%.4f' % x})}\n\n")

        f.write("="*30 + " 各弹遮蔽区间 " + "="*30 + "\n")
        f.write(f"第一枚: {intervals1}\n")
        f.write(f"第二枚: {intervals2}\n")
        f.write(f"第三枚: {intervals3}\n")
        f.write(f"合并后: {merged_intervals_exact}\n")

    print("详细参数与遮蔽区间已保存到 result1_details.txt")

    # --- 可视化：收敛曲线 ---
    plt.figure(figsize=(12, 6))
    plt.plot(gbest_history, marker='.', linestyle='-', label="GPU 近似最优时长")
    plt.axhline(y=total_duration_exact, color='r', linestyle='--', label=f'CPU 精确复核时长 ({total_duration_exact:.4f}s)')
    plt.xlabel("迭代次数")
    plt.ylabel("遮蔽时长(s)")
    plt.title("PSO优化收敛曲线 (GPU加速)")
    plt.grid(True)
    plt.legend()
    plt.savefig("convergence_curve_gpu.png")
    
    print("收敛曲线图已保存到 convergence_curve_gpu.png")
    plt.show()