#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3: 基于q2.py成功逻辑的递归三枚烟幕弹协同策略
思路: 环环相扣的递归优化，以每枚烟幕弹投放点为下一枚的新起点
Created on Wed Sep  5 2024
@author: Research Team Alpha  
@version: 3.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing as mp
import time
import warnings
import pandas as pd
from scipy.optimize import differential_evolution
import os

# 环境配置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========================= 核心物理参数 =========================
g = 9.8  # 重力加速度 (m/s²)
epsilon = 1e-15  # 数值计算保护阈值
dt_fine = 0.005   # 高精度时间步长
dt_coarse = 0.1   # 粗粒度时间步长
n_jobs = max(1, mp.cpu_count() - 1)

# 战场态势参数
fake_target = np.array([0.0, 0.0, 0.0])  # 假目标（导弹指向）
real_target = {
    "center": np.array([0.0, 200.0, 0.0]),  # 真目标中心
    "r": 7.0,      # 圆柱半径
    "h": 10.0      # 圆柱高度
}

# 无人机FY1初始参数
fy1_init_pos = np.array([17800.0, 0.0, 1800.0])  # 初始位置

# 烟幕参数
smoke_param = {
    "r": 10.0,        # 有效遮蔽半径
    "sink_speed": 3.0,         # 下沉速度
    "valid_time": 20.0     # 单枚有效时长
}

# 导弹M1参数
missile_param = {
    "init_pos": np.array([20000.0, 0.0, 2000.0]),  # 初始位置
    "speed": 300.0         # 飞行速度
}

# 预计算导弹轨迹参数
missile_trajectory = fake_target - missile_param["init_pos"]
missile_distance = np.linalg.norm(missile_trajectory)
missile_dir = missile_trajectory / missile_distance
missile_arrival_time = missile_distance / missile_param["speed"]

# ========================= 从q2.py继承的核心算法 =========================
class AdvancedSpatialSamplingEngine:
    """高级空间几何采样引擎（继承自q2.py）"""
    
    @staticmethod
    def generate_comprehensive_threat_volume_discretization(threat_config, density_factor=1.0):
        """生成威胁体积的全方位离散化采样网格"""
        sampling_vertices = []
        center = threat_config["center"]
        radius = threat_config["r"]
        height = threat_config["h"]
        center_xy = center[:2]
        z_bounds = [center[2], center[2] + height]
        
        # 根据密度因子调整采样点数量
        base_theta_points = max(10, int(60 * density_factor))
        base_height_points = max(5, int(20 * density_factor))
        
        # Phase 1: 顶底面密集圆周采样
        azimuthal_angles = np.linspace(0, 2*np.pi, base_theta_points, endpoint=False)
        for z_level in z_bounds:
            for angle in azimuthal_angles:
                x = center_xy[0] + radius * np.cos(angle)
                y = center_xy[1] + radius * np.sin(angle)
                sampling_vertices.append([x, y, z_level])
        
        # Phase 2: 侧面柱状表面采样
        height_layers = np.linspace(z_bounds[0], z_bounds[1], base_height_points)
        for z in height_layers:
            for angle in azimuthal_angles:
                x = center_xy[0] + radius * np.cos(angle)
                y = center_xy[1] + radius * np.sin(angle)
                sampling_vertices.append([x, y, z])
        
        # Phase 3: 内部体积网格化
        if density_factor > 0.5:  # 高密度时才添加内部点
            radial_layers = np.linspace(0, radius, max(3, int(6 * density_factor)))
            internal_heights = np.linspace(z_bounds[0], z_bounds[1], max(5, int(15 * density_factor)))
            internal_angles = np.linspace(0, 2*np.pi, max(8, int(16 * density_factor)), endpoint=False)
            
            for z in internal_heights:
                for r_dist in radial_layers:
                    for angle in internal_angles:
                        x = center_xy[0] + r_dist * np.cos(angle)
                        y = center_xy[1] + r_dist * np.sin(angle)
                        sampling_vertices.append([x, y, z])
        
        return np.unique(np.array(sampling_vertices, dtype=np.float64), axis=0)

class PrecisionNumericalKernel:
    """高精度数值计算内核（继承自q2.py）"""
    
    @staticmethod
    def analyze_line_segment_sphere_intersection(point_a, point_b, sphere_center, sphere_radius):
        """分析线段-球体相交几何关系"""
        direction_vec = point_b - point_a
        center_offset = sphere_center - point_a
        a_coeff = np.dot(direction_vec, direction_vec)
        
        # 零长度线段处理
        if a_coeff < epsilon:
            distance = np.linalg.norm(center_offset)
            return 1.0 if distance <= sphere_radius + epsilon else 0.0
        
        b_coeff = -2 * np.dot(direction_vec, center_offset)
        c_coeff = np.dot(center_offset, center_offset) - sphere_radius**2
        discriminant = b_coeff**2 - 4*a_coeff*c_coeff
        
        # 判别式分析
        if discriminant < -epsilon:
            return 0.0
        if discriminant < 0:
            discriminant = 0.0
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b_coeff - sqrt_disc) / (2*a_coeff)
        t2 = (-b_coeff + sqrt_disc) / (2*a_coeff)
        
        # 相交区间计算
        intersection_start = max(0.0, min(t1, t2))
        intersection_end = min(1.0, max(t1, t2))
        
        return max(0.0, intersection_end - intersection_start)
    
    @staticmethod
    def evaluate_target_occlusion_status(missile_pos, smoke_center, smoke_radius, target_vertices):
        """评估目标的完全遮蔽状态"""
        for vertex in target_vertices:
            intersection_ratio = PrecisionNumericalKernel.analyze_line_segment_sphere_intersection(
                missile_pos, vertex, smoke_center, smoke_radius
            )
            if intersection_ratio < epsilon:
                return False
        return True

class AdaptiveTemporalManager:
    """自适应时间序列管理（继承自q2.py）"""
    
    @staticmethod
    def construct_adaptive_time_sequence(start_time, end_time, critical_time=None):
        """构建自适应分辨率时间序列"""
        if critical_time is None:
            return np.arange(start_time, end_time + dt_coarse, dt_coarse)
        
        # 关键时刻附近的高分辨率窗口
        high_res_start = max(start_time, critical_time - 1.0)
        high_res_end = min(end_time, critical_time + 1.0)
        
        time_sequence = []
        
        # 前置粗粒度时间段
        if start_time < high_res_start:
            time_sequence.extend(np.arange(start_time, high_res_start, dt_coarse))
        
        # 中央高分辨率时间段
        time_sequence.extend(np.arange(high_res_start, high_res_end + dt_fine, dt_fine))
        
        # 后置粗粒度时间段
        if high_res_end < end_time:
            time_sequence.extend(np.arange(high_res_end, end_time + dt_coarse, dt_coarse))
        
        return np.unique(time_sequence)

# ========================= 单枚烟幕弹效能评估函数 =========================
def evaluate_single_smoke_effectiveness(params, start_pos, target_mesh):
    """
    评估单枚烟幕弹的遮蔽效能
    params: [theta, v, t1, t2] - 飞行方向、速度、投放延迟、起爆延迟
    start_pos: 起始位置（对第一枚是无人机初始位置，对后续是前一枚投放点）
    target_mesh: 目标采样网格
    """
    theta, v, t1, t2 = params
    
    # 参数约束验证
    if not (70.0 <= v <= 140.0):
        return 0.0
    if t1 < 0 or t2 < 0:
        return 0.0
    
    # Step 1: 计算投放点
    flight_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
    drop_point = start_pos + v * t1 * flight_dir
    
    # Step 2: 计算起爆点
    det_xy = drop_point[:2] + v * t2 * flight_dir[:2]
    det_z = drop_point[2] - 0.5 * g * t2**2
    
    if det_z < 5.0:  # 起爆点过低
        return 0.0
    
    det_point = np.array([det_xy[0], det_xy[1], det_z])
    
    # Step 3: 时间窗口分析
    abs_det_time = t1 + t2
    smoke_end_time = abs_det_time + smoke_param["valid_time"]
    analysis_end_time = min(smoke_end_time, missile_arrival_time)
    
    if abs_det_time >= analysis_end_time:
        return 0.0
    
    # Step 4: 关键时刻识别（导弹接近目标的时间）
    target_vec = real_target["center"] - missile_param["init_pos"]
    critical_distance = np.dot(target_vec, missile_dir)
    critical_time = critical_distance / missile_param["speed"]
    
    time_sequence = AdaptiveTemporalManager.construct_adaptive_time_sequence(
        abs_det_time, analysis_end_time, critical_time
    )
    
    # Step 5: 累积遮蔽效能计算
    total_occlusion_time = 0.0
    prev_time = None
    
    for curr_time in time_sequence:
        if prev_time is not None:
            time_increment = curr_time - prev_time
            
            # 导弹当前位置
            missile_pos = missile_param["init_pos"] + missile_param["speed"] * curr_time * missile_dir
            
            # 烟幕当前状态
            sink_duration = curr_time - abs_det_time
            current_smoke_z = det_point[2] - smoke_param["sink_speed"] * sink_duration
            
            if current_smoke_z < 2.0:  # 烟幕落地
                prev_time = curr_time
                continue
            
            current_smoke_center = np.array([det_point[0], det_point[1], current_smoke_z])
            
            # 遮蔽效能评估
            is_occluded = PrecisionNumericalKernel.evaluate_target_occlusion_status(
                missile_pos, current_smoke_center, smoke_param["r"], target_mesh
            )
            
            if is_occluded:
                total_occlusion_time += time_increment
        
        prev_time = curr_time
    
    return total_occlusion_time

# ========================= 递归三枚烟幕弹协同优化系统 =========================
class RecursiveTripleSmokeOptimizer:
    """递归三枚烟幕弹协同优化系统"""
    
    def __init__(self, target_mesh, min_interval=1.0):
        self.target_mesh = target_mesh
        self.min_interval = min_interval  # 最小投放间隔
        
        # 优化边界
        self.param_bounds = [
            (0.0, 2*np.pi),    # theta: 飞行方向
            (70.0, 140.0),     # v: 飞行速度
            (0.0, 80.0),       # t1: 投放延迟
            (0.0, 25.0)        # t2: 起爆延迟
        ]
    
    def optimize_first_smoke(self):
        """第一次优化：优化第一枚烟幕弹"""
        print("第一次优化：优化第一枚烟幕弹...")
        
        def objective_func(params):
            return -evaluate_single_smoke_effectiveness(params, fy1_init_pos, self.target_mesh)
        
        result = differential_evolution(
            objective_func, 
            self.param_bounds,
            maxiter=100,
            popsize=30,
            seed=42,
            polish=True
        )
        
        first_params = result.x
        first_effectiveness = -result.fun
        
        # 计算第一枚投放点
        theta1, v1, t1_1, t2_1 = first_params
        flight_dir1 = np.array([np.cos(theta1), np.sin(theta1), 0.0])
        drop_point1 = fy1_init_pos + v1 * t1_1 * flight_dir1
        
        print(f"第一枚最优参数: theta={np.degrees(theta1):.2f}°, v={v1:.2f}m/s, t1={t1_1:.2f}s, t2={t2_1:.2f}s")
        print(f"第一枚投放点: {drop_point1}")
        print(f"第一枚遮蔽效能: {first_effectiveness:.6f}s")
        
        return first_params, drop_point1, first_effectiveness
    
    def optimize_second_smoke(self, first_params, drop_point1):
        """递归优化：以第一枚投放点为起点优化第二枚"""
        print("\n递归优化：以第一枚投放点为起点优化第二枚...")
        
        theta1, v1, t1_1, t2_1 = first_params
        
        # 第二枚的投放时间必须在第一枚之后至少min_interval秒
        min_t1_2 = t1_1 + self.min_interval
        
        # 调整第二枚的时间边界
        second_bounds = [
            (0.0, 2*np.pi),           # theta2: 可以是任意方向
            (70.0, 140.0),            # v2: 可以是任意速度
            (min_t1_2, 80.0),         # t1_2: 必须晚于第一枚
            (0.0, 25.0)               # t2_2: 起爆延迟
        ]
        
        def objective_func(params):
            # 从第一枚投放点开始计算第二枚的效能
            theta2, v2, t1_2, t2_2 = params
            
            # 计算从投放点1到投放点2的相对参数
            relative_t1 = t1_2 - t1_1  # 相对于第一枚投放的时间差
            relative_params = [theta2, v2, relative_t1, t2_2]
            
            return -evaluate_single_smoke_effectiveness(relative_params, drop_point1, self.target_mesh)
        
        result = differential_evolution(
            objective_func,
            second_bounds,
            maxiter=100,
            popsize=30,
            seed=43,
            polish=True
        )
        
        second_params = result.x
        second_effectiveness = -result.fun
        
        # 计算第二枚投放点
        theta2, v2, t1_2, t2_2 = second_params
        flight_dir2 = np.array([np.cos(theta2), np.sin(theta2), 0.0])
        relative_t1 = t1_2 - t1_1
        drop_point2 = drop_point1 + v2 * relative_t1 * flight_dir2
        
        print(f"第二枚最优参数: theta={np.degrees(theta2):.2f}°, v={v2:.2f}m/s, t1={t1_2:.2f}s, t2={t2_2:.2f}s")
        print(f"第二枚投放点: {drop_point2}")
        print(f"第二枚遮蔽效能: {second_effectiveness:.6f}s")
        
        return second_params, drop_point2, second_effectiveness
    
    def optimize_third_smoke(self, first_params, second_params, drop_point2):
        """递归优化：以第二枚投放点为起点优化第三枚"""
        print("\n递归优化：以第二枚投放点为起点优化第三枚...")
        
        theta1, v1, t1_1, t2_1 = first_params
        theta2, v2, t1_2, t2_2 = second_params
        
        # 第三枚的投放时间必须在第二枚之后至少min_interval秒
        min_t1_3 = t1_2 + self.min_interval
        
        # 调整第三枚的时间边界
        third_bounds = [
            (0.0, 2*np.pi),           # theta3: 可以是任意方向
            (70.0, 140.0),            # v3: 可以是任意速度
            (min_t1_3, 80.0),         # t1_3: 必须晚于第二枚
            (0.0, 25.0)               # t2_3: 起爆延迟
        ]
        
        def objective_func(params):
            # 从第二枚投放点开始计算第三枚的效能
            theta3, v3, t1_3, t2_3 = params
            
            # 计算从投放点2到投放点3的相对参数
            relative_t1 = t1_3 - t1_2  # 相对于第二枚投放的时间差
            relative_params = [theta3, v3, relative_t1, t2_3]
            
            return -evaluate_single_smoke_effectiveness(relative_params, drop_point2, self.target_mesh)
        
        result = differential_evolution(
            objective_func,
            third_bounds,
            maxiter=100,
            popsize=30,
            seed=44,
            polish=True
        )
        
        third_params = result.x
        third_effectiveness = -result.fun
        
        # 计算第三枚投放点
        theta3, v3, t1_3, t2_3 = third_params
        flight_dir3 = np.array([np.cos(theta3), np.sin(theta3), 0.0])
        relative_t1 = t1_3 - t1_2
        drop_point3 = drop_point2 + v3 * relative_t1 * flight_dir3
        
        print(f"第三枚最优参数: theta={np.degrees(theta3):.2f}°, v={v3:.2f}m/s, t1={t1_3:.2f}s, t2={t2_3:.2f}s")
        print(f"第三枚投放点: {drop_point3}")
        print(f"第三枚遮蔽效能: {third_effectiveness:.6f}s")
        
        return third_params, drop_point3, third_effectiveness
    
    def calculate_combined_effectiveness(self, first_params, second_params, third_params):
        """计算三枚烟幕弹的联合遮蔽效能"""
        print("\n计算三枚烟幕弹的联合遮蔽效能...")
        
        # 重新计算每枚烟幕弹的遮蔽时间间隔
        theta1, v1, t1_1, t2_1 = first_params
        theta2, v2, t1_2, t2_2 = second_params  
        theta3, v3, t1_3, t2_3 = third_params
        
        # 计算投放点和起爆点
        flight_dir1 = np.array([np.cos(theta1), np.sin(theta1), 0.0])
        drop_point1 = fy1_init_pos + v1 * t1_1 * flight_dir1
        
        flight_dir2 = np.array([np.cos(theta2), np.sin(theta2), 0.0])
        relative_t1_2 = t1_2 - t1_1
        drop_point2 = drop_point1 + v2 * relative_t1_2 * flight_dir2
        
        flight_dir3 = np.array([np.cos(theta3), np.sin(theta3), 0.0])
        relative_t1_3 = t1_3 - t1_2
        drop_point3 = drop_point2 + v3 * relative_t1_3 * flight_dir3
        
        # 计算起爆点
        det_point1 = self._calculate_detonation_point(drop_point1, flight_dir1, v1, t2_1)
        det_point2 = self._calculate_detonation_point(drop_point2, flight_dir2, v2, t2_2)  
        det_point3 = self._calculate_detonation_point(drop_point3, flight_dir3, v3, t2_3)
        
        # 计算起爆时间
        det_time1 = t1_1 + t2_1
        det_time2 = t1_2 + t2_2
        det_time3 = t1_3 + t2_3
        
        # 收集所有遮蔽时间间隔
        all_intervals = []
        
        # 第一枚烟幕弹的遮蔽间隔
        intervals1 = self._get_smoke_intervals(det_point1, det_time1)
        all_intervals.extend(intervals1)
        
        # 第二枚烟幕弹的遮蔽间隔
        intervals2 = self._get_smoke_intervals(det_point2, det_time2)
        all_intervals.extend(intervals2)
        
        # 第三枚烟幕弹的遮蔽间隔
        intervals3 = self._get_smoke_intervals(det_point3, det_time3)
        all_intervals.extend(intervals3)
        
        # 合并重叠间隔并计算总时长
        merged_intervals = self._merge_intervals(all_intervals)
        total_time = sum([end - start for start, end in merged_intervals])
        
        print(f"第一枚遮蔽间隔数: {len(intervals1)}")
        print(f"第二枚遮蔽间隔数: {len(intervals2)}")
        print(f"第三枚遮蔽间隔数: {len(intervals3)}")
        print(f"合并后间隔数: {len(merged_intervals)}")
        print(f"三枚烟幕弹联合遮蔽总时长: {total_time:.6f}s")
        
        return total_time, merged_intervals, {
            'drop_points': [drop_point1, drop_point2, drop_point3],
            'det_points': [det_point1, det_point2, det_point3],
            'det_times': [det_time1, det_time2, det_time3],
            'intervals': [intervals1, intervals2, intervals3]
        }
    
    def _calculate_detonation_point(self, drop_point, flight_dir, velocity, t2):
        """计算起爆点"""
        det_xy = drop_point[:2] + velocity * t2 * flight_dir[:2]
        det_z = drop_point[2] - 0.5 * g * t2**2
        return np.array([det_xy[0], det_xy[1], det_z])
    
    def _get_smoke_intervals(self, det_point, det_time):
        """获取单枚烟幕弹的遮蔽时间间隔"""
        smoke_end_time = det_time + smoke_param["valid_time"]
        analysis_end_time = min(smoke_end_time, missile_arrival_time)
        
        if det_time >= analysis_end_time:
            return []
        
        # 关键时刻识别
        target_vec = real_target["center"] - missile_param["init_pos"]
        critical_distance = np.dot(target_vec, missile_dir)
        critical_time = critical_distance / missile_param["speed"]
        
        time_sequence = AdaptiveTemporalManager.construct_adaptive_time_sequence(
            det_time, analysis_end_time, critical_time
        )
        
        intervals = []
        in_occlusion = False
        interval_start = 0
        
        for curr_time in time_sequence:
            # 导弹当前位置
            missile_pos = missile_param["init_pos"] + missile_param["speed"] * curr_time * missile_dir
            
            # 烟幕当前状态
            sink_duration = curr_time - det_time
            current_smoke_z = det_point[2] - smoke_param["sink_speed"] * sink_duration
            
            if current_smoke_z < 2.0:  # 烟幕落地
                if in_occlusion:
                    intervals.append([interval_start, curr_time])
                    in_occlusion = False
                continue
            
            current_smoke_center = np.array([det_point[0], det_point[1], current_smoke_z])
            
            # 遮蔽效能评估
            is_occluded = PrecisionNumericalKernel.evaluate_target_occlusion_status(
                missile_pos, current_smoke_center, smoke_param["r"], self.target_mesh
            )
            
            # 更新遮蔽状态
            if is_occluded and not in_occlusion:
                interval_start = curr_time
                in_occlusion = True
            elif not is_occluded and in_occlusion:
                intervals.append([interval_start, curr_time])
                in_occlusion = False
        
        # 处理最后一个未结束的区间
        if in_occlusion:
            intervals.append([interval_start, analysis_end_time])
        
        return intervals
    
    def _merge_intervals(self, intervals):
        """合并重叠的时间间隔"""
        if not intervals:
            return []
        
        # 按开始时间排序
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]
        
        for current in sorted_intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1] + epsilon:  # 重叠或相邻
                merged[-1] = [last[0], max(last[1], current[1])]
            else:
                merged.append(current)
        
        return merged

# ========================= 二次全局优化 =========================
def secondary_global_optimization(target_mesh):
    """
    二次优化：优化第一枚烟幕弹的投放时间、无人机速度和飞行方向
    """
    print("\n执行二次全局优化...")
    
    # 8参数优化：第一枚的theta,v,t1,t2，第二第三枚的时间间隔和起爆延迟
    bounds = [
        (0.0, 2*np.pi),    # theta1: 第一枚飞行方向
        (70.0, 140.0),     # v1: 第一枚飞行速度  
        (0.0, 60.0),       # t1_1: 第一枚投放延迟
        (0.0, 20.0),       # t2_1: 第一枚起爆延迟
        (1.0, 30.0),       # delta_t2: 第一枚到第二枚的时间间隔
        (0.0, 20.0),       # t2_2: 第二枚起爆延迟
        (1.0, 30.0),       # delta_t3: 第二枚到第三枚的时间间隔
        (0.0, 20.0)        # t2_3: 第三枚起爆延迟
    ]
    
    def combined_objective(params):
        """组合目标函数：三枚烟幕弹的总遮蔽时长"""
        theta1, v1, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3 = params
        
        # 约束检查
        if not (70.0 <= v1 <= 140.0):
            return 0.0
        if delta_t2 < 1.0 or delta_t3 < 1.0:  # 最小间隔约束
            return 0.0
        if any(x < 0 for x in [t1_1, t2_1, t2_2, t2_3]):
            return 0.0
        
        # 计算三枚烟幕弹的时间参数
        t1_2 = t1_1 + delta_t2
        t1_3 = t1_2 + delta_t3
        
        # 假设第二第三枚延续第一枚的飞行方向和速度（简化策略）
        # 这里可以进一步优化为独立的方向和速度
        
        first_params = [theta1, v1, t1_1, t2_1]
        second_params = [theta1, v1, t1_2, t2_2]  # 沿用相同方向速度
        third_params = [theta1, v1, t1_3, t2_3]   # 沿用相同方向速度
        
        # 计算各枚烟幕弹的遮蔽间隔
        all_intervals = []
        
        # 第一枚（从无人机初始位置开始）
        eff1 = evaluate_single_smoke_effectiveness(first_params, fy1_init_pos, target_mesh)
        if eff1 > 0:
            # 计算实际的时间间隔（这里简化为单个连续区间）
            det_time1 = t1_1 + t2_1
            all_intervals.append([det_time1, det_time1 + min(eff1, smoke_param["valid_time"])])
        
        # 第二枚（从第一枚投放点开始）
        flight_dir1 = np.array([np.cos(theta1), np.sin(theta1), 0.0])
        drop_point1 = fy1_init_pos + v1 * t1_1 * flight_dir1
        relative_params2 = [theta1, v1, delta_t2, t2_2]
        eff2 = evaluate_single_smoke_effectiveness(relative_params2, drop_point1, target_mesh)
        if eff2 > 0:
            det_time2 = t1_2 + t2_2
            all_intervals.append([det_time2, det_time2 + min(eff2, smoke_param["valid_time"])])
        
        # 第三枚（从第二枚投放点开始）
        drop_point2 = drop_point1 + v1 * delta_t2 * flight_dir1  # 沿用相同方向
        relative_params3 = [theta1, v1, delta_t3, t2_3]
        eff3 = evaluate_single_smoke_effectiveness(relative_params3, drop_point2, target_mesh)
        if eff3 > 0:
            det_time3 = t1_3 + t2_3
            all_intervals.append([det_time3, det_time3 + min(eff3, smoke_param["valid_time"])])
        
        # 合并间隔并计算总时长
        if not all_intervals:
            return 0.0
        
        sorted_intervals = sorted(all_intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]
        
        for current in sorted_intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1] + epsilon:
                merged[-1] = [last[0], max(last[1], current[1])]
            else:
                merged.append(current)
        
        total_time = sum([end - start for start, end in merged])
        return total_time
    
    # 执行差分进化优化
    result = differential_evolution(
        lambda params: -combined_objective(params),  # 最大化目标函数
        bounds,
        maxiter=150,
        popsize=40,
        seed=45,
        polish=True,
        workers=1  # 避免并行计算冲突
    )
    
    optimal_params = result.x
    optimal_effectiveness = -result.fun
    
    return optimal_params, optimal_effectiveness

# ========================= 主程序执行 =========================
if __name__ == "__main__":
    start_time = time.time()
    
    print("=" * 80)
    print("问题3: 基于q2.py成功逻辑的三枚烟幕弹协同递归优化")
    print("目标: 基于4.7秒成功经验，实现环环相扣的递归策略")
    print("=" * 80)
    
    # Step 1: 生成目标采样网格（使用适中密度以平衡计算效率）
    print("生成目标采样网格...")
    target_mesh = AdvancedSpatialSamplingEngine.generate_comprehensive_threat_volume_discretization(
        real_target, density_factor=0.8  # 适中密度
    )
    print(f"目标网格节点数量: {len(target_mesh)}")
    
    # Step 2: 递归三步优化
    optimizer = RecursiveTripleSmokeOptimizer(target_mesh)
    
    # 第一次优化：第一枚烟幕弹
    first_params, drop_point1, first_eff = optimizer.optimize_first_smoke()
    
    # 递归优化：第二枚烟幕弹
    second_params, drop_point2, second_eff = optimizer.optimize_second_smoke(first_params, drop_point1)
    
    # 递归优化：第三枚烟幕弹
    third_params, drop_point3, third_eff = optimizer.optimize_third_smoke(first_params, second_params, drop_point2)
    
    # Step 3: 计算联合效能
    combined_time, merged_intervals, detailed_info = optimizer.calculate_combined_effectiveness(
        first_params, second_params, third_params
    )
    
    # Step 4: 二次全局优化
    print("\n" + "="*60)
    print("执行二次全局优化以提升整体策略...")
    secondary_params, secondary_effectiveness = secondary_global_optimization(target_mesh)
    
    print(f"递归策略总遮蔽时长: {combined_time:.6f}s")
    print(f"二次优化总遮蔽时长: {secondary_effectiveness:.6f}s")
    
    # 选择更好的策略
    if secondary_effectiveness > combined_time:
        print("二次优化策略更优！")
        final_effectiveness = secondary_effectiveness
        final_strategy = "二次优化策略"
        
        # 解析二次优化参数
        theta1, v1, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3 = secondary_params
        t1_2 = t1_1 + delta_t2
        t1_3 = t1_2 + delta_t3
        
        final_params = {
            'first': [theta1, v1, t1_1, t2_1],
            'second': [theta1, v1, t1_2, t2_2],
            'third': [theta1, v1, t1_3, t2_3]
        }
    else:
        print("递归策略更优！")
        final_effectiveness = combined_time
        final_strategy = "递归策略"
        
        final_params = {
            'first': first_params,
            'second': second_params,
            'third': third_params
        }
    
    # Step 5: 结果保存
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("【最终优化结果】")
    print(f"优化策略: {final_strategy}")
    print(f"三枚烟幕弹总遮蔽时间: {final_effectiveness:.6f}秒")
    print(f"计算耗时: {elapsed_time:.2f}秒")
    
    # 解析最终参数
    first_final = final_params['first']
    second_final = final_params['second']
    third_final = final_params['third']
    
    print(f"\n第一枚: θ={np.degrees(first_final[0]):.2f}°, v={first_final[1]:.2f}m/s, t1={first_final[2]:.2f}s, t2={first_final[3]:.2f}s")
    print(f"第二枚: θ={np.degrees(second_final[0]):.2f}°, v={second_final[1]:.2f}m/s, t1={second_final[2]:.2f}s, t2={second_final[3]:.2f}s")
    print(f"第三枚: θ={np.degrees(third_final[0]):.2f}°, v={third_final[1]:.2f}m/s, t1={third_final[2]:.2f}s, t2={third_final[3]:.2f}s")
    print("=" * 80)
    
    # 保存到Excel
    try:
        results_df = pd.DataFrame({
            '烟幕弹序号': ['第一枚', '第二枚', '第三枚'],
            '飞行方向(度)': [np.degrees(final_params['first'][0]), 
                           np.degrees(final_params['second'][0]), 
                           np.degrees(final_params['third'][0])],
            '飞行速度(m/s)': [final_params['first'][1], 
                            final_params['second'][1], 
                            final_params['third'][1]],
            '投放延迟(s)': [final_params['first'][2], 
                          final_params['second'][2], 
                          final_params['third'][2]],
            '起爆延迟(s)': [final_params['first'][3], 
                          final_params['second'][3], 
                          final_params['third'][3]],
            '总遮蔽时长(s)': [final_effectiveness] * 3,
            '优化策略': [final_strategy] * 3
        })
        
        results_df.to_excel('result1.xlsx', index=False)
        print("结果已保存至 result1.xlsx")
        
        # 创建answer3文件夹并保存详细报告
        os.makedirs('answer3', exist_ok=True)
        with open('answer3/递归优化策略详细报告.txt', 'w', encoding='utf-8') as f:
            f.write("问题3：三枚烟幕弹协同递归优化策略报告\n")
            f.write("="*50 + "\n\n")
            f.write(f"优化策略: {final_strategy}\n")
            f.write(f"总遮蔽时长: {final_effectiveness:.6f}秒\n")
            f.write(f"计算耗时: {elapsed_time:.2f}秒\n\n")
            f.write("详细参数:\n")
            f.write(f"第一枚: θ={np.degrees(first_final[0]):.2f}°, v={first_final[1]:.2f}m/s, t1={first_final[2]:.2f}s, t2={first_final[3]:.2f}s\n")
            f.write(f"第二枚: θ={np.degrees(second_final[0]):.2f}°, v={second_final[1]:.2f}m/s, t1={second_final[2]:.2f}s, t2={second_final[3]:.2f}s\n")
            f.write(f"第三枚: θ={np.degrees(third_final[0]):.2f}°, v={third_final[1]:.2f}m/s, t1={third_final[2]:.2f}s, t2={third_final[3]:.2f}s\n")
        
        print("详细报告已保存至 answer3/递归优化策略详细报告.txt")
        
    except Exception as e:
        print(f"保存结果时出错: {e}")
    
    print("\n" + "=" * 80)
    print("递归优化完成！基于q2.py的4.7秒成功经验")
    print("实现了环环相扣的三枚烟幕弹协同策略")
    print("所有结果已保存至 result1.xlsx 和 answer3 文件夹")
    print("=" * 80)
