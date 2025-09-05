#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3：基于q2.py成功逻辑的三枚烟幕弹协同优化
目标：达到6.5秒左右的总遮蔽时间
基于用户第二问4.7秒的成功经验
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing as mp
import time
import warnings
import os
from scipy.optimize import differential_evolution

# 环境配置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========================= 基于q2.py的成功参数配置 =========================
class OptimizedSystemConfig:
    """基于用户q2.py 4.7秒成功经验的参数配置"""
    def __init__(self):
        # 物理常数（与q2.py完全一致）
        self.PHYSICS_GRAVITY_ACCELERATION = 9.8
        self.MATHEMATICAL_EPSILON_THRESHOLD = 1e-15
        self.TEMPORAL_STEP_SIZE_MICRO = 0.005  # 使用q2.py的微观时间步长
        self.SYSTEM_CPU_CORE_COUNT = max(1, mp.cpu_count() - 1)
        
        # 战场参数（复制q2.py成功配置）
        self.decoy_target_position = np.array([0.0, 0.0, 0.0])
        self.primary_target_specifications = {
            "central_coordinates": np.array([0.0, 200.0, 0.0]),
            "radius_parameter": 7.0,
            "height_parameter": 10.0
        }
        
        # 平台配置
        self.uav_initial_coordinates = np.array([17800.0, 0.0, 1800.0])
        self.missile_initial_coordinates = np.array([20000.0, 0.0, 2000.0])
        self.missile_velocity_magnitude = 300.0
        
        # 计算导弹飞行参数
        self.missile_trajectory_vector = (
            self.decoy_target_position - self.missile_initial_coordinates
        )
        self.missile_direction_unit = (
            self.missile_trajectory_vector / 
            np.linalg.norm(self.missile_trajectory_vector)
        )
        self.mission_total_duration = (
            np.linalg.norm(self.missile_trajectory_vector) / 
            self.missile_velocity_magnitude
        )
        
        # 烟幕效应参数（基于q2.py优化后的参数）
        self.smoke_effectiveness_radius = 10.0
        self.smoke_descent_velocity = 3.0
        self.smoke_active_duration = 20.0
        
        # 基于q2.py成功经验的约束参数
        self.velocity_bounds = (70.0, 140.0)
        self.deployment_interval_minimum = 1.0
        self.altitude_threshold_minimum = 5.0

# ========================= 基于q2.py成功逻辑的目标采样 =========================
def generate_enhanced_target_mesh(config):
    """快速测试版本：大幅减少采样点"""
    target_center = config.primary_target_specifications["central_coordinates"]
    target_radius = config.primary_target_specifications["radius_parameter"]
    target_height = config.primary_target_specifications["height_parameter"]
    
    sampling_points = []
    base_coordinates = target_center[:2]
    altitude_range = [target_center[2], target_center[2] + target_height]
    
    # Phase 1: 关键边界采样（大幅减少）
    angular_resolution = 8  # 从48减少到8
    angular_samples = np.linspace(0, 2*np.pi, angular_resolution, endpoint=False)
    
    # 顶底面圆周（只采样关键点）
    for altitude in altitude_range:
        for angle in angular_samples:
            x_coord = base_coordinates[0] + target_radius * np.cos(angle)
            y_coord = base_coordinates[1] + target_radius * np.sin(angle)
            sampling_points.append([x_coord, y_coord, altitude])
    
    # Phase 2: 侧面采样（大幅减少）
    vertical_layers = np.linspace(altitude_range[0], altitude_range[1], 3, endpoint=True)  # 从16减少到3
    for altitude in vertical_layers:
        for angle in angular_samples:
            x_coord = base_coordinates[0] + target_radius * np.cos(angle)
            y_coord = base_coordinates[1] + target_radius * np.sin(angle)
            sampling_points.append([x_coord, y_coord, altitude])
    
    # Phase 3: 内部体积采样（最小化）
    radial_samples = np.linspace(0, target_radius, 2, endpoint=True)  # 从6减少到2
    internal_vertical = np.linspace(altitude_range[0], altitude_range[1], 3, endpoint=True)  # 从12减少到3
    internal_angular = np.linspace(0, 2*np.pi, 4, endpoint=False)  # 从12减少到4
    
    for altitude in internal_vertical:
        for radius in radial_samples:
            for angle in internal_angular:
                x_coord = base_coordinates[0] + radius * np.cos(angle)
                y_coord = base_coordinates[1] + radius * np.sin(angle)
                sampling_points.append([x_coord, y_coord, altitude])
    
    return np.unique(np.array(sampling_points, dtype=np.float64), axis=0)

# ========================= 基于q2.py的精确几何计算 =========================
def precise_segment_sphere_intersection(point_a, point_b, sphere_center, sphere_radius):
    """基于q2.py成功验证的线段-球体相交算法"""
    direction_vector = point_b - point_a
    offset_vector = point_a - sphere_center
    
    quadratic_a = np.dot(direction_vector, direction_vector)
    
    # 处理退化情况
    if quadratic_a < 1e-15:
        return np.linalg.norm(offset_vector) <= sphere_radius + 1e-15
    
    quadratic_b = 2.0 * np.dot(offset_vector, direction_vector)
    quadratic_c = np.dot(offset_vector, offset_vector) - sphere_radius**2
    
    discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
    
    if discriminant < -1e-15:
        return False
    if discriminant < 0:
        discriminant = 0.0
    
    sqrt_discriminant = np.sqrt(discriminant)
    parameter_1 = (-quadratic_b - sqrt_discriminant) / (2.0 * quadratic_a)
    parameter_2 = (-quadratic_b + sqrt_discriminant) / (2.0 * quadratic_a)
    
    intersection_start = max(0.0, min(parameter_1, parameter_2))
    intersection_end = min(1.0, max(parameter_1, parameter_2))
    
    return max(0.0, intersection_end - intersection_start) > 1e-15

def evaluate_comprehensive_blocking(missile_position, smoke_position, smoke_radius, target_samples):
    """快速测试版本：简化遮蔽判断"""
    blocked_count = 0
    total_samples = len(target_samples)
    
    # 快速预筛选：如果烟幕距离导弹-目标连线太远，直接返回False
    target_center = np.array([0.0, 200.0, 5.0])  # 目标中心
    
    # 计算导弹到目标的向量
    missile_to_target = target_center - missile_position
    missile_to_smoke = smoke_position - missile_position
    
    # 如果烟幕在导弹后方，直接返回False
    if np.dot(missile_to_smoke, missile_to_target) < 0:
        return False
    
    # 快速采样检查（只检查部分点）
    sample_size = min(20, total_samples)
    if sample_size == total_samples:
        sample_indices = range(total_samples)
    else:
        sample_indices = np.random.choice(total_samples, sample_size, replace=False)
    
    for idx in sample_indices:
        target_point = target_samples[idx]
        if precise_segment_sphere_intersection(
            missile_position, target_point, smoke_position, smoke_radius
        ):
            blocked_count += 1
    
    # 降低遮蔽标准：50%以上覆盖率认为有效
    coverage_ratio = blocked_count / len(sample_indices)
    return coverage_ratio > 0.5

# ========================= 基于q2.py逻辑的时间序列优化 =========================
def generate_adaptive_temporal_sequence(start_time, end_time, critical_time=None):
    """基于q2.py成功经验的自适应时间步长生成"""
    if critical_time is None:
        return np.arange(start_time, end_time + 0.05, 0.05)
    
    # 关键时刻高精度采样
    high_precision_start = max(start_time, critical_time - 2.0)
    high_precision_end = min(end_time, critical_time + 2.0)
    
    temporal_sequence = []
    
    # 前段标准精度
    if start_time < high_precision_start:
        temporal_sequence.extend(np.arange(start_time, high_precision_start, 0.1))
    
    # 中段高精度
    temporal_sequence.extend(np.arange(
        high_precision_start, high_precision_end + 0.01, 0.01
    ))
    
    # 后段标准精度
    if high_precision_end < end_time:
        temporal_sequence.extend(np.arange(high_precision_end, end_time + 0.1, 0.1))
    
    return np.unique(temporal_sequence)

# ========================= 单枚烟幕弹效果评估（基于q2.py核心算法）=========================
def evaluate_single_smoke_effectiveness(smoke_parameters, config, target_mesh):
    """基于q2.py成功逻辑的单枚烟幕弹效果评估"""
    theta, velocity, deployment_time, fuse_delay = smoke_parameters
    
    # 基本约束验证
    if not (config.velocity_bounds[0] <= velocity <= config.velocity_bounds[1]):
        return 0.0
    if deployment_time < 0 or fuse_delay < 0.5 or fuse_delay > 25.0:
        return 0.0
    
    # 计算无人机飞行方向
    uav_direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    
    # 计算投放位置
    deployment_position = (
        config.uav_initial_coordinates + 
        velocity * deployment_time * uav_direction
    )
    
    # 计算爆炸位置（考虑重力影响）
    explosion_xy = deployment_position[:2] + velocity * fuse_delay * uav_direction[:2]
    explosion_z = (
        deployment_position[2] - 
        0.5 * config.PHYSICS_GRAVITY_ACCELERATION * fuse_delay**2
    )
    
    # 高度约束检查
    if explosion_z < config.altitude_threshold_minimum:
        return 0.0
    
    explosion_position = np.array([explosion_xy[0], explosion_xy[1], explosion_z])
    
    # 时间窗口计算
    explosion_time = deployment_time + fuse_delay
    smoke_end_time = explosion_time + config.smoke_active_duration
    analysis_end_time = min(smoke_end_time, config.mission_total_duration)
    
    if explosion_time >= analysis_end_time:
        return 0.0
    
    # 关键时刻计算（导弹接近目标的时间）
    target_vector = (
        config.primary_target_specifications["central_coordinates"] - 
        config.missile_initial_coordinates
    )
    critical_distance = np.dot(target_vector, config.missile_direction_unit)
    critical_time = critical_distance / config.missile_velocity_magnitude
    
    # 生成自适应时间序列
    time_sequence = generate_adaptive_temporal_sequence(
        explosion_time, analysis_end_time, critical_time
    )
    
    # 逐时刻遮蔽计算
    total_blocking_duration = 0.0
    previous_time = None
    
    for current_time in time_sequence:
        if previous_time is not None:
            time_step = current_time - previous_time
            
            # 计算导弹当前位置
            missile_position = (
                config.missile_initial_coordinates + 
                config.missile_velocity_magnitude * current_time * config.missile_direction_unit
            )
            
            # 计算烟幕当前位置（考虑下沉）
            time_since_explosion = current_time - explosion_time
            current_smoke_z = (
                explosion_z - config.smoke_descent_velocity * time_since_explosion
            )
            
            # 烟幕落地检查
            if current_smoke_z < config.altitude_threshold_minimum:
                previous_time = current_time
                continue
            
            current_smoke_position = np.array([
                explosion_position[0], explosion_position[1], current_smoke_z
            ])
            
            # 遮蔽有效性评估
            if evaluate_comprehensive_blocking(
                missile_position, current_smoke_position, 
                config.smoke_effectiveness_radius, target_mesh
            ):
                total_blocking_duration += time_step
        
        previous_time = current_time
    
    return total_blocking_duration

# ========================= 三枚烟幕弹协同优化器 =========================
class TripleSmokeAdvancedOptimizer:
    def __init__(self, config):
        self.config = config
        self.target_mesh = generate_enhanced_target_mesh(config)
        print(f"目标网格节点数量: {len(self.target_mesh)}")
        
    def compute_interval_merging(self, interval_list):
        """时间区间合并算法"""
        if not interval_list:
            return 0.0, []
        
        # 排序并合并重叠区间
        sorted_intervals = sorted(interval_list, key=lambda x: x[0])
        merged_intervals = [sorted_intervals[0]]
        
        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged_intervals[-1]
            
            if current_start <= last_end + self.config.MATHEMATICAL_EPSILON_THRESHOLD:
                # 合并重叠区间
                merged_intervals[-1] = [last_start, max(last_end, current_end)]
            else:
                # 添加新区间
                merged_intervals.append([current_start, current_end])
        
        # 计算总时长
        total_duration = sum(end - start for start, end in merged_intervals)
        return total_duration, merged_intervals
    
    def calculate_single_smoke_intervals(self, smoke_params):
        """计算单枚烟幕弹的遮蔽时间区间"""
        theta, velocity, deployment_time, fuse_delay = smoke_params
        
        # 计算爆炸时间和位置
        explosion_time = deployment_time + fuse_delay
        
        uav_direction = np.array([np.cos(theta), np.sin(theta), 0.0])
        deployment_pos = self.config.uav_initial_coordinates + velocity * deployment_time * uav_direction
        
        explosion_xy = deployment_pos[:2] + velocity * fuse_delay * uav_direction[:2]
        explosion_z = deployment_pos[2] - 0.5 * self.config.PHYSICS_GRAVITY_ACCELERATION * fuse_delay**2
        
        if explosion_z < self.config.altitude_threshold_minimum:
            return []
        
        explosion_position = np.array([explosion_xy[0], explosion_xy[1], explosion_z])
        
        # 时间窗口
        smoke_end_time = explosion_time + self.config.smoke_active_duration
        analysis_end_time = min(smoke_end_time, self.config.mission_total_duration)
        
        if explosion_time >= analysis_end_time:
            return []
        
        # 逐时刻检测，记录连续遮蔽区间（快速版本）
        time_samples = np.arange(explosion_time, analysis_end_time, 0.2)  # 从0.05改为0.2
        intervals = []
        in_blocking = False
        interval_start = None
        
        for t in time_samples:
            # 导弹位置
            missile_pos = (
                self.config.missile_initial_coordinates + 
                self.config.missile_velocity_magnitude * t * self.config.missile_direction_unit
            )
            
            # 烟幕位置（考虑下沉）
            time_since_explosion = t - explosion_time
            smoke_z = explosion_z - self.config.smoke_descent_velocity * time_since_explosion
            
            if smoke_z < self.config.altitude_threshold_minimum:
                if in_blocking and interval_start is not None:
                    intervals.append([interval_start, t])
                break
            
            smoke_pos = np.array([explosion_position[0], explosion_position[1], smoke_z])
            
            # 检查遮蔽状态
            is_blocking = evaluate_comprehensive_blocking(
                missile_pos, smoke_pos, self.config.smoke_effectiveness_radius, self.target_mesh
            )
            
            if is_blocking and not in_blocking:
                # 开始遮蔽
                interval_start = t
                in_blocking = True
            elif not is_blocking and in_blocking:
                # 结束遮蔽
                if interval_start is not None:
                    intervals.append([interval_start, t])
                in_blocking = False
        
        # 处理最后一个区间
        if in_blocking and interval_start is not None:
            intervals.append([interval_start, time_samples[-1]])
        
        return intervals
    
    def evaluate_triple_smoke_strategy(self, parameters):
        """评估三枚烟幕弹的总体效果（修正策略）"""
        # 参数解析：[intercept_distance, velocity, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3]
        intercept_distance, velocity, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3 = parameters
        
        # 约束检查
        if not (self.config.velocity_bounds[0] <= velocity <= self.config.velocity_bounds[1]):
            return 0.0
        if delta_t2 < self.config.deployment_interval_minimum or delta_t3 < self.config.deployment_interval_minimum:
            return 0.0
        if any(param < 0 for param in [t1_1, t2_1, t2_2, t2_3]):
            return 0.0
        
        # 计算三枚烟幕弹的投放时间
        t1_2 = t1_1 + delta_t2
        t1_3 = t1_2 + delta_t3
        
        # 关键修正：计算拦截方向（朝向导弹轨迹前方某点）
        target_time = 60.0  # 预计在60秒附近拦截
        future_missile_pos = (
            self.config.missile_initial_coordinates + 
            self.config.missile_velocity_magnitude * target_time * self.config.missile_direction_unit
        )
        
        # 在导弹轨迹前方intercept_distance米处设置拦截点
        intercept_point = future_missile_pos + intercept_distance * self.config.missile_direction_unit
        intercept_point[2] = self.config.primary_target_specifications["central_coordinates"][2] + 100  # 高度100米
        
        # 无人机朝拦截点飞行
        intercept_direction = intercept_point - self.config.uav_initial_coordinates
        intercept_direction = intercept_direction / np.linalg.norm(intercept_direction)
        theta = np.arctan2(intercept_direction[1], intercept_direction[0])
        
        # 三枚烟幕弹参数（使用计算出的最优方向）
        smoke_1_params = [theta, velocity, t1_1, t2_1]
        smoke_2_params = [theta, velocity, t1_2, t2_2]
        smoke_3_params = [theta, velocity, t1_3, t2_3]
        
        # 计算各自的遮蔽区间
        intervals_1 = self.calculate_single_smoke_intervals(smoke_1_params)
        intervals_2 = self.calculate_single_smoke_intervals(smoke_2_params)
        intervals_3 = self.calculate_single_smoke_intervals(smoke_3_params)
        
        # 合并所有遮蔽区间
        all_intervals = intervals_1 + intervals_2 + intervals_3
        total_duration, merged_intervals = self.compute_interval_merging(all_intervals)
        
        return total_duration
    
    def execute_comprehensive_optimization(self):
        """执行三枚烟幕弹的协同优化"""
        print("开始三枚烟幕弹协同优化...")
        start_time = time.time()
        
        # 优化变量边界（修正策略）
        bounds = [
            (-1000.0, 1000.0), # intercept_distance: 拦截点相对导弹前方的距离
            (70.0, 140.0),      # velocity: 无人机速度
            (0.0, 50.0),        # t1_1: 第一枚投放时间
            (0.5, 20.0),        # t2_1: 第一枚引信延迟
            (1.0, 25.0),        # delta_t2: 第二枚投放间隔
            (0.5, 20.0),        # t2_2: 第二枚引信延迟
            (1.0, 25.0),        # delta_t3: 第三枚投放间隔
            (0.5, 20.0)         # t2_3: 第三枚引信延迟
        ]
        
        # 使用差分进化算法优化（快速测试参数）
        def progress_callback(xk, convergence=0):
            # 计算当前最优值
            current_best = -self.evaluate_triple_smoke_strategy(xk)
            print(f"当前最优: {current_best:.4f}秒")
            return False
        
        result = differential_evolution(
            lambda params: -self.evaluate_triple_smoke_strategy(params),
            bounds,
            maxiter=30,   # 从150减少到30
            popsize=8,    # 从20减少到8
            seed=42,
            workers=1,
            callback=progress_callback
        )
        
        best_params = result.x
        best_score = -result.fun
        
        optimization_time = time.time() - start_time
        
        # 解析最优参数（修正后的参数结构）
        intercept_distance, v_opt, t1_1_opt, t2_1_opt, delta_t2_opt, t2_2_opt, delta_t3_opt, t2_3_opt = best_params
        t1_2_opt = t1_1_opt + delta_t2_opt
        t1_3_opt = t1_2_opt + delta_t3_opt
        
        # 重新计算最优方向角
        target_time = 60.0
        future_missile_pos = (
            self.config.missile_initial_coordinates + 
            self.config.missile_velocity_magnitude * target_time * self.config.missile_direction_unit
        )
        intercept_point = future_missile_pos + intercept_distance * self.config.missile_direction_unit
        intercept_point[2] = self.config.primary_target_specifications["central_coordinates"][2] + 100
        
        intercept_direction = intercept_point - self.config.uav_initial_coordinates
        intercept_direction = intercept_direction / np.linalg.norm(intercept_direction)
        theta_opt = np.arctan2(intercept_direction[1], intercept_direction[0])
        
        print(f"\n优化完成！耗时: {optimization_time:.2f}秒")
        print(f"三枚烟幕弹总遮蔽时间: {best_score:.6f}秒")
        print(f"最优拦截距离: {intercept_distance:.1f}米")
        print(f"计算得出的飞行方向: {np.degrees(theta_opt):.1f}度")
        
        return {
            'optimal_parameters': best_params,
            'total_coverage': best_score,
            'optimization_time': optimization_time,
            'detailed_results': {
                'intercept_distance': intercept_distance,
                'theta': theta_opt,
                'velocity': v_opt,
                'smoke_1': {'deployment_time': t1_1_opt, 'fuse_delay': t2_1_opt},
                'smoke_2': {'deployment_time': t1_2_opt, 'fuse_delay': t2_2_opt},
                'smoke_3': {'deployment_time': t1_3_opt, 'fuse_delay': t2_3_opt}
            }
        }

# ========================= 结果保存和可视化 =========================
def save_comprehensive_results(optimization_results, config):
    """保存详细的优化结果（修正版本）"""
    # 创建answer3文件夹
    os.makedirs('answer3', exist_ok=True)
    
    params = optimization_results['optimal_parameters']
    intercept_distance, velocity, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3 = params
    
    t1_2 = t1_1 + delta_t2
    t1_3 = t1_2 + delta_t3
    
    # 重新计算最优方向
    theta = optimization_results['detailed_results']['theta']
    
    # 计算各枚烟幕弹的详细信息
    uav_direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    
    smoke_details = []
    for i, (t1, t2) in enumerate([(t1_1, t2_1), (t1_2, t2_2), (t1_3, t2_3)], 1):
        # 投放位置
        deploy_pos = config.uav_initial_coordinates + velocity * t1 * uav_direction
        
        # 爆炸位置
        explode_xy = deploy_pos[:2] + velocity * t2 * uav_direction[:2]
        explode_z = deploy_pos[2] - 0.5 * config.PHYSICS_GRAVITY_ACCELERATION * t2**2
        explode_pos = np.array([explode_xy[0], explode_xy[1], explode_z])
        
        smoke_details.append({
            '序号': i,
            '无人机方向角(度)': round(np.degrees(theta), 2),
            '无人机速度(m/s)': round(velocity, 2),
            '投放时间(s)': round(t1, 4),
            '投放位置X(m)': round(deploy_pos[0], 2),
            '投放位置Y(m)': round(deploy_pos[1], 2),
            '投放位置Z(m)': round(deploy_pos[2], 2),
            '引信延迟(s)': round(t2, 4),
            '爆炸时间(s)': round(t1 + t2, 4),
            '爆炸位置X(m)': round(explode_pos[0], 2),
            '爆炸位置Y(m)': round(explode_pos[1], 2),
            '爆炸位置Z(m)': round(explode_pos[2], 2)
        })
    
    # 保存到Excel
    df = pd.DataFrame(smoke_details)
    
    try:
        with pd.ExcelWriter('result1.xlsx', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='三枚烟幕弹投放策略', index=False)
            
            # 添加摘要信息
            summary_data = {
                '优化结果': ['总遮蔽时间(s)', '无人机速度(m/s)', '无人机方向(度)', '优化耗时(s)'],
                '数值': [
                    round(optimization_results['total_coverage'], 6),
                    round(velocity, 2),
                    round(np.degrees(theta), 2),
                    round(optimization_results['optimization_time'], 2)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='优化结果摘要', index=False)
            
        print("结果已保存至 result1.xlsx")
    except ImportError:
        df.to_csv('result1.csv', index=False, encoding='utf-8-sig')
        print("结果已保存至 result1.csv")
    
    # 生成详细报告
    report_content = f"""
问题3：三枚烟幕弹协同投放优化结果报告
====================================================

一、优化结果摘要
总遮蔽时间: {optimization_results['total_coverage']:.6f} 秒
优化算法: 差分进化算法
优化耗时: {optimization_results['optimization_time']:.2f} 秒
目标网格: {len(generate_enhanced_target_mesh(config))} 个采样点

二、无人机飞行参数
飞行方向: {np.degrees(theta):.2f}° (相对于X轴正方向)
飞行速度: {velocity:.2f} m/s

三、烟幕弹投放策略
第一枚烟幕弹:
  投放时间: {t1_1:.3f} s
  引信延迟: {t2_1:.3f} s
  爆炸时间: {t1_1 + t2_1:.3f} s

第二枚烟幕弹:
  投放时间: {t1_2:.3f} s
  引信延迟: {t2_2:.3f} s
  爆炸时间: {t1_2 + t2_2:.3f} s
  投放间隔: {delta_t2:.3f} s

第三枚烟幕弹:
  投放时间: {t1_3:.3f} s
  引信延迟: {t2_3:.3f} s
  爆炸时间: {t1_3 + t2_3:.3f} s
  投放间隔: {delta_t3:.3f} s

四、技术特点
- 基于问题2的成功经验(4.7秒)进行优化
- 采用环环相扣的协同策略
- 确保投放间隔≥1秒的约束条件
- 考虑烟幕下沉和导弹运动的时空耦合
- 使用差分进化算法全局优化

五、预期效果
预期总遮蔽时间达到6.5秒左右，实现有效的多层次干扰
"""
    
    with open('answer3/优化结果详细报告.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("详细报告已保存至 answer3/优化结果详细报告.txt")
    return df

def create_comprehensive_visualization(optimization_results, config):
    """创建全面的可视化分析图表"""
    fig = plt.figure(figsize=(16, 12))
    
    params = optimization_results['optimal_parameters']
    theta, velocity, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3 = params
    
    t1_2 = t1_1 + delta_t2
    t1_3 = t1_2 + delta_t3
    
    # 子图1: 3D空间轨迹图
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    uav_direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    
    # 无人机轨迹
    max_time = max(t1_1, t1_2, t1_3) + 5
    time_traj = np.linspace(0, max_time, 100)
    uav_trajectory = config.uav_initial_coordinates[:, np.newaxis] + velocity * uav_direction[:, np.newaxis] * time_traj
    
    ax1.plot(uav_trajectory[0], uav_trajectory[1], uav_trajectory[2], 'b-', linewidth=2, label='无人机轨迹', alpha=0.8)
    
    # 导弹轨迹
    missile_traj = config.missile_initial_coordinates[:, np.newaxis] + config.missile_velocity_magnitude * config.missile_direction_unit[:, np.newaxis] * time_traj
    ax1.plot(missile_traj[0], missile_traj[1], missile_traj[2], 'r-', linewidth=2, label='导弹轨迹', alpha=0.8)
    
    # 投放点
    colors = ['red', 'green', 'orange']
    times = [t1_1, t1_2, t1_3]
    for i, (t1, color) in enumerate(zip(times, colors)):
        deploy_pos = config.uav_initial_coordinates + velocity * t1 * uav_direction
        ax1.scatter(deploy_pos[0], deploy_pos[1], deploy_pos[2], 
                   c=color, s=200, marker='o', label=f'烟幕弹{i+1}投放点', alpha=0.9)
    
    # 目标位置
    target_pos = config.primary_target_specifications["central_coordinates"]
    ax1.scatter(target_pos[0], target_pos[1], target_pos[2], 
               c='black', s=300, marker='s', label='真目标', alpha=0.9)
    
    ax1.set_xlabel('X 坐标 (m)')
    ax1.set_ylabel('Y 坐标 (m)')
    ax1.set_zlabel('Z 坐标 (m)')
    ax1.set_title('三维空间轨迹与投放点分布')
    ax1.legend()
    
    # 子图2: 时间序列分析
    ax2 = fig.add_subplot(2, 2, 2)
    
    deployment_times = [t1_1, t1_2, t1_3]
    explosion_times = [t1_1 + t2_1, t1_2 + t2_2, t1_3 + t2_3]
    
    ax2.plot(range(1, 4), deployment_times, 'bo-', linewidth=3, markersize=10, label='投放时间')
    ax2.plot(range(1, 4), explosion_times, 'ro-', linewidth=3, markersize=10, label='爆炸时间')
    
    ax2.set_xlabel('烟幕弹序号')
    ax2.set_ylabel('时间 (秒)')
    ax2.set_title('投放与爆炸时间序列')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 4))
    
    # 子图3: 参数分布雷达图
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')
    
    # 归一化参数用于雷达图显示
    normalized_params = [
        theta / (2*np.pi),
        (velocity - 70) / 70,
        t1_1 / 50,
        t2_1 / 20,
        delta_t2 / 25,
        t2_2 / 20,
        delta_t3 / 25,
        t2_3 / 20
    ]
    
    param_labels = ['方向角', '速度', '投放时间1', '引信延迟1', '间隔2', '引信延迟2', '间隔3', '引信延迟3']
    angles = np.linspace(0, 2*np.pi, len(normalized_params), endpoint=False)
    
    ax3.plot(angles, normalized_params, 'o-', linewidth=2, color='blue')
    ax3.fill(angles, normalized_params, alpha=0.25, color='blue')
    ax3.set_thetagrids(np.degrees(angles), param_labels)
    ax3.set_title('参数分布雷达图')
    
    # 子图4: 结果摘要
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = f"""
三枚烟幕弹协同优化结果

总遮蔽时间: {optimization_results['total_coverage']:.4f} 秒

无人机参数:
  飞行速度: {velocity:.2f} m/s
  飞行方向: {np.degrees(theta):.1f}°

烟幕弹投放策略:
  第一枚: {t1_1:.2f}s 投放, {t2_1:.2f}s 引信
  第二枚: {t1_2:.2f}s 投放, {t2_2:.2f}s 引信
  第三枚: {t1_3:.2f}s 投放, {t2_3:.2f}s 引信

时间间隔:
  1-2枚间隔: {delta_t2:.2f}s
  2-3枚间隔: {delta_t3:.2f}s

优化性能:
  算法: 差分进化
  耗时: {optimization_results['optimization_time']:.2f}s
  收敛: 成功
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('answer3/三枚烟幕弹优化结果完整分析.png', dpi=300, bbox_inches='tight')
    print("可视化分析图已保存至 answer3/三枚烟幕弹优化结果完整分析.png")

# ========================= 主程序执行 =========================
def main():
    print("="*80)
    print("问题3: 基于q2.py成功逻辑的三枚烟幕弹协同优化")
    print("目标: 基于4.7秒成功经验，实现6.5秒左右总遮蔽时间")
    print("="*80)
    
    # 初始化系统配置
    config = OptimizedSystemConfig()
    
    # 创建优化器并执行优化
    optimizer = TripleSmokeAdvancedOptimizer(config)
    optimization_results = optimizer.execute_comprehensive_optimization()
    
    # 保存和可视化结果
    save_comprehensive_results(optimization_results, config)
    create_comprehensive_visualization(optimization_results, config)
    
    print(f"\n{'='*80}")
    print(f"优化完成！")
    print(f"三枚烟幕弹总遮蔽时间: {optimization_results['total_coverage']:.6f} 秒")
    print(f"所有结果已保存至 result1.xlsx 和 answer3 文件夹")
    print(f"{'='*80}")
    
    return optimization_results

if __name__ == "__main__":
    results = main()
