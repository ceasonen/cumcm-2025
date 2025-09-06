"""
问题4: 三架无人机协同烟幕弹投放策略优化
基于Q2成功核心算法的协同版本
Created on Wed Sep  6 2024
@author: Research Team Alpha
@version: 4.3.0 (基于Q2核心算法)
"""

import numpy as np
import pandas as pd
import math
import time
import warnings
from itertools import product
import random

warnings.filterwarnings('ignore')

# ========================= 全局常量定义（与Q2一致）=========================
PHYSICS_GRAVITY_ACCELERATION = 9.8  # 重力加速度常数 (m/s²)
MATHEMATICAL_EPSILON_THRESHOLD = 1e-15  # 数值计算保护阈值
TEMPORAL_STEP_SIZE_MACRO = 0.2   # 宏观时间步长参数
TEMPORAL_STEP_SIZE_MICRO = 0.05   # 微观时间步长参数

# 目标参数（与Q2对应）
primary_target_specifications = {
    "central_coordinates": np.array([0.0, 200.0, 0.0]),  # 主目标中心位置
    "radius_parameter": 7.0,      # 目标半径参数
    "height_parameter": 10.0      # 目标高度参数
}

# 无人机初始参数
drones_init = {
    1: np.array([17800.0, 0.0, 1800.0]),  # FY1
    2: np.array([12000.0, 1400.0, 1400.0]),  # FY2
    3: np.array([6000.0, -3000.0, 700.0])  # FY3
}

# 烟幕参数（与Q2对应）
smoke_countermeasure_parameters = {
    "effective_radius": 10.0,        # 烟幕有效遮蔽半径
    "descent_velocity": 3.0,         # 烟幕重力下降速度
    "operational_duration": 20.0     # 烟幕持续作用时间
}

# 导弹参数（与Q2对应）
hostile_missile_configuration = {
    "launch_position": np.array([20000.0, 0.0, 2000.0]),  # 导弹发射位置
    "flight_velocity": 300.0         # 导弹飞行速度
}

# 预计算导弹轨迹参数（与Q2相同）
decoy_target_position = np.array([0.0, 0.0, 0.0])  # 诱饵目标坐标
missile_trajectory_vector = (decoy_target_position - hostile_missile_configuration["launch_position"])
missile_flight_distance = np.linalg.norm(missile_trajectory_vector)
missile_direction_unit_vector = missile_trajectory_vector / missile_flight_distance
missile_total_flight_time = missile_flight_distance / hostile_missile_configuration["flight_velocity"]

# ========================= Q2核心几何算法（完全复制）=========================
class PrecisionNumericalComputationKernel:
    """高精度数值计算内核（Q2原版）"""
    
    @staticmethod
    def compute_euclidean_vector_norm(input_vector):
        """计算欧几里得向量范数（模长）"""
        return np.sqrt(np.sum(input_vector**2))
    
    @staticmethod
    def analyze_line_segment_sphere_intersection_geometry(point_alpha, point_beta, sphere_centroid, sphere_radius):
        """分析线段-球体相交几何关系（Q2核心算法）"""
        directional_vector = point_beta - point_alpha
        centroid_offset_vector = sphere_centroid - point_alpha
        quadratic_coefficient_a = np.dot(directional_vector, directional_vector)
        
        # 特殊情况：零长度线段处理
        if quadratic_coefficient_a < MATHEMATICAL_EPSILON_THRESHOLD:
            euclidean_distance = PrecisionNumericalComputationKernel.compute_euclidean_vector_norm(centroid_offset_vector)
            return 1.0 if euclidean_distance <= sphere_radius + MATHEMATICAL_EPSILON_THRESHOLD else 0.0
        
        quadratic_coefficient_b = -2 * np.dot(directional_vector, centroid_offset_vector)
        quadratic_coefficient_c = np.dot(centroid_offset_vector, centroid_offset_vector) - sphere_radius**2
        discriminant_value = quadratic_coefficient_b**2 - 4*quadratic_coefficient_a*quadratic_coefficient_c
        
        # 判别式分析
        if discriminant_value < -MATHEMATICAL_EPSILON_THRESHOLD:
            return 0.0
        if discriminant_value < 0:
            discriminant_value = 0.0
        
        discriminant_sqrt = np.sqrt(discriminant_value)
        parametric_solution_1 = (-quadratic_coefficient_b - discriminant_sqrt) / (2*quadratic_coefficient_a)
        parametric_solution_2 = (-quadratic_coefficient_b + discriminant_sqrt) / (2*quadratic_coefficient_a)
        
        # 相交区间计算
        intersection_interval_start = max(0.0, min(parametric_solution_1, parametric_solution_2))
        intersection_interval_end = min(1.0, max(parametric_solution_1, parametric_solution_2))
        
        return max(0.0, intersection_interval_end - intersection_interval_start)
    
    @staticmethod
    def evaluate_comprehensive_target_occlusion_status(interceptor_coordinates, obscurant_centroid, obscurant_radius, target_mesh_vertices):
        """评估目标的全方位遮蔽状态（Q2核心算法）"""
        for mesh_vertex in target_mesh_vertices:
            intersection_ratio = PrecisionNumericalComputationKernel.analyze_line_segment_sphere_intersection_geometry(
                interceptor_coordinates, mesh_vertex, obscurant_centroid, obscurant_radius
            )
            if intersection_ratio < MATHEMATICAL_EPSILON_THRESHOLD:
                return False
        return True

# ========================= Q2目标采样算法（简化版）=========================
class AdvancedSpatialSamplingEngine:
    """高级空间几何采样引擎（基于Q2）"""
    
    @staticmethod
    def generate_comprehensive_threat_volume_discretization(threat_configuration):
        """生成威胁体积的全方位离散化采样网格（简化版）"""
        sampling_vertices = []
        geometric_centroid = threat_configuration["central_coordinates"]
        radius_param, height_param = threat_configuration["radius_parameter"], threat_configuration["height_parameter"]
        planar_center = geometric_centroid[:2]
        altitude_bounds = [geometric_centroid[2], geometric_centroid[2] + height_param]
        
        # Phase 1: 顶底面密集圆周采样
        azimuthal_discretization = np.linspace(0, 2*np.pi, 30, endpoint=False)
        for elevation_level in altitude_bounds:
            for azimuth_angle in azimuthal_discretization:
                cartesian_x = planar_center[0] + radius_param * np.cos(azimuth_angle)
                cartesian_y = planar_center[1] + radius_param * np.sin(azimuth_angle)
                sampling_vertices.append([cartesian_x, cartesian_y, elevation_level])
        
        # Phase 2: 侧面柱状表面采样
        vertical_stratification = np.linspace(altitude_bounds[0], altitude_bounds[1], 10, endpoint=True)
        for elevation_stratum in vertical_stratification:
            for azimuth_angle in azimuthal_discretization:
                cartesian_x = planar_center[0] + radius_param * np.cos(azimuth_angle)
                cartesian_y = planar_center[1] + radius_param * np.sin(azimuth_angle)
                sampling_vertices.append([cartesian_x, cartesian_y, elevation_stratum])
        
        # Phase 3: 内部体积采样（简化）
        radial_stratification = np.linspace(0, radius_param, 3, endpoint=True)
        internal_elevation_layers = np.linspace(altitude_bounds[0], altitude_bounds[1], 5, endpoint=True)
        internal_azimuthal_sectors = np.linspace(0, 2*np.pi, 8, endpoint=False)
        
        for elevation_coordinate in internal_elevation_layers:
            for radial_distance in radial_stratification:
                for azimuthal_orientation in internal_azimuthal_sectors:
                    cartesian_x = planar_center[0] + radial_distance * np.cos(azimuthal_orientation)
                    cartesian_y = planar_center[1] + radial_distance * np.sin(azimuthal_orientation)
                    sampling_vertices.append([cartesian_x, cartesian_y, elevation_coordinate])
        
        return np.unique(np.array(sampling_vertices, dtype=np.float64), axis=0)

# ========================= Q2时间序列管理（简化版）=========================
class AdaptiveTemporalSequenceManager:
    """自适应时间序列管理系统（基于Q2）"""
    
    @staticmethod
    def construct_multi_resolution_temporal_sequence(sequence_start, sequence_end, critical_event_timestamp=None):
        """构建多分辨率时间序列（简化版）"""
        if critical_event_timestamp is None:
            return np.arange(sequence_start, sequence_end + TEMPORAL_STEP_SIZE_MACRO, 
                           TEMPORAL_STEP_SIZE_MACRO)
        
        # 关键事件周围的高分辨率窗口
        high_resolution_window_start = max(sequence_start, critical_event_timestamp - 2.0)
        high_resolution_window_end = min(sequence_end, critical_event_timestamp + 2.0)
        
        # 组合式时间序列构建
        temporal_sequence_components = []
        
        # 前置粗粒度时间段
        if sequence_start < high_resolution_window_start:
            temporal_sequence_components.extend(
                np.arange(sequence_start, high_resolution_window_start, TEMPORAL_STEP_SIZE_MACRO)
            )
        
        # 中央细粒度时间段
        temporal_sequence_components.extend(
            np.arange(high_resolution_window_start, high_resolution_window_end + TEMPORAL_STEP_SIZE_MICRO, 
                     TEMPORAL_STEP_SIZE_MICRO)
        )
        
        # 后置粗粒度时间段
        if high_resolution_window_end < sequence_end:
            temporal_sequence_components.extend(
                np.arange(high_resolution_window_end, sequence_end + TEMPORAL_STEP_SIZE_MACRO, 
                         TEMPORAL_STEP_SIZE_MACRO)
            )
        
        return np.unique(temporal_sequence_components)

# ========================= 单无人机战术评估（基于Q2）=========================
def single_drone_tactical_effectiveness(drone_id, strategic_parameter_vector, target_discretization_mesh):
    """单架无人机战术效能评估函数（基于Q2算法）"""
    flight_azimuth, platform_velocity, deployment_temporal_offset, detonation_temporal_offset = strategic_parameter_vector
    
    # 参数约束验证模块（与Q2相同）
    if not (70.0 <= platform_velocity <= 140.0):
        return 0.0
    if deployment_temporal_offset < 0 or detonation_temporal_offset < 0:
        return 0.0
    
    # Step 1: 计算载荷投放空间坐标
    flight_direction_unit_vector = np.array([np.cos(flight_azimuth), np.sin(flight_azimuth), 0.0], dtype=np.float64)
    uav_initial_position = drones_init[drone_id]
    deployment_spatial_coordinates = (uav_initial_position + 
                                    platform_velocity * deployment_temporal_offset * flight_direction_unit_vector)
    
    # Step 2: 计算载荷起爆空间坐标
    detonation_horizontal_displacement = deployment_spatial_coordinates[:2] + platform_velocity * detonation_temporal_offset * flight_direction_unit_vector[:2]
    gravitational_altitude_loss = deployment_spatial_coordinates[2] - 0.5 * PHYSICS_GRAVITY_ACCELERATION * detonation_temporal_offset**2
    
    if gravitational_altitude_loss < 5.0:
        return 0.0
    
    detonation_spatial_coordinates = np.array([detonation_horizontal_displacement[0], detonation_horizontal_displacement[1], gravitational_altitude_loss], dtype=np.float64)
    
    # Step 3: 时间窗口分析模块
    absolute_detonation_timestamp = deployment_temporal_offset + detonation_temporal_offset
    countermeasure_expiration_timestamp = absolute_detonation_timestamp + smoke_countermeasure_parameters["operational_duration"]
    
    analysis_termination_timestamp = min(countermeasure_expiration_timestamp, missile_total_flight_time)
    
    if absolute_detonation_timestamp >= analysis_termination_timestamp:
        return 0.0
    
    # Step 4: 关键时刻识别
    threat_displacement_vector = (primary_target_specifications["central_coordinates"] - 
                                hostile_missile_configuration["launch_position"])
    critical_engagement_distance = np.dot(threat_displacement_vector, missile_direction_unit_vector)
    critical_engagement_timestamp = critical_engagement_distance / hostile_missile_configuration["flight_velocity"]
    
    temporal_analysis_sequence = AdaptiveTemporalSequenceManager.construct_multi_resolution_temporal_sequence(
        absolute_detonation_timestamp, analysis_termination_timestamp, critical_engagement_timestamp
    )
    
    # Step 5: 累积遮蔽效能计算（与Q2相同）
    accumulated_occlusion_duration = 0.0
    previous_timestamp = None
    
    for current_timestamp in temporal_analysis_sequence:
        if previous_timestamp is not None:
            temporal_increment = current_timestamp - previous_timestamp
            
            # 拦截器当前位置计算
            interceptor_current_coordinates = (hostile_missile_configuration["launch_position"] + 
                                             hostile_missile_configuration["flight_velocity"] * 
                                             current_timestamp * missile_direction_unit_vector)
            
            # 烟幕当前状态评估
            gravitational_descent_duration = current_timestamp - absolute_detonation_timestamp
            current_obscurant_altitude = (detonation_spatial_coordinates[2] - 
                                        smoke_countermeasure_parameters["descent_velocity"] * 
                                        gravitational_descent_duration)
            
            if current_obscurant_altitude < 2.0:
                previous_timestamp = current_timestamp
                continue
            
            current_obscurant_centroid = np.array([detonation_spatial_coordinates[0], detonation_spatial_coordinates[1], current_obscurant_altitude], dtype=np.float64)
            
            # 遮蔽效能评估（使用Q2核心算法）
            is_target_occluded = PrecisionNumericalComputationKernel.evaluate_comprehensive_target_occlusion_status(
                interceptor_current_coordinates, current_obscurant_centroid, 
                smoke_countermeasure_parameters["effective_radius"], 
                target_discretization_mesh
            )
            
            if is_target_occluded:
                accumulated_occlusion_duration += temporal_increment
        
        previous_timestamp = current_timestamp
    
    return accumulated_occlusion_duration

# ========================= 协同策略评估=========================
def evaluate_combined_drone_strategies(strategies, target_discretization_mesh):
    """评估多架无人机的协同策略效果"""
    try:
        # 计算每个烟幕的时间窗口
        smoke_windows = []
        
        for strategy in strategies:
            drone_id = strategy['drone_id']
            params = [strategy['theta'], strategy['v'], strategy['t_deploy'], strategy['t_detonate']]
            
            # 计算起爆位置和时间
            flight_direction = np.array([np.cos(params[0]), np.sin(params[0]), 0.0])
            uav_pos = drones_init[drone_id]
            deployment_pos = uav_pos + params[1] * params[2] * flight_direction
            
            det_horizontal = deployment_pos[:2] + params[1] * params[3] * flight_direction[:2]
            det_z = deployment_pos[2] - 0.5 * PHYSICS_GRAVITY_ACCELERATION * params[3]**2
            
            if det_z < 5.0:
                continue
            
            abs_det_time = params[2] + params[3]
            smoke_end = abs_det_time + smoke_countermeasure_parameters["operational_duration"]
            
            smoke_windows.append({
                'start': abs_det_time,
                'end': min(smoke_end, missile_total_flight_time),
                'center': np.array([det_horizontal[0], det_horizontal[1], det_z])
            })
        
        if len(smoke_windows) == 0:
            return 0.0
        
        # 计算总遮蔽时间
        total_start = min(w['start'] for w in smoke_windows)
        total_end = max(w['end'] for w in smoke_windows)
        
        # 使用Q2的时间序列管理
        critical_time = np.mean([w['start'] for w in smoke_windows])
        time_sequence = AdaptiveTemporalSequenceManager.construct_multi_resolution_temporal_sequence(
            total_start, total_end, critical_time
        )
        
        total_coverage = 0.0
        prev_time = None
        
        for curr_time in time_sequence:
            if prev_time is not None:
                time_increment = curr_time - prev_time
                
                # 计算活跃烟幕
                active_smokes = []
                for window in smoke_windows:
                    if window['start'] <= curr_time <= window['end']:
                        sink_time = curr_time - window['start']
                        smoke_z = window['center'][2] - smoke_countermeasure_parameters["descent_velocity"] * sink_time
                        if smoke_z > 2.0:
                            active_smokes.append(np.array([window['center'][0], window['center'][1], smoke_z]))
                
                if len(active_smokes) == 0:
                    prev_time = curr_time
                    continue
                
                # 导弹位置
                missile_pos = (hostile_missile_configuration["launch_position"] + 
                             hostile_missile_configuration["flight_velocity"] * curr_time * missile_direction_unit_vector)
                
                # 检查任意烟幕是否能遮蔽目标（使用Q2算法）
                is_covered = False
                for smoke_center in active_smokes:
                    if PrecisionNumericalComputationKernel.evaluate_comprehensive_target_occlusion_status(
                        missile_pos, smoke_center, smoke_countermeasure_parameters["effective_radius"], 
                        target_discretization_mesh):
                        is_covered = True
                        break
                
                if is_covered:
                    total_coverage += time_increment
            
            prev_time = curr_time
        
        return total_coverage
    
    except:
        return 0.0

# ========================= 策略生成与优化=========================
def generate_drone_strategy(drone_id, target_mesh):
    """为指定无人机生成优化策略"""
    print(f"为无人机FY{drone_id}生成策略...")
    
    # 计算智能初始值
    drone_pos = drones_init[drone_id]
    target_pos = primary_target_specifications["central_coordinates"]
    
    to_target = target_pos - drone_pos
    distance = np.linalg.norm(to_target[:2])
    optimal_theta = math.atan2(to_target[1], to_target[0])
    
    # 估算参数
    flight_time = distance / 100.0
    height_diff = drone_pos[2] - target_pos[2]
    fall_time = math.sqrt(2 * max(height_diff, 50) / PHYSICS_GRAVITY_ACCELERATION)
    
    # 生成候选策略
    best_strategy = None
    best_score = 0.0
    
    # 参数搜索范围
    theta_range = np.linspace(optimal_theta - 0.3, optimal_theta + 0.3, 10)
    v_range = np.linspace(80, 120, 8)
    t_deploy_range = np.linspace(max(5, flight_time * 0.5), flight_time * 1.5, 8)
    t_detonate_range = np.linspace(max(2, fall_time * 0.5), min(20, fall_time * 1.5), 8)
    
    count = 0
    for theta in theta_range:
        for v in v_range:
            for t_deploy in t_deploy_range:
                for t_detonate in t_detonate_range:
                    count += 1
                    params = [theta, v, t_deploy, t_detonate]
                    score = single_drone_tactical_effectiveness(drone_id, params, target_mesh)
                    
                    if score > best_score:
                        best_score = score
                        best_strategy = {
                            'drone_id': drone_id,
                            'theta': theta,
                            'v': v,
                            't_deploy': t_deploy,
                            't_detonate': t_detonate,
                            'score': score
                        }
                        print(f"  策略改进: 遮蔽时间{score:.3f}s")
    
    if best_strategy is None:
        # 基础策略
        best_strategy = {
            'drone_id': drone_id,
            'theta': optimal_theta,
            'v': 100.0,
            't_deploy': max(5, flight_time),
            't_detonate': max(3, fall_time),
            'score': 0.0
        }
    
    print(f"FY{drone_id}最优策略: 遮蔽时间{best_strategy['score']:.3f}s")
    return best_strategy

# ========================= 主程序=========================
def main():
    start_time = time.time()
    
    print("="*60)
    print("问题4: 三架无人机协同烟幕弹投放策略优化")
    print("基于Q2成功核心算法的协同版本")
    print("="*60)
    
    # 生成目标采样网格（使用Q2算法）
    print("\n生成目标采样网格...")
    target_mesh = AdvancedSpatialSamplingEngine.generate_comprehensive_threat_volume_discretization(
        primary_target_specifications
    )
    print(f"目标采样点数量: {len(target_mesh)}")
    
    # 为每架无人机生成最优策略
    print(f"\n第1步：生成各无人机最优策略")
    strategies = []
    
    for drone_id in [1, 2, 3]:
        strategy = generate_drone_strategy(drone_id, target_mesh)
        strategies.append(strategy)
    
    # 评估协同效果
    print(f"\n第2步：评估三机协同效果")
    combined_score = evaluate_combined_drone_strategies(strategies, target_mesh)
    print(f"三机协同总遮蔽时间: {combined_score:.3f}s")
    
    # 保存结果
    print(f"\n第3步：保存优化结果")
    
    data = []
    for strategy in strategies:
        data.append([
            f"FY{strategy['drone_id']}",
            f"{math.degrees(strategy['theta']):.2f}°",
            f"{strategy['v']:.1f}",
            f"{strategy['t_deploy']:.2f}",
            f"{strategy['t_detonate']:.2f}",
            f"{strategy['score']:.3f}",
            f"{combined_score:.3f}"
        ])
    
    df = pd.DataFrame(data, columns=[
        "无人机", "飞行方向", "速度(m/s)", "投放延迟(s)", 
        "起爆延迟(s)", "单独遮蔽(s)", "协同总时间(s)"
    ])
    
    try:
        df.to_excel("result2.xlsx", index=False, engine="openpyxl")
        print("结果已保存至 result2.xlsx")
    except:
        df.to_csv("result2.csv", index=False, encoding='utf-8-sig')
        print("结果已保存至 result2.csv")
    
    # 总结
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    if combined_score >= 10.0:
        print(f"✓ 优化成功！协同遮蔽时间: {combined_score:.3f}s (≥10s)")
    elif combined_score >= 5.0:
        print(f"△ 接近目标！协同遮蔽时间: {combined_score:.3f}s")
    else:
        print(f"△ 基础方案：协同遮蔽时间: {combined_score:.3f}s")
    
    print(f"计算耗时: {elapsed:.1f}秒")
    print(f"{'='*60}")
    
    return strategies, combined_score

if __name__ == "__main__":
    strategies, score = main()
