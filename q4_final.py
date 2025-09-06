#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题4: 三架无人机协同烟幕弹投放策略优化
完全基于Q2成功算法的协同版本 - 目标11+秒
Created on Wed Sep  6 2024
@author: Research Team Alpha
@version: 4.3.0 (Q2核心算法移植版)
"""

import numpy as np
import pandas as pd
import math
import time
import warnings
from itertools import product

warnings.filterwarnings('ignore')

# ========================= Q2核心常数配置 =========================
PHYSICS_GRAVITY_ACCELERATION = 9.8
MATHEMATICAL_EPSILON_THRESHOLD = 1e-15
TEMPORAL_STEP_SIZE_MACRO = 0.1
TEMPORAL_STEP_SIZE_MICRO = 0.005

# 战场态势感知参数（Q2配置）
decoy_target_position = np.array([0.0, 0.0, 0.0])
primary_target_specifications = {
    "central_coordinates": np.array([0.0, 200.0, 0.0]),
    "radius_parameter": 7.0,
    "height_parameter": 10.0
}

# 烟幕参数（Q2配置）
smoke_countermeasure_parameters = {
    "effective_radius": 10.0,
    "descent_velocity": 3.0,
    "operational_duration": 20.0
}

# 敌方导弹参数（Q2配置）
hostile_missile_configuration = {
    "launch_position": np.array([20000.0, 0.0, 2000.0]),
    "flight_velocity": 300.0
}

# 三架无人机初始位置
uav_initial_positions = {
    1: np.array([17800.0, 0.0, 1800.0]),   # FY1
    2: np.array([12000.0, 1400.0, 1400.0]), # FY2  
    3: np.array([6000.0, -3000.0, 700.0])   # FY3
}

# 预计算导弹轨迹参数
missile_trajectory_vector = (decoy_target_position - hostile_missile_configuration["launch_position"])
missile_flight_distance = np.linalg.norm(missile_trajectory_vector)
missile_direction_unit_vector = missile_trajectory_vector / missile_flight_distance
missile_total_flight_time = missile_flight_distance / hostile_missile_configuration["flight_velocity"]

# ========================= Q2核心几何计算引擎 =========================
class AdvancedSpatialSamplingEngine:
    """Q2的高级空间几何采样引擎"""
    
    @staticmethod
    def generate_comprehensive_threat_volume_discretization(threat_configuration):
        """生成威胁体积的全方位离散化采样网格（Q2算法）"""
        sampling_vertices = []
        geometric_centroid = threat_configuration["central_coordinates"]
        radius_param, height_param = threat_configuration["radius_parameter"], threat_configuration["height_parameter"]
        planar_center = geometric_centroid[:2]
        altitude_bounds = [geometric_centroid[2], geometric_centroid[2] + height_param]
        
        # Phase 1: 顶底面密集圆周采样
        azimuthal_discretization = np.linspace(0, 2*np.pi, 60, endpoint=False)
        for elevation_level in altitude_bounds:
            for azimuth_angle in azimuthal_discretization:
                cartesian_x = planar_center[0] + radius_param * np.cos(azimuth_angle)
                cartesian_y = planar_center[1] + radius_param * np.sin(azimuth_angle)
                sampling_vertices.append([cartesian_x, cartesian_y, elevation_level])
        
        # Phase 2: 侧面柱状表面采样
        vertical_stratification = np.linspace(altitude_bounds[0], altitude_bounds[1], 20, endpoint=True)
        for elevation_stratum in vertical_stratification:
            for azimuth_angle in azimuthal_discretization:
                cartesian_x = planar_center[0] + radius_param * np.cos(azimuth_angle)
                cartesian_y = planar_center[1] + radius_param * np.sin(azimuth_angle)
                sampling_vertices.append([cartesian_x, cartesian_y, elevation_stratum])
        
        # Phase 3: 内部体积三维网格化
        radial_stratification = np.linspace(0, radius_param, 8, endpoint=True)
        for elevation_level in vertical_stratification[::2]:
            for radial_distance in radial_stratification:
                for azimuth_angle in azimuthal_discretization[::4]:
                    cartesian_x = planar_center[0] + radial_distance * np.cos(azimuth_angle)
                    cartesian_y = planar_center[1] + radial_distance * np.sin(azimuth_angle)
                    sampling_vertices.append([cartesian_x, cartesian_y, elevation_level])
        
        return np.unique(np.array(sampling_vertices, dtype=np.float64), axis=0)

class PrecisionNumericalComputationKernel:
    """Q2的精密数值计算内核"""
    
    @staticmethod
    def analyze_line_segment_sphere_intersection_geometry(line_start_vertex, line_terminal_vertex, 
                                                        sphere_centroid_coordinates, sphere_geometric_radius):
        """分析线段与球体相交几何关系（Q2核心算法）"""
        directional_displacement_vector = line_terminal_vertex - line_start_vertex
        centroid_displacement_vector = line_start_vertex - sphere_centroid_coordinates
        
        quadratic_coefficient_a = np.dot(directional_displacement_vector, directional_displacement_vector)
        if quadratic_coefficient_a < MATHEMATICAL_EPSILON_THRESHOLD:
            return np.linalg.norm(centroid_displacement_vector) <= sphere_geometric_radius + MATHEMATICAL_EPSILON_THRESHOLD
        
        quadratic_coefficient_b = 2.0 * np.dot(centroid_displacement_vector, directional_displacement_vector)
        quadratic_coefficient_c = (np.dot(centroid_displacement_vector, centroid_displacement_vector) - 
                                 sphere_geometric_radius**2)
        discriminant_value = quadratic_coefficient_b**2 - 4.0 * quadratic_coefficient_a * quadratic_coefficient_c
        
        if discriminant_value < -MATHEMATICAL_EPSILON_THRESHOLD:
            return False
        
        if discriminant_value < 0:
            discriminant_value = 0
        
        discriminant_square_root = np.sqrt(discriminant_value)
        intersection_parameter_1 = (-quadratic_coefficient_b - discriminant_square_root) / (2.0 * quadratic_coefficient_a)
        intersection_parameter_2 = (-quadratic_coefficient_b + discriminant_square_root) / (2.0 * quadratic_coefficient_a)
        
        return ((intersection_parameter_1 <= 1.0 + MATHEMATICAL_EPSILON_THRESHOLD and 
                intersection_parameter_2 >= -MATHEMATICAL_EPSILON_THRESHOLD))
    
    @staticmethod
    def evaluate_comprehensive_target_occlusion_status(interceptor_coordinates, obscurant_centroid, 
                                                     obscurant_radius, target_sampling_mesh):
        """评估综合目标遮蔽状态（Q2核心算法）"""
        for target_vertex_coordinates in target_sampling_mesh:
            intersection_detected = PrecisionNumericalComputationKernel.analyze_line_segment_sphere_intersection_geometry(
                interceptor_coordinates, target_vertex_coordinates, obscurant_centroid, obscurant_radius
            )
            if not intersection_detected:
                return False
        return True

class AdaptiveTemporalSequenceManager:
    """Q2的自适应时间序列管理器"""
    
    @staticmethod
    def construct_multi_resolution_temporal_sequence(analysis_commencement_timestamp, 
                                                   analysis_termination_timestamp, 
                                                   critical_engagement_timestamp):
        """构建多分辨率时间序列（Q2算法）"""
        temporal_sampling_points = []
        
        # 粗粒度时间采样
        coarse_temporal_sequence = np.arange(analysis_commencement_timestamp, 
                                           analysis_termination_timestamp, 
                                           TEMPORAL_STEP_SIZE_MACRO)
        temporal_sampling_points.extend(coarse_temporal_sequence)
        
        # 关键时刻高分辨率采样
        critical_temporal_neighborhood_radius = 2.0
        fine_sampling_commencement = max(analysis_commencement_timestamp, 
                                       critical_engagement_timestamp - critical_temporal_neighborhood_radius)
        fine_sampling_termination = min(analysis_termination_timestamp, 
                                      critical_engagement_timestamp + critical_temporal_neighborhood_radius)
        
        if fine_sampling_commencement < fine_sampling_termination:
            fine_temporal_sequence = np.arange(fine_sampling_commencement, 
                                             fine_sampling_termination, 
                                             TEMPORAL_STEP_SIZE_MICRO)
            temporal_sampling_points.extend(fine_temporal_sequence)
        
        # 时间点去重与排序
        unique_temporal_points = np.unique(np.array(temporal_sampling_points))
        return unique_temporal_points[(unique_temporal_points >= analysis_commencement_timestamp) & 
                                    (unique_temporal_points <= analysis_termination_timestamp)]

# ========================= 单无人机战术评估函数（基于Q2）=========================
def tactical_effectiveness_evaluation_function_single_drone(strategic_parameter_vector, target_discretization_mesh, drone_id):
    """单架无人机战术效能评估函数（Q2核心算法移植）"""
    flight_azimuth, platform_velocity, deployment_temporal_offset, detonation_temporal_offset = strategic_parameter_vector
    
    # 参数约束验证模块
    if not (70.0 <= platform_velocity <= 140.0):
        return 0.0
    if deployment_temporal_offset < 0 or detonation_temporal_offset < 0:
        return 0.0
    
    # 获取对应无人机的初始位置
    uav_initial_position_vector = uav_initial_positions[drone_id]
    
    # Step 1: 计算载荷投放空间坐标
    flight_direction_unit_vector = np.array([np.cos(flight_azimuth), np.sin(flight_azimuth), 0.0], dtype=np.float64)
    deployment_spatial_coordinates = (uav_initial_position_vector + 
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
    
    # Step 5: 累积遮蔽效能计算
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
            
            # 遮蔽效能评估
            is_target_occluded = PrecisionNumericalComputationKernel.evaluate_comprehensive_target_occlusion_status(
                interceptor_current_coordinates, current_obscurant_centroid, 
                smoke_countermeasure_parameters["effective_radius"], 
                target_discretization_mesh
            )
            
            if is_target_occluded:
                accumulated_occlusion_duration += temporal_increment
        
        previous_timestamp = current_timestamp
    
    return accumulated_occlusion_duration

# ========================= 多无人机协同评估函数 =========================
def evaluate_multi_drone_coordination(drone_strategies, target_discretization_mesh):
    """评估多架无人机协同遮蔽效果"""
    try:
        # 计算每个烟幕的时间窗口和位置
        smoke_windows = []
        
        for strategy in drone_strategies:
            drone_id = strategy['drone_id']
            flight_azimuth = strategy['theta']
            platform_velocity = strategy['v']
            deployment_temporal_offset = strategy['t_deploy']
            detonation_temporal_offset = strategy['t_detonate']
            
            uav_initial_position_vector = uav_initial_positions[drone_id]
            
            # 计算起爆位置
            flight_direction_unit_vector = np.array([np.cos(flight_azimuth), np.sin(flight_azimuth), 0.0])
            deployment_spatial_coordinates = (uav_initial_position_vector + 
                                            platform_velocity * deployment_temporal_offset * flight_direction_unit_vector)
            
            detonation_horizontal_displacement = deployment_spatial_coordinates[:2] + platform_velocity * detonation_temporal_offset * flight_direction_unit_vector[:2]
            gravitational_altitude_loss = deployment_spatial_coordinates[2] - 0.5 * PHYSICS_GRAVITY_ACCELERATION * detonation_temporal_offset**2
            
            if gravitational_altitude_loss < 5.0:
                continue
                
            detonation_spatial_coordinates = np.array([detonation_horizontal_displacement[0], detonation_horizontal_displacement[1], gravitational_altitude_loss])
            
            absolute_detonation_timestamp = deployment_temporal_offset + detonation_temporal_offset
            countermeasure_expiration_timestamp = absolute_detonation_timestamp + smoke_countermeasure_parameters["operational_duration"]
            
            smoke_windows.append({
                'start': absolute_detonation_timestamp,
                'end': countermeasure_expiration_timestamp,
                'center': detonation_spatial_coordinates
            })
        
        if len(smoke_windows) == 0:
            return 0.0
        
        # 计算总时间窗口
        overall_start = min(w['start'] for w in smoke_windows)
        overall_end = max(w['end'] for w in smoke_windows)
        analysis_termination_timestamp = min(overall_end, missile_total_flight_time)
        
        # 关键时刻识别
        threat_displacement_vector = (primary_target_specifications["central_coordinates"] - 
                                    hostile_missile_configuration["launch_position"])
        critical_engagement_distance = np.dot(threat_displacement_vector, missile_direction_unit_vector)
        critical_engagement_timestamp = critical_engagement_distance / hostile_missile_configuration["flight_velocity"]
        
        # 构建时间序列
        temporal_analysis_sequence = AdaptiveTemporalSequenceManager.construct_multi_resolution_temporal_sequence(
            overall_start, analysis_termination_timestamp, critical_engagement_timestamp
        )
        
        # 累积遮蔽效能计算
        accumulated_occlusion_duration = 0.0
        previous_timestamp = None
        
        for current_timestamp in temporal_analysis_sequence:
            if previous_timestamp is not None:
                temporal_increment = current_timestamp - previous_timestamp
                
                # 拦截器当前位置
                interceptor_current_coordinates = (hostile_missile_configuration["launch_position"] + 
                                                 hostile_missile_configuration["flight_velocity"] * 
                                                 current_timestamp * missile_direction_unit_vector)
                
                # 检查当前时刻是否有有效烟幕遮蔽
                target_occluded = False
                
                for window in smoke_windows:
                    if window['start'] <= current_timestamp <= window['end']:
                        # 计算烟幕当前位置
                        gravitational_descent_duration = current_timestamp - window['start']
                        current_obscurant_altitude = (window['center'][2] - 
                                                    smoke_countermeasure_parameters["descent_velocity"] * 
                                                    gravitational_descent_duration)
                        
                        if current_obscurant_altitude >= 2.0:
                            current_obscurant_centroid = np.array([window['center'][0], window['center'][1], current_obscurant_altitude])
                            
                            # 检查是否遮蔽
                            is_occluded = PrecisionNumericalComputationKernel.evaluate_comprehensive_target_occlusion_status(
                                interceptor_current_coordinates, current_obscurant_centroid, 
                                smoke_countermeasure_parameters["effective_radius"], 
                                target_discretization_mesh
                            )
                            
                            if is_occluded:
                                target_occluded = True
                                break
                
                if target_occluded:
                    accumulated_occlusion_duration += temporal_increment
            
            previous_timestamp = current_timestamp
        
        return accumulated_occlusion_duration
        
    except Exception as e:
        print(f"协同评估错误: {e}")
        return 0.0

# ========================= 智能策略生成 =========================
def generate_optimized_strategy(drone_id, target_discretization_mesh):
    """为指定无人机生成优化策略"""
    uav_pos = uav_initial_positions[drone_id]
    target_pos = primary_target_specifications["central_coordinates"]
    
    # 计算朝向目标的最优方向
    to_target = target_pos - uav_pos
    distance = np.linalg.norm(to_target[:2])
    optimal_theta = math.atan2(to_target[1], to_target[0])
    
    # 估算合理参数
    flight_time = distance / 100.0
    height_diff = uav_pos[2] - (target_pos[2] + primary_target_specifications["height_parameter"]/2)
    fall_time = math.sqrt(2 * max(height_diff, 50) / PHYSICS_GRAVITY_ACCELERATION) if height_diff > 0 else 8.0
    
    # 生成多个候选策略
    best_strategy = None
    best_score = 0.0
    
    print(f"为无人机FY{drone_id}优化策略...")
    
    # 扩大参数搜索范围（更加激进的搜索）
    theta_variations = [0, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, -0.5, 0.5]
    v_variations = [70, 80, 90, 100, 110, 120, 130, 140]
    t_deploy_base = max(5, flight_time * 0.7)
    t_deploy_variations = [0, -20, -10, -5, 5, 10, 15, 20, 30, 40]
    t_detonate_base = max(3, min(fall_time, 15))
    t_detonate_variations = [0, -5, -3, -1, 1, 2, 3, 5, 8, 10, 15]
    
    count = 0
    for dtheta in theta_variations:
        for v in v_variations:
            for dt_deploy in t_deploy_variations:
                for dt_detonate in t_detonate_variations:
                    count += 1
                    
                    theta = optimal_theta + dtheta
                    t_deploy = max(2, t_deploy_base + dt_deploy)
                    t_detonate = max(1, min(20, t_detonate_base + dt_detonate))
                    
                    # 评估策略
                    params = [theta, v, t_deploy, t_detonate]
                    score = tactical_effectiveness_evaluation_function_single_drone(
                        params, target_discretization_mesh, drone_id)
                    
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
                        print(f"  策略改进 #{count}: 遮蔽时间{score:.3f}s")
    
    if best_strategy is None:
        # 回退策略
        best_strategy = {
            'drone_id': drone_id,
            'theta': optimal_theta,
            'v': 100,
            't_deploy': t_deploy_base,
            't_detonate': t_detonate_base,
            'score': 0.0
        }
    
    return best_strategy

# ========================= 主优化程序 =========================
def main():
    start_time = time.time()
    
    print("="*60)
    print("问题4: 三架无人机协同烟幕弹投放策略优化")
    print("完全基于Q2成功算法的协同版本 - 目标11+秒")
    print("="*60)
    
    # 生成目标采样网格（使用Q2的算法）
    print("\n生成目标采样网格...")
    target_discretization_mesh = AdvancedSpatialSamplingEngine.generate_comprehensive_threat_volume_discretization(
        primary_target_specifications
    )
    print(f"目标采样点数量: {len(target_discretization_mesh)}")
    
    # 第1步：为每架无人机生成最优策略
    print(f"\n第1步：生成各无人机最优策略")
    drone_strategies = []
    
    for drone_id in [1, 2, 3]:
        strategy = generate_optimized_strategy(drone_id, target_discretization_mesh)
        drone_strategies.append(strategy)
        print(f"FY{drone_id}最优策略: 遮蔽时间{strategy['score']:.3f}s")
    
    # 第2步：评估协同效果
    print(f"\n第2步：评估三机协同效果")
    combined_score = evaluate_multi_drone_coordination(drone_strategies, target_discretization_mesh)
    print(f"三机协同总遮蔽时间: {combined_score:.3f}s")
    
    # 第3步：如果效果不佳，进行组合优化
    if combined_score < 10.0:
        print(f"\n第3步：进行组合策略优化...")
        
        # 为每架无人机生成多个候选策略
        all_candidates = {}
        for drone_id in [1, 2, 3]:
            candidates = []
            strategy = drone_strategies[drone_id-1]
            
            # 在最优策略周围生成变化
            base_params = [strategy['theta'], strategy['v'], strategy['t_deploy'], strategy['t_detonate']]
            
            variations = [
                [0, 0, 0, 0],       # 原策略
                [0.2, 15, 10, 5],   # 大变化1
                [-0.2, -15, -10, -5], # 大变化2
                [0.3, 20, 15, 8],   # 大变化3
                [-0.3, -20, -15, -8], # 大变化4
                [0.1, 10, 5, 3],    # 中等变化1
                [-0.1, -10, -5, -3], # 中等变化2
                [0.4, 25, 20, 10],  # 极大变化1
                [-0.4, -25, -20, -10], # 极大变化2
                [0.05, 5, 3, 2],    # 小变化1
                [-0.05, -5, -3, -2]  # 小变化2
            ]
            
            for i, var in enumerate(variations):
                theta = base_params[0] + var[0]
                v = max(70, min(140, base_params[1] + var[1]))
                t_deploy = max(2, base_params[2] + var[2])
                t_detonate = max(1, min(20, base_params[3] + var[3]))
                
                params = [theta, v, t_deploy, t_detonate]
                score = tactical_effectiveness_evaluation_function_single_drone(
                    params, target_discretization_mesh, drone_id)
                
                candidates.append({
                    'drone_id': drone_id,
                    'theta': theta,
                    'v': v,
                    't_deploy': t_deploy,
                    't_detonate': t_detonate,
                    'score': score
                })
            
            all_candidates[drone_id] = candidates
            print(f"FY{drone_id}生成{len(candidates)}个候选策略")
        
        # 寻找最优组合
        best_combination = None
        best_combined_score = combined_score
        total_combinations = len(all_candidates[1]) * len(all_candidates[2]) * len(all_candidates[3])
        print(f"评估{total_combinations}种策略组合...")
        
        count = 0
        for s1 in all_candidates[1]:
            for s2 in all_candidates[2]:
                for s3 in all_candidates[3]:
                    count += 1
                    test_strategies = [s1, s2, s3]
                    test_score = evaluate_multi_drone_coordination(test_strategies, target_discretization_mesh)
                    
                    if test_score > best_combined_score:
                        best_combined_score = test_score
                        best_combination = test_strategies
                        print(f"  组合改进 #{count}: 总遮蔽时间{test_score:.3f}s")
        
        if best_combination is not None:
            drone_strategies = best_combination
            combined_score = best_combined_score
    
    # 第4步：保存结果
    print(f"\n第4步：保存优化结果")
    
    # 准备数据
    data = []
    for strategy in drone_strategies:
        data.append([
            f"FY{strategy['drone_id']}",
            f"{math.degrees(strategy['theta']):.2f}°",
            f"{strategy['v']:.1f}",
            f"{strategy['t_deploy']:.2f}",
            f"{strategy['t_detonate']:.2f}",
            f"{strategy['score']:.3f}",
            f"{combined_score:.3f}"
        ])
    
    # 保存到Excel
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
    if combined_score >= 11.0:
        print(f"✓ 达到目标！协同遮蔽时间: {combined_score:.3f}s (≥11s)")
    elif combined_score >= 8.0:
        print(f"△ 接近目标！协同遮蔽时间: {combined_score:.3f}s")
    else:
        print(f"△ 基础方案：协同遮蔽时间: {combined_score:.3f}s")
    
    print(f"计算耗时: {elapsed:.1f}秒")
    print(f"{'='*60}")
    
    return drone_strategies, combined_score

if __name__ == "__main__":
    strategies, score = main()
