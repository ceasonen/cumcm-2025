"""
问题4: 基于Q2成功算法的三无人机版本
直接复制Q2的成功模式到三无人机场景
"""

import numpy as np
import pandas as pd
import math
import time
import random

# 全局常量（完全复制Q2）
PHYSICS_GRAVITY_ACCELERATION = 9.8
MATHEMATICAL_EPSILON_THRESHOLD = 1e-15
TEMPORAL_STEP_SIZE_MACRO = 0.1
TEMPORAL_STEP_SIZE_MICRO = 0.005

# 目标参数（完全复制Q2）
primary_target_specifications = {
    "central_coordinates": np.array([0.0, 200.0, 0.0]),
    "radius_parameter": 7.0,
    "height_parameter": 10.0
}

# 烟幕参数（完全复制Q2）
smoke_countermeasure_parameters = {
    "effective_radius": 10.0,
    "descent_velocity": 3.0,
    "operational_duration": 20.0
}

# 导弹参数（完全复制Q2）
hostile_missile_configuration = {
    "launch_position": np.array([20000.0, 0.0, 2000.0]),
    "flight_velocity": 300.0
}

# 三架无人机初始位置
uav_positions = {
    1: np.array([17800.0, 0.0, 1800.0]),  # FY1
    2: np.array([12000.0, 1400.0, 1400.0]),  # FY2
    3: np.array([6000.0, -3000.0, 700.0])  # FY3
}

# 预计算导弹轨迹参数（完全复制Q2）
decoy_target_position = np.array([0.0, 0.0, 0.0])
missile_trajectory_vector = (decoy_target_position - hostile_missile_configuration["launch_position"])
missile_flight_distance = np.linalg.norm(missile_trajectory_vector)
missile_direction_unit_vector = missile_trajectory_vector / missile_flight_distance
missile_total_flight_time = missile_flight_distance / hostile_missile_configuration["flight_velocity"]

# 生成目标采样点（完全复制Q2的逻辑）
def generate_comprehensive_threat_volume_discretization(threat_configuration):
    """完全复制Q2的目标采样算法"""
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
    
    return np.array(sampling_vertices)

# 生成目标采样网格
target_discretization_mesh = generate_comprehensive_threat_volume_discretization(primary_target_specifications)

# 线段球体相交算法（完全复制Q2）
def analyze_line_segment_sphere_intersection_geometry(line_start_point, line_end_point, sphere_center_coordinates, sphere_radius_parameter):
    """完全复制Q2的相交检测算法"""
    line_start_point = np.array(line_start_point, dtype=np.float64)
    line_end_point = np.array(line_end_point, dtype=np.float64)
    sphere_center_coordinates = np.array(sphere_center_coordinates, dtype=np.float64)
    
    line_direction_vector = line_end_point - line_start_point
    line_to_sphere_vector = line_start_point - sphere_center_coordinates
    
    quadratic_coefficient_a = np.dot(line_direction_vector, line_direction_vector)
    if quadratic_coefficient_a < MATHEMATICAL_EPSILON_THRESHOLD:
        point_to_center_distance = np.linalg.norm(line_start_point - sphere_center_coordinates)
        return 1.0 if point_to_center_distance <= sphere_radius_parameter else 0.0
    
    quadratic_coefficient_b = 2 * np.dot(line_to_sphere_vector, line_direction_vector)
    quadratic_coefficient_c = np.dot(line_to_sphere_vector, line_to_sphere_vector) - sphere_radius_parameter ** 2
    discriminant_value = quadratic_coefficient_b ** 2 - 4 * quadratic_coefficient_a * quadratic_coefficient_c
    
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

# 目标遮蔽评估（完全复制Q2）
def evaluate_comprehensive_target_occlusion_status(interceptor_coordinates, obscurant_centroid, obscurant_radius, target_mesh_vertices):
    """完全复制Q2的遮蔽评估算法"""
    for mesh_vertex in target_mesh_vertices:
        intersection_ratio = analyze_line_segment_sphere_intersection_geometry(
            interceptor_coordinates, mesh_vertex, obscurant_centroid, obscurant_radius
        )
        if intersection_ratio < MATHEMATICAL_EPSILON_THRESHOLD:
            return False
    return True

# 时间序列构建（完全复制Q2）
def construct_multi_resolution_temporal_sequence(sequence_start, sequence_end, critical_event_timestamp=None):
    """完全复制Q2的时间序列构建算法"""
    if critical_event_timestamp is None:
        return np.arange(sequence_start, sequence_end + TEMPORAL_STEP_SIZE_MACRO, 
                       TEMPORAL_STEP_SIZE_MACRO)
    
    # 关键事件周围的高分辨率窗口
    high_resolution_window_start = max(sequence_start, critical_event_timestamp - 1.0)
    high_resolution_window_end = min(sequence_end, critical_event_timestamp + 1.0)
    
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

# 单无人机战术效能评估（基于Q2，适配多无人机）
def evaluate_single_drone_effectiveness(drone_id, strategic_parameter_vector):
    """基于Q2逻辑的单无人机效能评估"""
    flight_azimuth, platform_velocity, deployment_temporal_offset, detonation_temporal_offset = strategic_parameter_vector
    
    # 参数约束验证（复制Q2逻辑）
    if not (70.0 <= platform_velocity <= 140.0):
        return 0.0
    if deployment_temporal_offset < 0 or detonation_temporal_offset < 0:
        return 0.0
    
    # 使用对应无人机的初始位置
    uav_initial_position_vector = uav_positions[drone_id]
    
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
    
    temporal_analysis_sequence = construct_multi_resolution_temporal_sequence(
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
            is_target_occluded = evaluate_comprehensive_target_occlusion_status(
                interceptor_current_coordinates, current_obscurant_centroid, 
                smoke_countermeasure_parameters["effective_radius"], 
                target_discretization_mesh
            )
            
            if is_target_occluded:
                accumulated_occlusion_duration += temporal_increment
        
        previous_timestamp = current_timestamp
    
    return accumulated_occlusion_duration

# 简单网格搜索优化
def simple_grid_search_for_drone(drone_id):
    """为单个无人机进行简单网格搜索"""
    print(f"为无人机FY{drone_id}进行网格搜索...")
    
    best_score = 0.0
    best_params = None
    
    # 根据无人机位置调整参数范围
    if drone_id == 1:
        theta_values = np.linspace(-0.05, 0.05, 5)
        v_values = np.linspace(90, 110, 5)
        td_values = np.linspace(160, 180, 5)
        tau_values = np.linspace(15, 25, 5)
    elif drone_id == 2:
        theta_values = np.linspace(-0.1, 0.1, 5)
        v_values = np.linspace(80, 120, 5)
        td_values = np.linspace(100, 130, 5)
        tau_values = np.linspace(10, 20, 5)
    else:  # drone_id == 3
        theta_values = np.linspace(-0.2, 0.2, 5)
        v_values = np.linspace(70, 110, 5)
        td_values = np.linspace(40, 80, 5)
        tau_values = np.linspace(5, 15, 5)
    
    total_combinations = len(theta_values) * len(v_values) * len(td_values) * len(tau_values)
    count = 0
    
    for theta in theta_values:
        for v in v_values:
            for td in td_values:
                for tau in tau_values:
                    count += 1
                    if count % 50 == 0:
                        print(f"  进度: {count}/{total_combinations}")
                    
                    params = [theta, v, td, tau]
                    score = evaluate_single_drone_effectiveness(drone_id, params)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        print(f"  FY{drone_id}新最优: {score:.3f}s, 参数: θ={math.degrees(theta):.1f}°, v={v:.1f}, td={td:.1f}, tau={tau:.1f}")
    
    return best_params, best_score

# 主程序
def main():
    print("="*60)
    print("问题4: 基于Q2成功算法的三无人机协同优化")
    print("="*60)
    
    start_time = time.time()
    
    # 为每架无人机单独优化
    drone_results = {}
    
    for drone_id in [1, 2, 3]:
        print(f"\n优化无人机FY{drone_id}...")
        best_params, best_score = simple_grid_search_for_drone(drone_id)
        drone_results[drone_id] = {
            'params': best_params,
            'score': best_score
        }
        print(f"FY{drone_id}最终结果: {best_score:.3f}s")
    
    # 计算总遮蔽时间（简单相加，忽略重叠）
    total_score = sum(result['score'] for result in drone_results.values())
    
    # 保存结果
    data = []
    for drone_id, result in drone_results.items():
        if result['params'] is not None:
            theta, v, td, tau = result['params']
            data.append([
                f"FY{drone_id}",
                f"{math.degrees(theta):.2f}°",
                f"{v:.1f}",
                f"{td:.1f}",
                f"{tau:.1f}",
                f"{result['score']:.3f}s"
            ])
    
    data.append(["总计", "", "", "", "", f"{total_score:.3f}s"])
    
    df = pd.DataFrame(data, columns=["无人机", "方向角", "速度(m/s)", "投放延迟(s)", "起爆延迟(s)", "遮蔽时间"])
    
    try:
        df.to_excel("q4_q2_style_result.xlsx", index=False, engine="openpyxl")
        print(f"\n结果已保存到 q4_q2_style_result.xlsx")
    except:
        df.to_csv("q4_q2_style_result.csv", index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到 q4_q2_style_result.csv")
    
    print("\n" + "="*60)
    print(f"总遮蔽时间: {total_score:.3f}秒")
    if total_score >= 10.0:
        print("✓ 满足10秒以上要求")
    else:
        print("✗ 未达到10秒目标")
    
    elapsed = time.time() - start_time
    print(f"计算耗时: {elapsed:.1f}秒")
    print("="*60)

if __name__ == "__main__":
    main()
