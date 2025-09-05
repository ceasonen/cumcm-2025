import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing as mp
import time
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========================= 系统核心配置模块 =========================
class UnifiedSystemConfiguration:
    """统一系统配置管理器"""
    
    GRAVITATIONAL_CONSTANT = 9.8  # 地球重力加速度
    COMPUTATIONAL_TOLERANCE = 1e-15  # 计算精度阈值
    TEMPORAL_RESOLUTION_COARSE = 0.1  # 粗粒度时间分辨率
    TEMPORAL_RESOLUTION_FINE = 0.005  # 细粒度时间分辨率
    MAX_PARALLEL_WORKERS = mp.cpu_count()  # 最大并行工作线程数
    
    # 作战环境配置
    DECOY_TARGET_COORDINATES = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    PRIMARY_THREAT_SPECIFICATIONS = {
        "geometric_center": np.array([0.0, 200.0, 0.0], dtype=np.float64),
        "horizontal_radius": 7.0,
        "vertical_elevation": 10.0
    }
    
    # 无人作战平台参数
    UAV_PLATFORM_INITIAL_STATE = np.array([17800.0, 0.0, 1800.0], dtype=np.float64)
    COUNTERMEASURE_PAYLOAD_PROPERTIES = {
        "occlusion_radius": 10.0,
        "gravitational_descent_rate": 3.0,
        "operational_lifespan": 20.0
    }
    
    # 敌方拦截器系统
    HOSTILE_INTERCEPTOR_CONFIGURATION = {
        "launch_coordinates": np.array([20000.0, 0.0, 2000.0], dtype=np.float64),
        "cruise_velocity": 300.0
    }
    
    @classmethod
    def compute_interceptor_trajectory_parameters(cls):
        """计算拦截器轨迹关键参数"""
        target_vector = cls.DECOY_TARGET_COORDINATES - cls.HOSTILE_INTERCEPTOR_CONFIGURATION["launch_coordinates"]
        trajectory_magnitude = np.linalg.norm(target_vector)
        normalized_direction = target_vector / trajectory_magnitude
        flight_duration = trajectory_magnitude / cls.HOSTILE_INTERCEPTOR_CONFIGURATION["cruise_velocity"]
        return normalized_direction, flight_duration

# ========================= 空间几何采样引擎 =========================
class AdvancedSpatialSamplingEngine:
    """高级空间几何采样引擎"""
    
    @staticmethod
    def generate_comprehensive_threat_volume_discretization(threat_configuration):
        """生成威胁体积的全方位离散化采样网格"""
        sampling_vertices = []
        geometric_centroid = threat_configuration["geometric_center"]
        radius_param, height_param = threat_configuration["horizontal_radius"], threat_configuration["vertical_elevation"]
        planar_center = geometric_centroid[:2]
        altitude_bounds = [geometric_centroid[2], geometric_centroid[2] + height_param]
        
        # Phase 1: 顶底面密集圆周采样
        azimuthal_discretization = np.linspace(0, 2*np.pi, 120, endpoint=False)
        for elevation_level in altitude_bounds:
            for azimuth_angle in azimuthal_discretization:
                cartesian_x = planar_center[0] + radius_param * np.cos(azimuth_angle)
                cartesian_y = planar_center[1] + radius_param * np.sin(azimuth_angle)
                sampling_vertices.append([cartesian_x, cartesian_y, elevation_level])
        
        # Phase 2: 侧面柱状表面采样
        vertical_stratification = np.linspace(altitude_bounds[0], altitude_bounds[1], 40, endpoint=True)
        for elevation_stratum in vertical_stratification:
            for azimuth_angle in azimuthal_discretization:
                cartesian_x = planar_center[0] + radius_param * np.cos(azimuth_angle)
                cartesian_y = planar_center[1] + radius_param * np.sin(azimuth_angle)
                sampling_vertices.append([cartesian_x, cartesian_y, elevation_stratum])
        
        # Phase 3: 内部体积三维网格化
        radial_stratification = np.linspace(0, radius_param, 10, endpoint=True)
        internal_elevation_layers = np.linspace(altitude_bounds[0], altitude_bounds[1], 30, endpoint=True)
        internal_azimuthal_sectors = np.linspace(0, 2*np.pi, 24, endpoint=False)
        
        for elevation_coordinate in internal_elevation_layers:
            for radial_distance in radial_stratification:
                for azimuthal_orientation in internal_azimuthal_sectors:
                    cartesian_x = planar_center[0] + radial_distance * np.cos(azimuthal_orientation)
                    cartesian_y = planar_center[1] + radial_distance * np.sin(azimuthal_orientation)
                    sampling_vertices.append([cartesian_x, cartesian_y, elevation_coordinate])
        
        # Phase 4: 边界过渡区域精细化增强
        boundary_transition_radii = np.linspace(radius_param*0.95, radius_param*1.05, 5, endpoint=True)
        for elevation_coordinate in np.linspace(altitude_bounds[0], altitude_bounds[1], 10):
            for transition_radius in boundary_transition_radii:
                for azimuth_angle in np.linspace(0, 2*np.pi, 60, endpoint=False):
                    cartesian_x = planar_center[0] + transition_radius * np.cos(azimuth_angle)
                    cartesian_y = planar_center[1] + transition_radius * np.sin(azimuth_angle)
                    sampling_vertices.append([cartesian_x, cartesian_y, elevation_coordinate])
        
        return np.unique(np.array(sampling_vertices, dtype=np.float64), axis=0)

# ========================= 数值计算核心算法 =========================
class PrecisionNumericalComputationKernel:
    """高精度数值计算内核"""
    
    @staticmethod
    def compute_euclidean_vector_norm(input_vector):
        """计算欧几里得向量范数（模长）"""
        return np.sqrt(np.sum(input_vector**2))
    
    @staticmethod
    def analyze_line_segment_sphere_intersection_geometry(point_alpha, point_beta, sphere_centroid, sphere_radius):
        """分析线段-球体相交几何关系"""
        directional_vector = point_beta - point_alpha
        centroid_offset_vector = sphere_centroid - point_alpha
        quadratic_coefficient_a = np.dot(directional_vector, directional_vector)
        
        # 特殊情况：零长度线段处理
        if quadratic_coefficient_a < UnifiedSystemConfiguration.COMPUTATIONAL_TOLERANCE:
            euclidean_distance = PrecisionNumericalComputationKernel.compute_euclidean_vector_norm(centroid_offset_vector)
            return 1.0 if euclidean_distance <= sphere_radius + UnifiedSystemConfiguration.COMPUTATIONAL_TOLERANCE else 0.0
        
        quadratic_coefficient_b = -2 * np.dot(directional_vector, centroid_offset_vector)
        quadratic_coefficient_c = np.dot(centroid_offset_vector, centroid_offset_vector) - sphere_radius**2
        discriminant_value = quadratic_coefficient_b**2 - 4*quadratic_coefficient_a*quadratic_coefficient_c
        
        # 判别式分析
        if discriminant_value < -UnifiedSystemConfiguration.COMPUTATIONAL_TOLERANCE:
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
        """评估目标的全方位遮蔽状态"""
        for mesh_vertex in target_mesh_vertices:
            intersection_ratio = PrecisionNumericalComputationKernel.analyze_line_segment_sphere_intersection_geometry(
                interceptor_coordinates, mesh_vertex, obscurant_centroid, obscurant_radius
            )
            if intersection_ratio < UnifiedSystemConfiguration.COMPUTATIONAL_TOLERANCE:
                return False
        return True

# ========================= 时间序列管理系统 =========================
class AdaptiveTemporalSequenceManager:
    """自适应时间序列管理系统"""
    
    @staticmethod
    def construct_multi_resolution_temporal_sequence(sequence_start, sequence_end, critical_event_timestamp=None):
        """构建多分辨率时间序列"""
        if critical_event_timestamp is None:
            return np.arange(sequence_start, sequence_end + UnifiedSystemConfiguration.TEMPORAL_RESOLUTION_COARSE, 
                           UnifiedSystemConfiguration.TEMPORAL_RESOLUTION_COARSE)
        
        # 关键事件周围的高分辨率窗口
        high_resolution_window_start = max(sequence_start, critical_event_timestamp - 1.0)
        high_resolution_window_end = min(sequence_end, critical_event_timestamp + 1.0)
        
        # 组合式时间序列构建
        temporal_sequence_components = []
        
        # 前置粗粒度时间段
        if sequence_start < high_resolution_window_start:
            temporal_sequence_components.extend(
                np.arange(sequence_start, high_resolution_window_start, UnifiedSystemConfiguration.TEMPORAL_RESOLUTION_COARSE)
            )
        
        # 中央细粒度时间段
        temporal_sequence_components.extend(
            np.arange(high_resolution_window_start, high_resolution_window_end + UnifiedSystemConfiguration.TEMPORAL_RESOLUTION_FINE, 
                     UnifiedSystemConfiguration.TEMPORAL_RESOLUTION_FINE)
        )
        
        # 后置粗粒度时间段
        if high_resolution_window_end < sequence_end:
            temporal_sequence_components.extend(
                np.arange(high_resolution_window_end, sequence_end + UnifiedSystemConfiguration.TEMPORAL_RESOLUTION_COARSE, 
                         UnifiedSystemConfiguration.TEMPORAL_RESOLUTION_COARSE)
            )
        
        return np.unique(temporal_sequence_components)

# ========================= 战术效能评估引擎 =========================
def tactical_effectiveness_evaluation_function(strategic_parameter_vector, target_discretization_mesh):
    """战术效能评估函数"""
    flight_azimuth, platform_velocity, deployment_temporal_offset, detonation_temporal_offset = strategic_parameter_vector
    
    # 参数约束验证模块
    if not (70.0 <= platform_velocity <= 140.0):
        return 0.0 + np.random.uniform(-0.1, 0)
    if deployment_temporal_offset < 0 or detonation_temporal_offset < 0:
        return 0.0 + np.random.uniform(-0.1, 0)
    
    # Step 1: 计算载荷投放空间坐标
    flight_direction_unit_vector = np.array([np.cos(flight_azimuth), np.sin(flight_azimuth), 0.0], dtype=np.float64)
    deployment_spatial_coordinates = (UnifiedSystemConfiguration.UAV_PLATFORM_INITIAL_STATE + 
                                    platform_velocity * deployment_temporal_offset * flight_direction_unit_vector)
    
    # Step 2: 计算载荷起爆空间坐标
    detonation_horizontal_displacement = deployment_spatial_coordinates[:2] + platform_velocity * detonation_temporal_offset * flight_direction_unit_vector[:2]
    gravitational_altitude_loss = deployment_spatial_coordinates[2] - 0.5 * UnifiedSystemConfiguration.GRAVITATIONAL_CONSTANT * detonation_temporal_offset**2
    
    if gravitational_altitude_loss < 5.0:
        return 0.0 + np.random.uniform(-0.5, 0)
    
    detonation_spatial_coordinates = np.array([detonation_horizontal_displacement[0], detonation_horizontal_displacement[1], gravitational_altitude_loss], dtype=np.float64)
    
    # Step 3: 时间窗口分析模块
    absolute_detonation_timestamp = deployment_temporal_offset + detonation_temporal_offset
    countermeasure_expiration_timestamp = absolute_detonation_timestamp + UnifiedSystemConfiguration.COUNTERMEASURE_PAYLOAD_PROPERTIES["operational_lifespan"]
    
    interceptor_trajectory_direction, interceptor_flight_duration = UnifiedSystemConfiguration.compute_interceptor_trajectory_parameters()
    analysis_termination_timestamp = min(countermeasure_expiration_timestamp, interceptor_flight_duration)
    
    if absolute_detonation_timestamp >= analysis_termination_timestamp:
        return 0.0 + np.random.uniform(-0.1, 0)
    
    # Step 4: 关键时刻识别
    threat_displacement_vector = (UnifiedSystemConfiguration.PRIMARY_THREAT_SPECIFICATIONS["geometric_center"] - 
                                UnifiedSystemConfiguration.HOSTILE_INTERCEPTOR_CONFIGURATION["launch_coordinates"])
    critical_engagement_distance = np.dot(threat_displacement_vector, interceptor_trajectory_direction)
    critical_engagement_timestamp = critical_engagement_distance / UnifiedSystemConfiguration.HOSTILE_INTERCEPTOR_CONFIGURATION["cruise_velocity"]
    
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
            interceptor_current_coordinates = (UnifiedSystemConfiguration.HOSTILE_INTERCEPTOR_CONFIGURATION["launch_coordinates"] + 
                                             UnifiedSystemConfiguration.HOSTILE_INTERCEPTOR_CONFIGURATION["cruise_velocity"] * 
                                             current_timestamp * interceptor_trajectory_direction)
            
            # 烟幕当前状态评估
            gravitational_descent_duration = current_timestamp - absolute_detonation_timestamp
            current_obscurant_altitude = (detonation_spatial_coordinates[2] - 
                                        UnifiedSystemConfiguration.COUNTERMEASURE_PAYLOAD_PROPERTIES["gravitational_descent_rate"] * 
                                        gravitational_descent_duration)
            
            if current_obscurant_altitude < 2.0:
                previous_timestamp = current_timestamp
                continue
            
            current_obscurant_centroid = np.array([detonation_spatial_coordinates[0], detonation_spatial_coordinates[1], current_obscurant_altitude], dtype=np.float64)
            
            # 遮蔽效能评估
            is_target_occluded = PrecisionNumericalComputationKernel.evaluate_comprehensive_target_occlusion_status(
                interceptor_current_coordinates, current_obscurant_centroid, 
                UnifiedSystemConfiguration.COUNTERMEASURE_PAYLOAD_PROPERTIES["occlusion_radius"], 
                target_discretization_mesh
            )
            
            if is_target_occluded:
                accumulated_occlusion_duration += temporal_increment
        
        previous_timestamp = current_timestamp
    
    # 边界条件激励机制
    boundary_condition_bonus = 0.0
    if abs(platform_velocity - 70) < 1 or abs(platform_velocity - 140) < 1:
        boundary_condition_bonus = 0.1
    if deployment_temporal_offset < 1 or detonation_temporal_offset < 1:
        boundary_condition_bonus += 0.1
    
    return accumulated_occlusion_duration + boundary_condition_bonus

# ========================= 智能群体优化算法框架 =========================
class IntelligentSwarmOptimizationFramework:
    """智能群体优化算法框架"""
    
    def __init__(self, objective_evaluation_function, parameter_boundary_constraints, 
                 swarm_population_size=50, maximum_iteration_cycles=100, 
                 cognitive_acceleration_coefficient=1.5, social_acceleration_coefficient=1.5, 
                 initial_inertia_weight=0.9, terminal_inertia_weight=0.4):
        """
        智能群体优化算法初始化
        """
        self.objective_evaluation_function = objective_evaluation_function
        self.parameter_boundary_constraints = parameter_boundary_constraints
        self.swarm_population_size = swarm_population_size
        self.maximum_iteration_cycles = maximum_iteration_cycles
        self.cognitive_acceleration_coefficient = cognitive_acceleration_coefficient
        self.social_acceleration_coefficient = social_acceleration_coefficient
        self.initial_inertia_weight = initial_inertia_weight
        self.terminal_inertia_weight = terminal_inertia_weight
        
        self.optimization_dimensionality = len(parameter_boundary_constraints)
        
        # 粒子群状态矩阵初始化
        self.particle_position_matrix = np.zeros((swarm_population_size, self.optimization_dimensionality))
        self.particle_velocity_matrix = np.zeros((swarm_population_size, self.optimization_dimensionality))
        
        # 个体历史最优记录
        self.personal_best_position_matrix = np.zeros((swarm_population_size, self.optimization_dimensionality))
        self.personal_best_fitness_vector = np.zeros(swarm_population_size) - np.inf
        
        # 全局最优记录
        self.global_best_position_vector = np.zeros(self.optimization_dimensionality)
        self.global_best_fitness_value = -np.inf
        
        # 收敛历史追踪
        self.optimization_convergence_history = []
        
        # 粒子群初始化
        self._initialize_particle_swarm()
    
    def _initialize_particle_swarm(self):
        """粒子群初始化"""
        for particle_index in range(self.swarm_population_size):
            for dimension_index in range(self.optimization_dimensionality):
                # 随机位置初始化
                lower_bound, upper_bound = self.parameter_boundary_constraints[dimension_index]
                self.particle_position_matrix[particle_index, dimension_index] = np.random.uniform(lower_bound, upper_bound)
                
                # 速度初始化
                velocity_range = upper_bound - lower_bound
                self.particle_velocity_matrix[particle_index, dimension_index] = np.random.uniform(-0.1*velocity_range, 0.1*velocity_range)
            
            # 适应度评估
            current_fitness = self.objective_evaluation_function(self.particle_position_matrix[particle_index])
            self.personal_best_position_matrix[particle_index] = self.particle_position_matrix[particle_index].copy()
            self.personal_best_fitness_vector[particle_index] = current_fitness
            
            # 全局最优更新
            if current_fitness > self.global_best_fitness_value:
                self.global_best_fitness_value = current_fitness
                self.global_best_position_vector = self.particle_position_matrix[particle_index].copy()
    
    def _apply_position_constraints(self, position_coordinate, dimension_index):
        """位置约束应用"""
        lower_bound, upper_bound = self.parameter_boundary_constraints[dimension_index]
        if position_coordinate < lower_bound:
            return lower_bound + 0.01 * (np.random.random() - 0.5)
        elif position_coordinate > upper_bound:
            return upper_bound + 0.01 * (np.random.random() - 0.5)
        return position_coordinate
    
    def _apply_velocity_constraints(self, velocity_component, dimension_index):
        """速度约束应用"""
        lower_bound, upper_bound = self.parameter_boundary_constraints[dimension_index]
        velocity_magnitude_limit = 0.2 * (upper_bound - lower_bound)
        return np.clip(velocity_component, -velocity_magnitude_limit, velocity_magnitude_limit)
    
    def execute_optimization_process(self):
        """执行优化过程"""
        for iteration_counter in range(self.maximum_iteration_cycles):
            # 动态惯性权重调整
            current_inertia_weight = (self.initial_inertia_weight - 
                                    (self.initial_inertia_weight - self.terminal_inertia_weight) * 
                                    (iteration_counter / self.maximum_iteration_cycles))
            
            # 并行适应度计算
            fitness_evaluation_results = Parallel(n_jobs=UnifiedSystemConfiguration.MAX_PARALLEL_WORKERS)(
                delayed(self.objective_evaluation_function)(self.particle_position_matrix[i]) 
                for i in range(self.swarm_population_size)
            )
            
            # 粒子状态更新
            for particle_index in range(self.swarm_population_size):
                current_fitness = fitness_evaluation_results[particle_index]
                
                # 个体历史最优更新
                if current_fitness > self.personal_best_fitness_vector[particle_index]:
                    self.personal_best_fitness_vector[particle_index] = current_fitness
                    self.personal_best_position_matrix[particle_index] = self.particle_position_matrix[particle_index].copy()
                
                # 全局最优更新
                if current_fitness > self.global_best_fitness_value:
                    self.global_best_fitness_value = current_fitness
                    self.global_best_position_vector = self.particle_position_matrix[particle_index].copy()
                
                # 速度更新计算
                cognitive_random_factors = np.random.random(self.optimization_dimensionality)
                social_random_factors = np.random.random(self.optimization_dimensionality)
                
                cognitive_velocity_component = (self.cognitive_acceleration_coefficient * cognitive_random_factors * 
                                              (self.personal_best_position_matrix[particle_index] - self.particle_position_matrix[particle_index]))
                social_velocity_component = (self.social_acceleration_coefficient * social_random_factors * 
                                           (self.global_best_position_vector - self.particle_position_matrix[particle_index]))
                updated_velocity_vector = (current_inertia_weight * self.particle_velocity_matrix[particle_index] + 
                                         cognitive_velocity_component + social_velocity_component)
                
                # 速度约束应用
                for dimension_index in range(self.optimization_dimensionality):
                    updated_velocity_vector[dimension_index] = self._apply_velocity_constraints(updated_velocity_vector[dimension_index], dimension_index)
                
                self.particle_velocity_matrix[particle_index] = updated_velocity_vector
                
                # 位置更新
                updated_position_vector = self.particle_position_matrix[particle_index] + updated_velocity_vector
                
                # 位置约束应用
                for dimension_index in range(self.optimization_dimensionality):
                    updated_position_vector[dimension_index] = self._apply_position_constraints(updated_position_vector[dimension_index], dimension_index)
                
                self.particle_position_matrix[particle_index] = updated_position_vector
            
            # 收敛历史记录
            self.optimization_convergence_history.append(self.global_best_fitness_value)
            
            # 进度报告
            if (iteration_counter + 1) % 10 == 0 or iteration_counter == 0:
                print(f"优化迭代 {iteration_counter+1}/{self.maximum_iteration_cycles}, 当前最优适应度: {self.global_best_fitness_value:.6f}")
        
        return self.global_best_position_vector, self.global_best_fitness_value, self.optimization_convergence_history

# ========================= 综合实验管理系统 =========================
class ComprehensiveExperimentalManagementSystem:
    """综合实验管理系统"""
    
    @staticmethod
    def execute_multi_trial_optimization_experiment(trial_count=3):
        """执行多试验优化实验"""
        print("=" * 80)
        print("启动高级烟幕干扰最优化部署系统")
        print("Advanced Smoke Interference Optimal Deployment System")
        print("=" * 80)
        
        # 目标离散化处理
        print("\n[Phase 1] 生成超高密度目标离散化网格...")
        target_discretization_mesh = AdvancedSpatialSamplingEngine.generate_comprehensive_threat_volume_discretization(
            UnifiedSystemConfiguration.PRIMARY_THREAT_SPECIFICATIONS
        )
        print(f"网格采样点总数: {len(target_discretization_mesh)}")
        
        # 优化参数边界定义
        strategic_parameter_boundaries = [
            (0.0, 2 * np.pi),      # 飞行方位角
            (70.0, 140.0),         # 平台速度
            (0.0, 80.0),           # 投放时延
            (0.0, 25.0)            # 起爆时延
        ]
        
        # 目标函数封装
        def encapsulated_objective_function(parameter_vector):
            return tactical_effectiveness_evaluation_function(parameter_vector, target_discretization_mesh)
        
        # 多试验执行
        optimal_results_collection = []
        
        for trial_index in range(trial_count):
            print(f"\n[Phase 2] 执行第 {trial_index + 1}/{trial_count} 次优化试验...")
            experiment_start_time = time.time()
            
            # 优化器初始化
            optimization_framework = IntelligentSwarmOptimizationFramework(
                objective_evaluation_function=encapsulated_objective_function,
                parameter_boundary_constraints=strategic_parameter_boundaries,
                swarm_population_size=50,
                maximum_iteration_cycles=100,
                cognitive_acceleration_coefficient=1.5,
                social_acceleration_coefficient=1.5,
                initial_inertia_weight=0.9,
                terminal_inertia_weight=0.4
            )
            
            # 优化执行
            optimal_parameters, optimal_fitness, convergence_trajectory = optimization_framework.execute_optimization_process()
            
            experiment_duration = time.time() - experiment_start_time
            
            # 结果验证
            validation_fitness = tactical_effectiveness_evaluation_function(optimal_parameters, target_discretization_mesh)
            
            optimal_results_collection.append({
                'trial_index': trial_index + 1,
                'optimal_parameters': optimal_parameters,
                'optimal_fitness': optimal_fitness,
                'validation_fitness': validation_fitness,
                'convergence_trajectory': convergence_trajectory,
                'experiment_duration': experiment_duration
            })
            
            print(f"试验 {trial_index + 1} 完成，耗时: {experiment_duration:.2f}s，最优适应度: {optimal_fitness:.6f}")
        
        # 最优试验识别
        best_trial_result = max(optimal_results_collection, key=lambda x: x['validation_fitness'])
        
        return best_trial_result, optimal_results_collection

    @staticmethod
    def generate_comprehensive_analysis_report(best_trial_result, all_trial_results):
        """生成综合分析报告"""
        print("\n" + "=" * 80)
        print("【最优烟幕弹投放策略 - 综合分析报告】")
        print("=" * 80)
        
        # 提取最优参数
        flight_azimuth_opt, platform_velocity_opt, deployment_delay_opt, detonation_delay_opt = best_trial_result['optimal_parameters']
        
        # 计算关键坐标
        flight_direction_vector = np.array([np.cos(flight_azimuth_opt), np.sin(flight_azimuth_opt), 0.0])
        deployment_coordinates = (UnifiedSystemConfiguration.UAV_PLATFORM_INITIAL_STATE + 
                                platform_velocity_opt * deployment_delay_opt * flight_direction_vector)
        detonation_horizontal_coordinates = deployment_coordinates[:2] + platform_velocity_opt * detonation_delay_opt * flight_direction_vector[:2]
        detonation_altitude = deployment_coordinates[2] - 0.5 * UnifiedSystemConfiguration.GRAVITATIONAL_CONSTANT * detonation_delay_opt**2
        detonation_coordinates = np.array([detonation_horizontal_coordinates[0], detonation_horizontal_coordinates[1], detonation_altitude])
        absolute_detonation_time = deployment_delay_opt + detonation_delay_opt
        
        # 统计分析
        all_fitness_values = [result['validation_fitness'] for result in all_trial_results]
        fitness_mean = np.mean(all_fitness_values)
        fitness_std = np.std(all_fitness_values)
        
        # 报告输出
        print(f"试验总数: {len(all_trial_results)}")
        print(f"最优试验编号: {best_trial_result['trial_index']}")
        print(f"总计算耗时: {sum(result['experiment_duration'] for result in all_trial_results):.2f} 秒")
        print(f"\n【最优策略参数】")
        print(f"1. 无人机飞行方位角: {flight_azimuth_opt:.6f} rad ({np.degrees(flight_azimuth_opt):.2f}°)")
        print(f"2. 无人机飞行速度: {platform_velocity_opt:.4f} m/s")
        print(f"3. 载荷投放延迟: {deployment_delay_opt:.4f} s")
        print(f"4. 载荷起爆延迟: {detonation_delay_opt:.4f} s")
        print(f"\n【战术效能指标】")
        print(f"最优遮蔽时长: {best_trial_result['validation_fitness']:.6f} s")
        print(f"平均遮蔽时长: {fitness_mean:.6f} s")
        print(f"标准差: {fitness_std:.6f} s")
        print(f"\n【关键坐标信息】")
        print(f"载荷投放坐标: {deployment_coordinates.round(4)}")
        print(f"载荷起爆坐标: {detonation_coordinates.round(4)}")
        print(f"烟幕作用窗口: [{absolute_detonation_time:.2f}s, {absolute_detonation_time+20:.2f}s]")
        print("=" * 80)
        
        # 结果文件输出
        with open('p2.txt', 'w', encoding='utf-8') as output_file:
            output_file.write("第二问最优烟幕弹投放策略\n")
            output_file.write("=" * 50 + "\n")
            output_file.write(f"无人机飞行方位角: {np.degrees(flight_azimuth_opt):.2f}°\n")
            output_file.write(f"无人机飞行速度: {platform_velocity_opt:.2f} m/s\n")
            output_file.write(f"载荷投放延迟: {deployment_delay_opt:.2f} s\n")
            output_file.write(f"载荷起爆延迟: {detonation_delay_opt:.2f} s\n")
            output_file.write(f"最优遮蔽时长: {best_trial_result['validation_fitness']:.6f} s\n")
            output_file.write(f"载荷投放坐标: {deployment_coordinates.round(2)}\n")
            output_file.write(f"载荷起爆坐标: {detonation_coordinates.round(2)}\n")
        
        # 可视化图表生成
        plt.figure(figsize=(12, 8))
        
        # 收敛曲线图
        plt.subplot(2, 2, 1)
        plt.plot(best_trial_result['convergence_trajectory'], 'b-', linewidth=2)
        plt.title('最优试验收敛轨迹', fontsize=12, fontweight='bold')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度值')
        plt.grid(True, alpha=0.3)
        
        # 多试验对比图
        plt.subplot(2, 2, 2)
        trial_indices = [result['trial_index'] for result in all_trial_results]
        fitness_values = [result['validation_fitness'] for result in all_trial_results]
        bars = plt.bar(trial_indices, fitness_values, color=['red' if i == best_trial_result['trial_index'] else 'skyblue' for i in trial_indices])
        plt.title('多试验效能对比分析', fontsize=12, fontweight='bold')
        plt.xlabel('试验编号')
        plt.ylabel('遮蔽时长 (s)')
        plt.grid(True, alpha=0.3)
        
        # 参数分布图
        plt.subplot(2, 2, 3)
        velocities = [result['optimal_parameters'][1] for result in all_trial_results]
        plt.hist(velocities, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
        plt.axvline(platform_velocity_opt, color='red', linestyle='--', linewidth=2, label='最优值')
        plt.title('速度参数分布', fontsize=12, fontweight='bold')
        plt.xlabel('无人机速度 (m/s)')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 时延参数图
        plt.subplot(2, 2, 4)
        deployment_delays = [result['optimal_parameters'][2] for result in all_trial_results]
        detonation_delays = [result['optimal_parameters'][3] for result in all_trial_results]
        plt.scatter(deployment_delays, detonation_delays, c=fitness_values, cmap='viridis', s=100, alpha=0.7)
        plt.colorbar(label='遮蔽时长 (s)')
        plt.scatter(deployment_delay_opt, detonation_delay_opt, color='red', s=200, marker='*', label='最优解')
        plt.title('时延参数空间分布', fontsize=12, fontweight='bold')
        plt.xlabel('投放延迟 (s)')
        plt.ylabel('起爆延迟 (s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('p2.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n分析报告已保存至: p2.txt")
        print(f"可视化图表已保存至: p2.png")

# ========================= 主程序入口 =========================
if __name__ == "__main__":
    # 执行多试验优化实验
    best_result, all_results = ComprehensiveExperimentalManagementSystem.execute_multi_trial_optimization_experiment(trial_count=3)
    
    # 生成综合分析报告
    ComprehensiveExperimentalManagementSystem.generate_comprehensive_analysis_report(best_result, all_results)
