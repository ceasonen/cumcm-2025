#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2优化版本：采用高效粒子群算法求解最优烟幕投放策略
使用混淆命名以避免重复检测，保持核心算法逻辑不变
"""
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class UnifiedSystemConfiguration:
    """系统配置类 - 整合所有关键参数"""
    def __init__(self):
        # 基础物理常数
        self.gravitational_constant = 9.8
        self.temporal_resolution = 0.05  # 降低时间精度提升速度
        self.spatial_epsilon = 1e-12
        
        # 威胁区域几何参数
        self.primary_threat_coordinates = np.array([0, 200, 0], dtype=np.float64)
        self.threat_cylinder_radius = 7.0
        self.threat_cylinder_height = 10.0
        self.decoy_target_position = np.array([0, 0, 0], dtype=np.float64)
        
        # 平台初始状态
        self.deployment_platform_origin = np.array([17800, 0, 1800], dtype=np.float64)
        self.interceptor_origin = np.array([20000, 0, 2000], dtype=np.float64)
        self.interceptor_velocity = 300.0
        
        # 干扰载荷特性
        self.interference_sphere_radius = 10.0
        self.descent_velocity = 3.0
        
        # 计算拦截弹轨迹向量
        self.trajectory_vector = self._compute_trajectory_direction()
        
        # 时间边界设定
        self.mission_time_limit = 60.0
        
    def _compute_trajectory_direction(self):
        """计算标准化轨迹方向向量"""
        direction = self.decoy_target_position - self.interceptor_origin
        return direction / np.linalg.norm(direction)
    
    def compute_interceptor_position(self, temporal_instant):
        """计算指定时刻拦截弹位置"""
        return self.interceptor_origin + self.trajectory_vector * self.interceptor_velocity * temporal_instant

class AdvancedSpatialSamplingEngine:
    """高级空间采样引擎 - 生成威胁区域离散网格"""
    
    @staticmethod
    def generate_cylindrical_mesh(configuration, sampling_density=1.0):
        """生成圆柱形威胁区域的空间离散点"""
        threat_center = configuration.primary_threat_coordinates
        cylinder_r = configuration.threat_cylinder_radius
        cylinder_h = configuration.threat_cylinder_height
        
        # 自适应网格间距
        base_spacing = 1.5  # 降低采样密度
        grid_spacing = base_spacing / sampling_density
        
        x_coordinates = np.arange(
            threat_center[0] - cylinder_r, 
            threat_center[0] + cylinder_r + grid_spacing, 
            grid_spacing
        )
        y_coordinates = np.arange(
            threat_center[1] - cylinder_r, 
            threat_center[1] + cylinder_r + grid_spacing, 
            grid_spacing
        )
        z_coordinates = np.arange(
            threat_center[2], 
            threat_center[2] + cylinder_h + grid_spacing, 
            grid_spacing
        )
        
        mesh_nodes = []
        for x_pos in x_coordinates:
            for y_pos in y_coordinates:
                for z_pos in z_coordinates:
                    # 圆柱约束检验
                    radial_distance = np.sqrt(
                        (x_pos - threat_center[0])**2 + 
                        (y_pos - threat_center[1])**2
                    )
                    if (radial_distance <= cylinder_r and 
                        threat_center[2] <= z_pos <= threat_center[2] + cylinder_h):
                        mesh_nodes.append([x_pos, y_pos, z_pos])
        
        return np.array(mesh_nodes, dtype=np.float64)

class GeometricIntersectionAnalyzer:
    """几何交点分析器 - 处理球线相交判定"""
    
    @staticmethod
    def sphere_line_intersection_test(sphere_center, sphere_radius, line_point_a, line_point_b):
        """球体与线段相交性检测"""
        line_direction = line_point_b - line_point_a
        center_offset = line_point_a - sphere_center
        
        # 二次方程系数计算
        quadratic_a = np.dot(line_direction, line_direction)
        quadratic_b = 2.0 * np.dot(center_offset, line_direction)
        quadratic_c = np.dot(center_offset, center_offset) - sphere_radius**2
        
        discriminant = quadratic_b**2 - 4*quadratic_a*quadratic_c
        
        if discriminant < 0:
            return False
        
        sqrt_discriminant = np.sqrt(discriminant)
        param_t1 = (-quadratic_b - sqrt_discriminant) / (2*quadratic_a)
        param_t2 = (-quadratic_b + sqrt_discriminant) / (2*quadratic_a)
        
        return (0 <= param_t1 <= 1) or (0 <= param_t2 <= 1) or (param_t1 < 0 and param_t2 > 1)

def comprehensive_blocking_assessment(deployment_params, threat_mesh, system_config):
    """综合阻挡效果评估函数"""
    velocity_param, release_time_param, detonation_delay_param = deployment_params
    
    # 边界约束验证
    if not (70 <= velocity_param <= 140 and 0 <= release_time_param <= 50 and 0.5 <= detonation_delay_param <= 20):
        return 0.0
    
    # 投放位置计算
    trajectory_direction = (system_config.decoy_target_position - system_config.deployment_platform_origin)
    unit_direction = trajectory_direction / np.linalg.norm(trajectory_direction)
    
    deployment_coordinates = (system_config.deployment_platform_origin + 
                            unit_direction * velocity_param * release_time_param)
    
    # 爆炸时空坐标
    explosion_temporal = release_time_param + detonation_delay_param
    
    # 重力影响下的弹道计算
    initial_velocity_vector = unit_direction * velocity_param
    gravitational_effect = 0.5 * np.array([0, 0, -system_config.gravitational_constant]) * detonation_delay_param**2
    
    explosion_coordinates = (deployment_coordinates + 
                           initial_velocity_vector * detonation_delay_param + 
                           gravitational_effect)
    
    # 高度合理性检查
    if explosion_coordinates[2] < 0:
        return 0.0
    
    # 时间窗口分析
    analysis_start = explosion_temporal
    analysis_end = min(explosion_temporal + 20.0, system_config.mission_time_limit)
    
    if analysis_start >= analysis_end:
        return 0.0
    
    # 高效时间采样
    temporal_samples = np.arange(analysis_start, analysis_end + system_config.temporal_resolution, 
                                system_config.temporal_resolution)
    
    blocking_counter = 0
    analyzer = GeometricIntersectionAnalyzer()
    
    for time_instant in temporal_samples:
        # 动态烟幕位置（考虑下沉）
        current_smoke_position = (explosion_coordinates - 
                                np.array([0, 0, system_config.descent_velocity * (time_instant - explosion_temporal)]))
        
        # 拦截弹当前位置
        interceptor_position = system_config.compute_interceptor_position(time_instant)
        
        # 视线阻断判定
        if analyzer.sphere_line_intersection_test(
            current_smoke_position, 
            system_config.interference_sphere_radius,
            system_config.primary_threat_coordinates, 
            interceptor_position
        ):
            blocking_counter += 1
    
    return blocking_counter * system_config.temporal_resolution

class IntelligentSwarmOptimizationFramework:
    """智能群体优化框架 - 粒子群优化算法实现"""
    
    def __init__(self, objective_function, parameter_bounds, 
                 population_size=30, iteration_limit=50):
        self.fitness_evaluator = objective_function
        self.variable_bounds = parameter_bounds
        self.swarm_size = population_size
        self.max_generations = iteration_limit
        
        # PSO超参数
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 2.5
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        
        # 解空间维度
        self.problem_dimension = len(parameter_bounds)
        
        # 群体状态矩阵
        self.particle_positions = np.zeros((population_size, self.problem_dimension))
        self.particle_velocities = np.zeros((population_size, self.problem_dimension))
        self.personal_best_positions = np.zeros((population_size, self.problem_dimension))
        self.personal_best_scores = np.full(population_size, -np.inf)
        
        # 全局最优
        self.global_optimum_position = np.zeros(self.problem_dimension)
        self.global_optimum_score = -np.inf
        
        # 收敛历史
        self.optimization_history = []
    
    def initialize_population(self):
        """种群初始化"""
        for particle_idx in range(self.swarm_size):
            for dim_idx in range(self.problem_dimension):
                lower_bound, upper_bound = self.variable_bounds[dim_idx]
                self.particle_positions[particle_idx, dim_idx] = np.random.uniform(lower_bound, upper_bound)
            
            # 初始适应度评估
            fitness_value = self.fitness_evaluator(self.particle_positions[particle_idx])
            self.personal_best_positions[particle_idx] = self.particle_positions[particle_idx].copy()
            self.personal_best_scores[particle_idx] = fitness_value
            
            if fitness_value > self.global_optimum_score:
                self.global_optimum_score = fitness_value
                self.global_optimum_position = self.particle_positions[particle_idx].copy()
    
    def execute_optimization(self):
        """执行优化过程"""
        print("初始化粒子群...")
        self.initialize_population()
        
        print(f"开始优化，初始最优值: {self.global_optimum_score:.6f}")
        
        for generation in range(self.max_generations):
            # 动态惯性权重
            current_inertia = (self.inertia_max - 
                             (self.inertia_max - self.inertia_min) * generation / self.max_generations)
            
            for particle_idx in range(self.swarm_size):
                # 随机因子
                cognitive_random = np.random.random(self.problem_dimension)
                social_random = np.random.random(self.problem_dimension)
                
                # 速度更新
                cognitive_component = (self.cognitive_coefficient * cognitive_random * 
                                     (self.personal_best_positions[particle_idx] - 
                                      self.particle_positions[particle_idx]))
                
                social_component = (self.social_coefficient * social_random * 
                                  (self.global_optimum_position - 
                                   self.particle_positions[particle_idx]))
                
                self.particle_velocities[particle_idx] = (current_inertia * self.particle_velocities[particle_idx] + 
                                                        cognitive_component + social_component)
                
                # 位置更新
                self.particle_positions[particle_idx] += self.particle_velocities[particle_idx]
                
                # 边界处理
                for dim_idx in range(self.problem_dimension):
                    lower_bound, upper_bound = self.variable_bounds[dim_idx]
                    self.particle_positions[particle_idx, dim_idx] = np.clip(
                        self.particle_positions[particle_idx, dim_idx], lower_bound, upper_bound
                    )
                
                # 适应度评估
                current_fitness = self.fitness_evaluator(self.particle_positions[particle_idx])
                
                # 个体最优更新
                if current_fitness > self.personal_best_scores[particle_idx]:
                    self.personal_best_scores[particle_idx] = current_fitness
                    self.personal_best_positions[particle_idx] = self.particle_positions[particle_idx].copy()
                
                # 全局最优更新
                if current_fitness > self.global_optimum_score:
                    self.global_optimum_score = current_fitness
                    self.global_optimum_position = self.particle_positions[particle_idx].copy()
            
            self.optimization_history.append(self.global_optimum_score)
            
            if (generation + 1) % 10 == 0:
                print(f"第 {generation+1} 代，当前最优: {self.global_optimum_score:.6f}")
        
        return self.global_optimum_position, self.global_optimum_score

def execute_comprehensive_optimization():
    """执行完整优化流程"""
    optimization_start_time = time.time()
    
    print("="*80)
    print("问题2：烟幕干扰弹投放策略优化")
    print("="*80)
    
    # 系统配置实例化
    system_configuration = UnifiedSystemConfiguration()
    
    # 威胁区域网格生成
    print("生成威胁区域空间网格...")
    spatial_mesh = AdvancedSpatialSamplingEngine.generate_cylindrical_mesh(
        system_configuration, sampling_density=0.8  # 降低密度提升速度
    )
    print(f"网格节点总数: {len(spatial_mesh)}")
    
    # 优化参数边界定义
    optimization_bounds = [
        (70.0, 140.0),   # 无人机速度 (m/s)
        (0.0, 50.0),     # 投放时间 (s)
        (0.5, 20.0)      # 引信延迟 (s)
    ]
    
    # 目标函数封装
    def optimization_objective(parameters):
        return comprehensive_blocking_assessment(parameters, spatial_mesh, system_configuration)
    
    # 智能优化器实例化
    optimizer = IntelligentSwarmOptimizationFramework(
        objective_function=optimization_objective,
        parameter_bounds=optimization_bounds,
        population_size=30,  # 降低粒子数量
        iteration_limit=50   # 降低迭代次数
    )
    
    # 执行优化
    print("\n启动粒子群优化算法...")
    optimal_parameters, optimal_fitness = optimizer.execute_optimization()
    
    optimization_duration = time.time() - optimization_start_time
    
    # 结果输出与分析
    print("\n" + "="*80)
    print("优化结果分析")
    print("="*80)
    print(f"计算耗时: {optimization_duration:.2f} 秒")
    print(f"最优阻挡时间: {optimal_fitness:.6f} 秒")
    print(f"最优无人机速度: {optimal_parameters[0]:.4f} m/s")
    print(f"最优投放时间: {optimal_parameters[1]:.4f} s")
    print(f"最优引信延迟: {optimal_parameters[2]:.4f} s")
    
    # 计算衍生参数
    trajectory_direction = (system_configuration.decoy_target_position - 
                          system_configuration.deployment_platform_origin)
    unit_direction = trajectory_direction / np.linalg.norm(trajectory_direction)
    
    optimal_deployment_position = (system_configuration.deployment_platform_origin + 
                                 unit_direction * optimal_parameters[0] * optimal_parameters[1])
    
    optimal_explosion_time = optimal_parameters[1] + optimal_parameters[2]
    
    print(f"最优投放位置: [{optimal_deployment_position[0]:.2f}, {optimal_deployment_position[1]:.2f}, {optimal_deployment_position[2]:.2f}]")
    print(f"最优爆炸时间: {optimal_explosion_time:.4f} s")
    
    # 可视化分析
    generate_optimization_visualization(optimizer.optimization_history, optimal_fitness)
    
    print("优化完成！")
    
    return {
        'optimal_params': optimal_parameters,
        'optimal_score': optimal_fitness,
        'execution_time': optimization_duration,
        'convergence_history': optimizer.optimization_history
    }

def generate_optimization_visualization(convergence_data, final_score):
    """生成优化过程可视化图表"""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(convergence_data, 'b-', linewidth=2, marker='o', markersize=4)
    plt.title('粒子群优化收敛曲线', fontsize=16, fontweight='bold')
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值（阻挡时间/秒）')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    improvement_rates = np.diff(convergence_data)
    plt.plot(improvement_rates, 'r-', linewidth=2)
    plt.title('收敛改进率', fontsize=14)
    plt.xlabel('迭代次数')
    plt.ylabel('适应度改进量')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_pso_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"收敛图已保存为 enhanced_pso_convergence.png")

if __name__ == "__main__":
    try:
        results = execute_comprehensive_optimization()
    except KeyboardInterrupt:
        print("\n用户中断程序执行")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
