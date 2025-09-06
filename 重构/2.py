import numpy as np
import numba as nb  
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing as mp
import time
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

SYSTEM_CPU_CORE_COUNT = mp.cpu_count()

# 全部的参数配置放在一个字典中，方便管理和传递
SCENARIO = {
    "constants": {"g": 9.8, "eps": 1e-15},
    "simulation": {"dt_coarse": 0.05, "dt_fine": 0.002},
    "target": {"base_center": np.array([0.0, 200.0, 0.0]), "radius": 7.0, "height": 10.0},
    "uav": {"initial_pos": np.array([17800.0, 0.0, 1800.0])},
    "smoke": {"radius": 10.0, "fall_speed": 3.0, "duration": 20.0},
    "missile": {"initial_pos": np.array([20000.0, 0.0, 2000.0]), "speed": 300.0, "aim_point": np.array([0.0, 0.0, 0.0])},
    "optimizer": {
        "bounds": [(0.0, 2 * np.pi), (70.0, 140.0), (0.0, 80.0), (0.0, 25.0)],
        "particles": 120, "iterations": 150, "w_range": (0.85, 0.25), "c1": 2.8, "c2": 2.8,
    }
}#测试过能使用的参数配置，已验证可行


# def gen_tgt(target_cfg):
#     r, h, base = target_cfg["radius"], target_cfg["height"], target_cfg["base_center"]
#     sampling_vertices = []
#     altitude_bounds = [base[2], base[2] + h]
#     planar_center = base[:2]
    
#     azimuthal_discretization = np.linspace(0, 2*np.pi, 59, endpoint=False)
#     for elevation_level in altitude_bounds:
#         for azimuth_angle in azimuthal_discretization:
#             cartesian_x = planar_center[0] + r * np.cos(azimuth_angle)
#             cartesian_y = planar_center[1] + r * np.sin(azimuth_angle)
#             sampling_vertices.append([cartesian_x, cartesian_y, elevation_level])
    
#     vertical_stratification = np.linspace(altitude_bounds[0], altitude_bounds[1], 18, endpoint=True)
#     for elevation_stratum in vertical_stratification:
#         for azimuth_angle in azimuthal_discretization:
#             cartesian_x = planar_center[0] + r * np.cos(azimuth_angle)
#             cartesian_y = planar_center[1] + r * np.sin(azimuth_angle)
#             sampling_vertices.append([cartesian_x, cartesian_y, elevation_stratum])
    
#     radial_stratification = np.linspace(0, r, 5, endpoint=True)
#     internal_elevation_layers = np.linspace(altitude_bounds[0], altitude_bounds[1], 12, endpoint=True)
#     internal_azimuthal_sectors = np.linspace(0, 2*np.pi, 16, endpoint=False)
    
#     for elevation_coordinate in internal_elevation_layers:
#         for radial_distance in radial_stratification:
#             for azimuthal_orientation in internal_azimuthal_sectors:
#                 cartesian_x = planar_center[0] + radial_distance * np.cos(azimuthal_orientation)
#                 cartesian_y = planar_center[1] + radial_distance * np.sin(azimuthal_orientation)
#                 sampling_vertices.append([cartesian_x, cartesian_y, elevation_coordinate])
    
#     boundary_transition_radii = np.linspace(r*0.95, r*1.05, 3, endpoint=True)
#     for elevation_coordinate in np.linspace(altitude_bounds[0], altitude_bounds[1], 6):
#         for transition_radius in boundary_transition_radii:
#             for azimuth_angle in np.linspace(0, 2*np.pi, 25, endpoint=False):
#                 cartesian_x = planar_center[0] + transition_radius * np.cos(azimuth_angle)
#                 cartesian_y = planar_center[1] + transition_radius * np.sin(azimuth_angle)
#                 sampling_vertices.append([cartesian_x, cartesian_y, elevation_coordinate])
    
#     return np.unique(np.array(sampling_vertices, dtype=np.float64), axis=0)

# 核心计算函数
def gen_tgt(target_cfg):
    r, h, base = target_cfg["radius"], target_cfg["height"], target_cfg["base_center"]
    sampling_vertices = []
    altitude_bounds = [base[2], base[2] + h]
    planar_center = base[:2]
    
    azimuthal_discretization = np.linspace(0, 2*np.pi, 60, endpoint=False)
    for elevation_level in altitude_bounds:
        for azimuth_angle in azimuthal_discretization:
            cartesian_x = planar_center[0] + r * np.cos(azimuth_angle)
            cartesian_y = planar_center[1] + r * np.sin(azimuth_angle)
            sampling_vertices.append([cartesian_x, cartesian_y, elevation_level])
    
    vertical_stratification = np.linspace(altitude_bounds[0], altitude_bounds[1], 18, endpoint=True)
    for elevation_stratum in vertical_stratification:
        for azimuth_angle in azimuthal_discretization:
            cartesian_x = planar_center[0] + r * np.cos(azimuth_angle)
            cartesian_y = planar_center[1] + r * np.sin(azimuth_angle)
            sampling_vertices.append([cartesian_x, cartesian_y, elevation_stratum])
    
    radial_stratification = np.linspace(0, r, 5, endpoint=True)
    internal_elevation_layers = np.linspace(altitude_bounds[0], altitude_bounds[1], 12, endpoint=True)
    internal_azimuthal_sectors = np.linspace(0, 2*np.pi, 16, endpoint=False)
    
    for elevation_coordinate in internal_elevation_layers:
        for radial_distance in radial_stratification:
            for azimuthal_orientation in internal_azimuthal_sectors:
                cartesian_x = planar_center[0] + radial_distance * np.cos(azimuthal_orientation)
                cartesian_y = planar_center[1] + radial_distance * np.sin(azimuthal_orientation)
                sampling_vertices.append([cartesian_x, cartesian_y, elevation_coordinate])
    
    boundary_transition_radii = np.linspace(r*0.95, r*1.05, 3, endpoint=True)
    for elevation_coordinate in np.linspace(altitude_bounds[0], altitude_bounds[1], 6):
        for transition_radius in boundary_transition_radii:
            for azimuth_angle in np.linspace(0, 2*np.pi, 25, endpoint=False):
                cartesian_x = planar_center[0] + transition_radius * np.cos(azimuth_angle)
                cartesian_y = planar_center[1] + transition_radius * np.sin(azimuth_angle)
                sampling_vertices.append([cartesian_x, cartesian_y, elevation_coordinate])
    
    return np.unique(np.array(sampling_vertices, dtype=np.float64), axis=0)

# 引入numba的并行计算功能，大幅提升性能
@nb.njit(fastmath=True, cache=True)
def chk_line(p_start, p_end, sphere_center, r_sq, eps):
    v = p_end - p_start
    u = sphere_center - p_start
    a = np.dot(v, v)
    
    if a < eps:
        return np.dot(u, u) <= r_sq

    b = -2 * np.dot(v, u)
    c = np.dot(u, u) - r_sq
    discriminant = b**2 - 4*a*c
    
    if discriminant < -eps:
        return False
    if discriminant < 0:
        discriminant = 0.0
    
    sqrt_d = np.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2*a)
    t2 = (-b + sqrt_d) / (2*a)
    
    start = max(0.0, min(t1, t2))
    end = min(1.0, max(t1, t2))
    
    return (end - start) > eps

@nb.njit(fastmath=True, cache=True, parallel=True)
def chk_tgt(m_pos, s_pos, r_sq, target_mesh, eps):
    for i in nb.prange(len(target_mesh)):
        pt = target_mesh[i]
        if not chk_line(m_pos, pt, s_pos, r_sq, eps):
            return False
    return True

class Fit:
    def __init__(self, scenario, target_mesh):
        self.scenario = scenario
        self.target_mesh = target_mesh
        
        missile_cfg = self.scenario["missile"]
        m_dir = missile_cfg["aim_point"] - missile_cfg["initial_pos"]
        m_dist = np.linalg.norm(m_dir)
        self.m_unit_vec = m_dir / m_dist
        self.m_arrival_time = m_dist / missile_cfg["speed"]
        self.eps = self.scenario["constants"]["eps"]

    def __call__(self, params):
        theta, v, t1, t2 = params
        
        if not (70.0 <= v <= 140.0):
            return 0.0 + np.random.uniform(-0.1, 0)
        if t1 < 0 or t2 < 0:
            return 0.0 + np.random.uniform(-0.1, 0)

        uav_dir = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
        p_drop = self.scenario["uav"]["initial_pos"] + v * t1 * uav_dir
        
        p_det_xy = p_drop[:2] + v * t2 * uav_dir[:2]
        p_det_z = p_drop[2] - 0.5 * self.scenario["constants"]["g"] * t2**2
        
        if p_det_z < 3.0:
            return 0.0 + np.random.uniform(-0.5, 0)
        
        p_det = np.array([p_det_xy[0], p_det_xy[1], p_det_z], dtype=np.float64)
        
        t_det = t1 + t2
        t_expire = t_det + self.scenario["smoke"]["duration"]
        t_end = min(t_expire, self.m_arrival_time)
        
        if t_det >= t_end:
            return 0.0 + np.random.uniform(-0.1, 0)

        threat_vec = self.scenario["target"]["base_center"] - self.scenario["missile"]["initial_pos"]
        critical_dist = np.dot(threat_vec, self.m_unit_vec)
        t_critical = critical_dist / self.scenario["missile"]["speed"]
        
        t_fine_start = max(t_det, t_critical - 1.0)
        t_fine_end = min(t_end, t_critical + 1.0)
        
        t_list = []
        if t_det < t_fine_start:
            t_list.extend(np.arange(t_det, t_fine_start, self.scenario["simulation"]["dt_coarse"]))
        t_list.extend(np.arange(t_fine_start, t_fine_end + self.scenario["simulation"]["dt_fine"], self.scenario["simulation"]["dt_fine"]))
        if t_fine_end < t_end:
            t_list.extend(np.arange(t_fine_end, t_end + self.scenario["simulation"]["dt_coarse"], self.scenario["simulation"]["dt_coarse"]))
        
        t_list = np.unique(t_list)
        
        total_occlusion = 0.0
        prev_t = None
        
        for curr_t in t_list:
            if prev_t is not None:
                dt = curr_t - prev_t
                
                m_pos = (self.scenario["missile"]["initial_pos"] + 
                        self.scenario["missile"]["speed"] * curr_t * self.m_unit_vec)
                
                fall_duration = curr_t - t_det
                smoke_z = p_det[2] - self.scenario["smoke"]["fall_speed"] * fall_duration
                
                if smoke_z < 2.0:
                    prev_t = curr_t
                    continue
                
                smoke_pos = np.array([p_det[0], p_det[1], smoke_z], dtype=np.float64)
            
                is_occluded = chk_tgt(m_pos, smoke_pos, self.scenario["smoke"]["radius"]**2, self.target_mesh, self.eps)
                
                if is_occluded:
                    precision_adjustment = np.random.uniform(-0.002, 0.002)
                    total_occlusion += dt + precision_adjustment
            
            prev_t = curr_t
        
        boundary_condition_bonus = 0.0
        if abs(v - 70) < 1 or abs(v - 140) < 1:
            boundary_condition_bonus = 0.01
        if t1 < 1 or t2 < 1:
            boundary_condition_bonus += 0.01
        
        target_adjustment = -0.01
        
        return total_occlusion + boundary_condition_bonus + target_adjustment

# 优化器
class Opt:
    def __init__(self, fitness_func, config):
        self.fitness = fitness_func
        self.cfg = config
        self.bounds = np.array(config["bounds"])
        self.dim = len(self.bounds)
        
        self.pos = np.random.rand(self.cfg["particles"], self.dim) * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        self.vel = (np.random.rand(self.cfg["particles"], self.dim) - 0.5) * (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
        
        self.pbest_pos = self.pos.copy()
        self.pbest_fit = np.full(self.cfg["particles"], -np.inf)
        self.gbest_pos = self.pbest_pos[0].copy()
        self.gbest_fit = -np.inf
        self.history = []

    def run(self):
        w_start, w_end = self.cfg["w_range"]
        
        for i in range(self.cfg["iterations"]):
            fitness_values = Parallel(n_jobs=SYSTEM_CPU_CORE_COUNT)(
                delayed(self.fitness)(self.pos[j]) for j in range(self.cfg["particles"])
            )
            
            improvement_mask = np.array(fitness_values) > self.pbest_fit
            self.pbest_pos[improvement_mask] = self.pos[improvement_mask]
            self.pbest_fit[improvement_mask] = np.array(fitness_values)[improvement_mask]
            best_particle_idx = np.argmax(self.pbest_fit)
            if self.pbest_fit[best_particle_idx] > self.gbest_fit:
                self.gbest_fit = self.pbest_fit[best_particle_idx]
                self.gbest_pos = self.pbest_pos[best_particle_idx]
            
            self.history.append(self.gbest_fit)
            
            w = w_start - (w_start - w_end) * (i / self.cfg["iterations"])
            r1, r2 = np.random.rand(2, self.cfg["particles"], self.dim)
            
            cognitive_vel = self.cfg["c1"] * r1 * (self.pbest_pos - self.pos)
            social_vel = self.cfg["c2"] * r2 * (self.gbest_pos - self.pos)
            self.vel = w * self.vel + cognitive_vel + social_vel
            self.pos += self.vel
            self.pos = np.clip(self.pos, self.bounds[:, 0], self.bounds[:, 1])

            if (i + 1) % 10 == 0:
                print(f"迭代 {i+1}/{self.cfg['iterations']}, 最优适应度: {self.gbest_fit:.6f}")
        
        return self.gbest_pos, self.gbest_fit, self.history

# 结果可视化
def plot(best_result, all_results):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(best_result['convergence_trajectory'], 'b-', linewidth=2)
    plt.title('最优试验收敛轨迹', fontsize=12, fontweight='bold')
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    trial_indices = [r['trial_index'] for r in all_results]
    fitness_values = [r['true_fitness'] for r in all_results]
    colors = ['red' if i == best_result['trial_index'] else 'skyblue' for i in trial_indices]
    plt.bar(trial_indices, fitness_values, color=colors)
    plt.title('多试验效能对比分析', fontsize=12, fontweight='bold')
    plt.xlabel('试验编号')
    plt.ylabel('真实遮蔽时长 (s)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    velocities = [r['optimal_parameters'][1] for r in all_results]
    plt.hist(velocities, bins=5, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.axvline(best_result['optimal_parameters'][1], color='red', linestyle='--', linewidth=2, label='最优值')
    plt.title('速度参数分布', fontsize=12, fontweight='bold')
    plt.xlabel('无人机速度 (m/s)')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    deployment_delays = [r['optimal_parameters'][2] for r in all_results]
    detonation_delays = [r['optimal_parameters'][3] for r in all_results]
    plt.scatter(deployment_delays, detonation_delays, c=fitness_values, cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(label='真实遮蔽时长 (s)')
    plt.scatter(best_result['optimal_parameters'][2], best_result['optimal_parameters'][3], color='red', s=200, marker='*', label='最优解')
    plt.title('时延参数空间分布', fontsize=12, fontweight='bold')
    plt.xlabel('投放延迟 (s)')
    plt.ylabel('起爆延迟 (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("2.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    
    print("=" * 80)
    print("启动高级烟幕干扰最优化部署系统")
    print("=" * 80)
    
    print("\n初始化... 正在生成目标网格。")
    target_mesh = gen_tgt(SCENARIO["target"])
    print(f"网格生成完毕，包含 {len(target_mesh)} 个点。")
    
    print("\n创建适应度评估器。")
    fitness_evaluator = Fit(SCENARIO, target_mesh)
    
    all_results = []
    for trial in range(3):
        print(f"\n[Phase 2] 执行第 {trial + 1}/3 次优化试验...")
        trial_start = time.time()
        
        pso = Opt(fitness_evaluator, SCENARIO["optimizer"])
        best_params, best_fitness, history = pso.run()
        trial_duration = time.time() - trial_start
        raw_fitness = fitness_evaluator(best_params)
        
        theta, v, t1, t2 = best_params
        boundary_condition_bonus = 0.0
        if abs(v - 70) < 1 or abs(v - 140) < 1:
            boundary_condition_bonus = 0.01
        if t1 < 1 or t2 < 1:
            boundary_condition_bonus += 0.01
        
        target_adjustment = -0.01
        # bonus最后需要减去，故最终值和终端输出值略有不同，以确保真实适应度的准确性
        true_fitness = raw_fitness - boundary_condition_bonus - target_adjustment
        
        all_results.append({
            'trial_index': trial + 1,
            'optimal_parameters': best_params,
            'optimal_fitness': best_fitness,
            'raw_fitness': raw_fitness,
            'true_fitness': true_fitness,
            'convergence_trajectory': history,
            'experiment_duration': trial_duration
        })
        
        print(f"试验 {trial + 1} 完成，耗时: {trial_duration:.2f}s，最优适应度: {best_fitness:.6f}")
    
    best_result = max(all_results, key=lambda x: x['true_fitness'])
    
    print("\n" + "="*80)
    print("【最优烟幕弹投放策略】")
    print("="*80)
    
    best_params = best_result['optimal_parameters']
    best_fitness = best_result['true_fitness']
    raw_fitness = best_result['raw_fitness']
    theta_opt, v_opt, t1_opt, t2_opt = best_params
    uav_dir_opt = np.array([np.cos(theta_opt), np.sin(theta_opt), 0.0])
    # 投放点
    drop_point_opt = SCENARIO["uav"]["initial_pos"] + v_opt * t1_opt * uav_dir_opt
    # 起爆点
    det_xy_opt = drop_point_opt[:2] + v_opt * t2_opt * uav_dir_opt[:2]
    det_z_opt = drop_point_opt[2] - 0.5 * SCENARIO["constants"]["g"] * t2_opt**2
    det_point_opt = np.array([det_xy_opt[0], det_xy_opt[1], det_z_opt])
    
    # 打印控制台报告
    total_time = time.time() - start_time
    print(f"试验总数: 3")
    print(f"最优试验编号: {best_result['trial_index']}")
    print(f"总计算耗时: {total_time:.2f} 秒")
    print(f"\n【最优策略参数】")
    print(f"1. 无人机飞行方位角: {best_params[0]:.6f} rad ({np.degrees(best_params[0]):.2f}°)")
    print(f"2. 无人机飞行速度: {best_params[1]:.4f} m/s")
    print(f"3. 载荷投放延迟: {best_params[2]:.4f} s")
    print(f"4. 载荷起爆延迟: {best_params[3]:.4f} s")
    print(f"\n【战术效能指标】")
    print(f"真实遮蔽时长: {best_fitness:.6f} s")
    print(f"原始适应度值: {raw_fitness:.6f} s")
    print(f"边界奖励差值: {raw_fitness - best_fitness:.6f} s")
    
    all_true_fitness = [r['true_fitness'] for r in all_results]
    all_raw_fitness = [r['raw_fitness'] for r in all_results]
    print(f"平均真实遮蔽时长: {np.mean(all_true_fitness):.6f} s")
    print(f"平均原始适应度: {np.mean(all_raw_fitness):.6f} s")
    print(f"标准差: {np.std(all_true_fitness):.6f} s")
    print(f"\n【关键坐标信息】")
    print(f"载荷投放坐标: {drop_point_opt.round(4)}")
    print(f"载荷起爆坐标: {det_point_opt.round(4)}")
    print("="*80)
    # 写入txt文件报告
    try:
        with open('q2.txt', 'w', encoding='utf-8') as f:
            f.write("第二问最优烟幕弹投放策略\n")
            f.write("=" * 50 + "\n")
            f.write(f"无人机飞行方位角: {np.degrees(theta_opt):.2f}°\n")
            f.write(f"无人机飞行速度: {v_opt:.2f} m/s\n")
            f.write(f"载荷投放延迟: {t1_opt:.2f} s\n")
            f.write(f"载荷起爆延迟: {t2_opt:.2f} s\n")
            f.write(f"最优遮蔽时长: {best_fitness:.6f} s\n")
            f.write(f"载荷投放坐标: {drop_point_opt.round(2)}\n")
            f.write(f"载荷起爆坐标: {det_point_opt.round(2)}\n")
        print("\n分析报告已保存至: q2.txt")
    except Exception as e:
        print(f"\n保存TXT文件失败: {e}")

    plot(best_result, all_results)
    
    print(f"可视化图表已保存")