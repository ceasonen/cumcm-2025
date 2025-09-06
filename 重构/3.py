import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import time
import multiprocessing
from dataclasses import dataclass

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class SceneConfig:
    g: float = 9.8
    eps: float = 1e-12
    dt: float = 0.01
    cpu_cores: int = max(1, multiprocessing.cpu_count() - 2)

    target_center: np.ndarray = np.array([0.0, 200.0, 0.0])
    target_radius: float = 7.0
    target_height: float = 10.0
    uav_initial_pos: np.ndarray = np.array([17800.0, 0.0, 1800.0])
    missile_initial_pos: np.ndarray = np.array([20000.0, 0.0, 2000.0])
    missile_speed: float = 300.0
    missile_aim_point: np.ndarray = np.array([0.0, 0.0, 0.0])
    smoke_radius: float = 10.0
    smoke_fall_speed: float = 3.0
    smoke_duration: float = 20.0

def generate_target_volume(center, radius, height, density=1.0):
    n_theta, n_h, n_r = int(60*density), int(20*density), int(5*density)
    
    thetas = np.linspace(0, 2 * np.pi, n_theta)
    heights = np.linspace(center[2], center[2] + height, n_h)
    radii = np.linspace(0, radius, n_r)

    t_grid, h_grid = np.meshgrid(thetas, heights)
    x_side = center[0] + radius * np.cos(t_grid)
    y_side = center[1] + radius * np.sin(t_grid)
    side_points = np.vstack([x_side.ravel(), y_side.ravel(), h_grid.ravel()]).T

    t_grid_caps, r_grid_caps = np.meshgrid(thetas, radii)
    x_caps = center[0] + r_grid_caps * np.cos(t_grid_caps)
    y_caps = center[1] + r_grid_caps * np.sin(t_grid_caps)
    top_points = np.vstack([x_caps.ravel(), y_caps.ravel(), np.full_like(x_caps.ravel(), center[2] + height)]).T
    bottom_points = np.vstack([x_caps.ravel(), y_caps.ravel(), np.full_like(x_caps.ravel(), center[2])]).T
    
    return np.unique(np.vstack([side_points, top_points, bottom_points]), axis=0)

@nb.njit(fastmath=True, cache=True)
def _is_occluded_jit(p_start, p_end, sphere_center, r_sq, eps):
    v = p_end - p_start
    u = sphere_center - p_start
    a = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
    if a < eps: return (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) <= r_sq
    
    b = -2 * (v[0]*u[0] + v[1]*u[1] + v[2]*u[2])
    c = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) - r_sq
    discriminant = b*b - 4*a*c
    if discriminant < 0: return False
    
    sqrt_d = np.sqrt(discriminant)
    t1, t2 = (-b - sqrt_d) / (2*a), (-b + sqrt_d) / (2*a)
    return (t1 <= 1.0 and t2 >= 0.0)

@nb.njit(fastmath=True, cache=True, parallel=True)
def check_volume_occlusion(missile_pos, smoke_pos, target_volume, r_sq, eps):
    for i in nb.prange(target_volume.shape[0]):
        if not _is_occluded_jit(missile_pos, target_volume[i], smoke_pos, r_sq, eps):
            return False
    return True

class Missile:
    def __init__(self, cfg: SceneConfig):
        self.initial_pos = cfg.missile_initial_pos
        self.speed = cfg.missile_speed
        direction = cfg.missile_aim_point - self.initial_pos
        self.unit_vec = direction / np.linalg.norm(direction)
        self.arrival_time = np.linalg.norm(direction) / self.speed

    def position_at(self, t: float) -> np.ndarray:
        return self.initial_pos + self.unit_vec * self.speed * t

class SmokeCloud:
    def __init__(self, cfg: SceneConfig, theta, v, t_deploy, t_fuse):
        self.cfg = cfg
        self.t_deploy = t_deploy
        self.t_fuse = t_fuse
        self.t_detonation = t_deploy + t_fuse
        
        uav_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
        p_drop = cfg.uav_initial_pos + uav_dir * v * t_deploy
        p_det_xy = p_drop[:2] + uav_dir[:2] * v * t_fuse
        p_det_z = p_drop[2] - 0.5 * cfg.g * t_fuse**2
        self.detonation_point = np.array([p_det_xy[0], p_det_xy[1], p_det_z])
        
        # 添加投放点信息
        self.drop_point = p_drop
        
        self.t_start = self.t_detonation
        self.t_end = self.t_detonation + cfg.smoke_duration

    def center_at(self, t: float) -> np.ndarray:
        time_since_det = t - self.t_detonation
        return self.detonation_point - np.array([0, 0, self.cfg.smoke_fall_speed * time_since_det])

class ObscurationSimulator:
    def __init__(self, cfg: SceneConfig, target_volume: np.ndarray, missile: Missile):
        self.cfg = cfg
        self.target_volume = target_volume
        self.missile = missile
        self.timeline = np.arange(0, self.missile.arrival_time, cfg.dt)
        self.missile_trajectory = np.array([missile.position_at(t) for t in self.timeline])

    def evaluate_strategy(self, params: np.ndarray) -> float:
        theta, v, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3 = params
        
        if not (70.0 <= v <= 140.0): return 0.0
        if delta_t2 < 1.0 or delta_t3 < 1.0: return 0.0
        if t1_1 < 0 or t2_1 < 0 or t2_2 < 0 or t2_3 < 0: return 0.0

        t1_2 = t1_1 + delta_t2
        t1_3 = t1_2 + delta_t3
        
        bombs = [
            SmokeCloud(self.cfg, theta, v, t1_1, t2_1),
            SmokeCloud(self.cfg, theta, v, t1_2, t2_2),
            SmokeCloud(self.cfg, theta, v, t1_3, t2_3),
        ]
        
        global_obscuration_mask = np.zeros_like(self.timeline, dtype=bool)

        for bomb in bombs:
            if bomb.detonation_point[2] < 0: continue
            
            active_indices = np.where((self.timeline >= bomb.t_start) & (self.timeline <= bomb.t_end))[0]
            if len(active_indices) == 0: continue

            bomb_obscuration_mask = np.zeros_like(self.timeline, dtype=bool)
            for idx in active_indices:
                smoke_pos = bomb.center_at(self.timeline[idx])
                if smoke_pos[2] < 0: break
                
                is_obscured = check_volume_occlusion(
                    self.missile_trajectory[idx], smoke_pos, self.target_volume, 
                    self.cfg.smoke_radius**2, self.cfg.eps
                )
                if is_obscured:
                    bomb_obscuration_mask[idx] = True
            
            global_obscuration_mask |= bomb_obscuration_mask
            
        base_fitness = np.sum(global_obscuration_mask) * self.cfg.dt

        precision_noise = np.random.normal(0, 0.001)  

    def calculate_bomb_obscuration_count(self, bomb: 'SmokeCloud') -> int:
        if bomb.detonation_point[2] < 0:
            return 0
            
        active_indices = np.where((self.timeline >= bomb.t_start) & (self.timeline <= bomb.t_end))[0]
        if len(active_indices) == 0:
            return 0

        bomb_obscuration_mask = np.zeros_like(self.timeline, dtype=bool)
        for idx in active_indices:
            smoke_pos = bomb.center_at(self.timeline[idx])
            if smoke_pos[2] < 0:
                break
                
            is_obscured = check_volume_occlusion(
                self.missile_trajectory[idx], smoke_pos, self.target_volume, 
                self.cfg.smoke_radius**2, self.cfg.eps
            )
            if is_obscured:
                bomb_obscuration_mask[idx] = True
        
        return np.sum(bomb_obscuration_mask)

class PSO:
    def __init__(self, fitness_func, bounds, particles=50, iterations=120, cores=1):
        self.fitness_func = fitness_func
        self.bounds = np.array(bounds)
        self.n_particles = particles
        self.n_iter = iterations
        self.n_cores = cores
        self.dim = len(self.bounds)

    def run(self):
        pos = np.random.rand(self.n_particles, self.dim) * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        vel = (np.random.rand(self.n_particles, self.dim) - 0.5) * (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
        
        pbest_pos = pos.copy()
        pbest_fit = np.full(self.n_particles, -1.0)
        gbest_pos, gbest_fit = None, -1.0
        history = []

        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            for i in range(self.n_iter):
                fitness_values = np.array(list(executor.map(self.fitness_func, pos)))
                
                update_mask = fitness_values > pbest_fit
                pbest_pos[update_mask] = pos[update_mask]
                pbest_fit[update_mask] = fitness_values[update_mask]
                
                if np.max(pbest_fit) > gbest_fit:
                    gbest_fit = np.max(pbest_fit)
                    gbest_pos = pbest_pos[np.argmax(pbest_fit)]
                
                history.append(gbest_fit)
                
                w = 0.9 - 0.5 * (i / self.n_iter)
                r1, r2 = np.random.rand(2, self.n_particles, self.dim)
                
                cognitive_vel = 2.0 * r1 * (pbest_pos - pos)
                social_vel = 2.0 * r2 * (gbest_pos - pos)
                vel = w * vel + cognitive_vel + social_vel
                pos = np.clip(pos + vel, self.bounds[:, 0], self.bounds[:, 1])

                if (i + 1) % 10 == 0:
                    print(f"迭代 {i+1:>3}/{self.n_iter} | 最优适应度: {gbest_fit:.4f} 秒")
        
        return gbest_pos, gbest_fit, history

if __name__ == "__main__":
    start_time = time.time()
    
    print("--- 正在初始化仿真框架 ---")
    config = SceneConfig()
    missile = Missile(config)
    target_volume = generate_target_volume(config.target_center, config.target_radius, config.target_height)
    simulator = ObscurationSimulator(config, target_volume, missile)
    
    print(f"目标已离散化为 {len(target_volume)} 个点。")
    print(f"使用 {config.cpu_cores} 个CPU核心进行优化。")

    bounds = [
        (0.0, 2*np.pi),
        (70.0, 140.0),
        (0.0, 60.0),
        (0.0, 20.0),
        (1.0, 30.0),
        (0.0, 20.0),
        (1.0, 30.0),
        (0.0, 20.0)
    ]

    print("\n--- 启动粒子群优化 ---")
    optimizer = PSO(simulator.evaluate_strategy, bounds, particles=50, iterations=120, cores=config.cpu_cores)
    best_strategy, max_obscuration, history = optimizer.run()

    print("\n--- 优化完成 ---")
    
    theta, v, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3 = best_strategy
    t1_2 = t1_1 + delta_t2
    t1_3 = t1_2 + delta_t3
    
    bombs_final = [
        SmokeCloud(config, theta, v, t1_1, t2_1),
        SmokeCloud(config, theta, v, t1_2, t2_2),
        SmokeCloud(config, theta, v, t1_3, t2_3),
    ]
    
    # 计算每个烟幕弹的遮蔽区间数量
    bomb_obscuration_counts = []
    for bomb in bombs_final:
        count = simulator.calculate_bomb_obscuration_count(bomb)
        bomb_obscuration_counts.append(count)
    
    df_data = []
    for i, (bomb, obs_count) in enumerate(zip(bombs_final, bomb_obscuration_counts), 1):
        df_data.append({
            "无人机航向(rad)": theta,
            "无人机速度(m/s)": v,
            "投放点X(m)": bomb.drop_point[0],
            "投放点Y(m)": bomb.drop_point[1],
            "投放点Z(m)": bomb.drop_point[2],
            "起爆点X(m)": bomb.detonation_point[0],
            "起爆点Y(m)": bomb.detonation_point[1],
            "起爆点Z(m)": bomb.detonation_point[2],
            "遮蔽区间数量": obs_count
        })
    results_df = pd.DataFrame(df_data)
    results_df.to_excel("3.xlsx", index=False, float_format="%.2f")

    print(f"\n总执行耗时: {time.time() - start_time:.2f} 秒")
    print("="*60)
    print("           最优协同策略报告")
    print("="*60)
    print(f"最大总遮蔽时长: {max_obscuration:.4f} 秒")
    print(f"无人机飞行角度: {np.degrees(theta):.2f}°")
    print(f"无人机飞行速度: {v:.2f} m/s")
    print("\n部署详情:")
    print(results_df.to_string(index=False))
    print("\n完整报告已保存至 '3.xlsx'")
    print("="*60)

    plt.figure(figsize=(10, 6))
    plt.plot(history, marker='o', linestyle='-', markersize=4, color='b')
    plt.title("PSO优化收敛曲线", fontsize=16)
    plt.xlabel("迭代次数", fontsize=12)
    plt.ylabel("总遮蔽时长 (秒)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("3.png")
    plt.show()