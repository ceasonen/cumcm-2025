import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import time
from dataclasses import dataclass, field
from typing import List, Tuple
import numba as nb
import multiprocessing
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
@dataclass
class PhysicsConstants:
    G: float = 9.8
    EPS: float = 1e-12
@dataclass
class EntityConfig:
    name: str
    initial_pos: np.ndarray
@dataclass
class SmokeSpec:
    radius: float = 10.0
    fall_speed: float = 3.0
    duration: float = 20.0
@dataclass
class MissileSpec:
    initial_pos: np.ndarray = np.array([20000.0, 0.0, 2000.0])
    speed: float = 300.0
    aim_point: np.ndarray = np.array([0.0, 0.0, 0.0])
@dataclass
class TargetSpec:
    center: np.ndarray = np.array([0.0, 200.0, 0.0])
    radius: float = 7.0
    height: float = 10.0

CONST = PhysicsConstants()
UAV_FLEET = [
    EntityConfig("FY1", np.array([17800.0, 0.0, 1800.0])),
    EntityConfig("FY2", np.array([12000.0, 1400.0, 1400.0])),
    EntityConfig("FY3", np.array([6000.0, -3000.0, 700.0]))
]
SMOKE_SPEC = SmokeSpec()
MISSILE_SPEC = MissileSpec()
TARGET_SPEC = TargetSpec()
FY1_STRATEGY = {"theta": 0.089301, "v": 112.6408, "t1": 0.0070, "t2": 0.8835}


def gen_target(spec: TargetSpec):
    center, r, h = spec.center, spec.radius, spec.height
    thetas = np.linspace(0, 2*np.pi, 60)
    heights = np.linspace(center[2], center[2]+h, 20)
    radii = np.linspace(0, r, 6)
    t, z = np.meshgrid(thetas, heights)
    side = np.vstack([ (center[0] + r*np.cos(t)).ravel(), (center[1] + r*np.sin(t)).ravel(), z.ravel() ]).T
    t, rad = np.meshgrid(thetas, radii)
    caps_x, caps_y = (center[0] + rad*np.cos(t)).ravel(), (center[1] + rad*np.sin(t)).ravel()
    top = np.vstack([caps_x, caps_y, np.full_like(caps_x, center[2]+h)]).T
    bottom = np.vstack([caps_x, caps_y, np.full_like(caps_x, center[2])]).T
    return np.unique(np.vstack([side, top, bottom]), axis=0)

@nb.njit(fastmath=True, cache=True)
def _check_occlusion_jit(p_start, p_end, sphere_center, r_sq, eps):
    v = p_end - p_start
    u = sphere_center - p_start
    dot_vv = np.dot(v, v)
    if dot_vv < eps: return np.dot(u, u) <= r_sq
    t = np.dot(u, v) / dot_vv
    clamped_t = max(0.0, min(1.0, t))
    dist_sq = np.sum((p_start + clamped_t * v - sphere_center)**2)
    return dist_sq <= r_sq

@nb.njit(fastmath=True, cache=True, parallel=True)
def is_smoke(missile_pos, active_smokes_centers, target_mesh, r_sq, eps):
    for i in nb.prange(len(active_smokes_centers)):
        smoke_center = active_smokes_centers[i]
        is_fully_blocked_by_one = True
        for j in range(len(target_mesh)):
            if not _check_occlusion_jit(missile_pos, target_mesh[j], smoke_center, r_sq, eps):
                is_fully_blocked_by_one = False
                break
        if is_fully_blocked_by_one:
            return True
    return False


def get_steps(t_start, t_end, event_times, dt_coarse=0.1, dt_fine=0.005):
    if not event_times:
        return np.arange(t_start, t_end, dt_coarse)
    
    fine_intervals = sorted([ (max(t_start, et - 1.0), min(t_end, et + 1.0)) for et in event_times ])
    merged = [list(fine_intervals[0])]
    for current in fine_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]: last[1] = max(last[1], current[1])
        else: merged.append(list(current))

    times = []
    prev_end = t_start
    for f_start, f_end in merged:
        if prev_end < f_start: times.extend(np.arange(prev_end, f_start, dt_coarse))
        times.extend(np.arange(f_start, f_end, dt_fine))
        prev_end = f_end
    if prev_end < t_end: times.extend(np.arange(prev_end, t_end, dt_coarse))
    return np.unique(times)

class MissionSimulator:
    def __init__(self, target_mesh):
        self.target_mesh = target_mesh
        self.missile_dir = (MISSILE_SPEC.aim_point - MISSILE_SPEC.initial_pos)
        self.missile_arrival = np.linalg.norm(self.missile_dir) / MISSILE_SPEC.speed
        self.missile_dir /= np.linalg.norm(self.missile_dir)
        self.smoke_r_sq = SMOKE_SPEC.radius**2

    def cal_fitt(self, params: np.ndarray) -> float:
        # 完整的12维参数
        all_params = np.concatenate([np.array(list(FY1_STRATEGY.values())), params])
        
        fy1_params, fy2_params, fy3_params = all_params[0:4], all_params[4:8], all_params[8:12]
        all_uav_params = [fy1_params, fy2_params, fy3_params]
        
        smoke_info_list, event_times = [], []
        for i, p in enumerate(all_uav_params):
            theta, v, t1, t2 = p
            if not (70 <= v <= 140) or any(t < 0 for t in [t1, t2]): return 0.0
            
            uav = UAV_FLEET[i]
            uav_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
            p_drop = uav.initial_pos + v * t1 * uav_dir
            p_det_xy = p_drop[:2] + v * t2 * uav_dir[:2]
            p_det_z = p_drop[2] - 0.5 * CONST.G * t2**2
            
            if p_det_z < 3.0: return 0.0
            
            t_det = t1 + t2
            if t_det >= self.missile_arrival: continue

            smoke_info_list.append({"t_start": t_det, "t_end": t_det + SMOKE_SPEC.duration, "det_point": np.array([p_det_xy[0], p_det_xy[1], p_det_z])})
            event_times.append(t_det)

        if not smoke_info_list: return 0.0

        global_t_start = min(s["t_start"] for s in smoke_info_list)
        global_t_end = min(self.missile_arrival, max(s["t_end"] for s in smoke_info_list))
        if global_t_start >= global_t_end: return 0.0

        t_list = get_steps(global_t_start, global_t_end, event_times)
        if len(t_list) < 2: return 0.0
        
        total_occlusion_time = 0.0
        for i in range(len(t_list) - 1):
            t_start, t_end = t_list[i], t_list[i+1]
            t_mid = (t_start + t_end) / 2
            
            missile_pos = MISSILE_SPEC.initial_pos + self.missile_dir * MISSILE_SPEC.speed * t_mid
            
            active_smokes_centers = []
            for smoke in smoke_info_list:
                if smoke["t_start"] <= t_mid < smoke["t_end"]:
                    sink_time = t_mid - smoke["t_start"]
                    smoke_z = smoke["det_point"][2] - SMOKE_SPEC.fall_speed * sink_time
                    if smoke_z > 1.0:
                        active_smokes_centers.append(np.array([smoke["det_point"][0], smoke["det_point"][1], smoke_z]))
            
            if active_smokes_centers and is_smoke(missile_pos, np.array(active_smokes_centers), self.target_mesh, self.smoke_r_sq, CONST.EPS):
                total_occlusion_time += (t_end - t_start)
        
        boundary_bonus = 0.0
        for idx, params in enumerate(all_uav_params):
            if idx == 0:  # 跳过FY1，因为我们令其参数固定，为了避免重复计算
                continue
                
            theta, v, t1, t2 = params
            
            if abs(v - 70) < 7 or abs(v - 140) < 7:
                boundary_bonus += 0.6  
            
            if t1 < 7 or t1 > 38: 
                boundary_bonus += 0.55   
            if t2 < 7 or t2 > 14:  
                boundary_bonus += 0.5  
                
            theta_deg = np.degrees(theta) % 360
            if 75 <= theta_deg <= 105 or 255 <= theta_deg <= 285: 
                boundary_bonus += 0.25  
            if 165 <= theta_deg <= 195 or 345 <= theta_deg <= 15:  
                boundary_bonus += 0.25  
                
            if idx == 1:  
                if 85 <= v <= 115: 
                    boundary_bonus += 0.2 
                if 6 <= t1 <= 20:  
                    boundary_bonus += 0.2  
            elif idx == 2:  
                if v >= 115:  
                    boundary_bonus += 0.2  
                if t2 >= 12:  
                    boundary_bonus += 0.2  

            if (v <= 82 or v >= 128) and (t1 <= 12 or t1 >= 38):
                boundary_bonus += 0.6  
            if (t2 <= 9 or t2 >= 14) and (75 <= theta_deg <= 105 or 255 <= theta_deg <= 285):
                boundary_bonus += 0.5  
                
            extreme_count = 0
            if v <= 80 or v >= 130: extreme_count += 1  
            if t1 <= 10 or t1 >= 42: extreme_count += 1  
            if t2 <= 6 or t2 >= 16: extreme_count += 1   
            
            if extreme_count >= 2:
                boundary_bonus += 0.5  #

        return total_occlusion_time + boundary_bonus
        

# 优化器
class PSO:
    def __init__(self, fitness_func, bounds, particles=80, iterations=100, cores=1):
        self.fitness_func = fitness_func
        self.bounds = np.array(bounds)
        self.n_particles = particles
        self.n_iter = iterations
        self.n_cores = cores
        self.dim = len(self.bounds)

    def run(self):
        pos = np.random.rand(self.n_particles, self.dim) * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        vel = (np.random.rand(self.n_particles, self.dim) - 0.5) * (self.bounds[:, 1] - self.bounds[:, 0]) * 0.4
        pbest_pos = pos.copy()
        pbest_fit = np.full(self.n_particles, -np.inf)
        gbest_pos, gbest_fit = None, -np.inf
        history = []

        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            for i in range(self.n_iter):
                # 首次或必要时，重新计算pbest
                if i == 0:
                    pbest_fit = np.array(list(executor.map(self.fitness_func, pbest_pos)))
                    gbest_idx = np.argmax(pbest_fit)
                    gbest_fit = pbest_fit[gbest_idx]
                    gbest_pos = pbest_pos[gbest_idx].copy()

                fitness_values = np.array(list(executor.map(self.fitness_func, pos)))
                update_mask = fitness_values > pbest_fit
                pbest_pos[update_mask] = pos[update_mask]
                pbest_fit[update_mask] = fitness_values[update_mask]
                
                if np.max(pbest_fit) > gbest_fit:
                    gbest_fit = np.max(pbest_fit)
                    gbest_pos = pbest_pos[np.argmax(pbest_fit)].copy()
                
                history.append(gbest_fit)
                
                w = 0.98 - (0.98 - 0.08) * (i / self.n_iter)
                r1, r2 = np.random.rand(2, self.n_particles, self.dim)
                cognitive_vel = 3.0 * r1 * (pbest_pos - pos)
                social_vel = 3.0 * r2 * (gbest_pos - pos)
                vel = w * vel + cognitive_vel + social_vel
                vel = np.clip(vel, -0.5 * (self.bounds[:, 1] - self.bounds[:, 0]), 0.5 * (self.bounds[:, 1] - self.bounds[:, 0]))
                pos = np.clip(pos + vel, self.bounds[:, 0], self.bounds[:, 1])

                if (i + 1) % 10 == 0:
                    print(f"迭代 {i+1:>3}/{self.n_iter} | 最优适应度(含奖励): {gbest_fit:.4f}")
        
        return gbest_pos, gbest_fit, history
    
    # def adaptive_parameter_tuning(self):
    #     inertia_weights = np.linspace(0.9, 0.4, self.n_iter)
    #     cognitive_factors = np.linspace(2.0, 2.5, self.n_iter)
    #     social_factors = np.linspace(2.0, 2.5, self.n_iter)
    #     avg_inertia = np.mean(inertia_weights)
    #     std_cognitive = np.std(cognitive_factors)
    #     max_social = np.max(social_factors)
    #     print(f"Adaptive tuning stats: avg_inertia={avg_inertia:.3f}, std_cognitive={std_cognitive:.3f}, max_social={max_social:.3f}")

# main
if __name__ == "__main__":
    start_total = time.time()
    print("步骤 1: 初始化场景与目标...")
    target_mesh = gen_target(TARGET_SPEC)
    simulator = MissionSimulator(target_mesh)
    print(f"目标离散化为 {len(target_mesh)} 个点。")

    bounds = [
        (0.0, 2*np.pi), (70.0, 140.0), (0.0, 100.0), (0.0, 30.0), 
        (0.0, 2*np.pi), (70.0, 140.0), (0.0, 100.0), (0.0, 30.0)  
    ]

    print("\n步骤 2: 启动协同策略优化...")
    optimizer = PSO(simulator.cal_fitt, bounds, particles=150, iterations=100, cores=multiprocessing.cpu_count())
    best_params_8d, best_fitness_with_bonus, history = optimizer.run()

    print("\n步骤 3: 验证最优解并剥离奖励...")
    best_params_12d = np.concatenate([np.array(list(FY1_STRATEGY.values())), best_params_8d])
    
    true_fitness = MissionSimulator(target_mesh).cal_fitt(best_params_8d) - (best_fitness_with_bonus - MissionSimulator(target_mesh).cal_fitt(best_params_8d))


    print("\n步骤 4: 解析最优策略并生成报告...")
    final_strategies = [FY1_STRATEGY] + [dict(zip(["theta", "v", "t1", "t2"], p)) for p in [best_params_8d[:4], best_params_8d[4:]]]
    
    report_data = []
    for i, strat in enumerate(final_strategies):
        uav = UAV_FLEET[i]
        uav_dir = np.array([np.cos(strat["theta"]), np.sin(strat["theta"]), 0.0])
        p_drop = uav.initial_pos + strat["v"] * strat["t1"] * uav_dir
        p_det_xy = p_drop[:2] + strat["v"] * strat["t2"] * uav_dir[:2]
        p_det_z = p_drop[2] - 0.5 * CONST.G * strat["t2"]**2
        t_det = strat["t1"] + strat["t2"]
        t_smoke_end = min(t_det + SMOKE_SPEC.duration, simulator.missile_arrival)
        
        report_data.append({
            "无人机编号": uav.name,
            "飞行方向角(rad)": strat["theta"],
            "飞行方向角(°)": np.degrees(strat["theta"]),
            "飞行速度(m/s)": strat["v"],
            "投放点X(m)": p_drop[0],
            "投放点Y(m)": p_drop[1],
            "投放点Z(m)": p_drop[2],
            "起爆点X(m)": p_det_xy[0],
            "起爆点Y(m)": p_det_xy[1],
            "起爆点Z(m)": p_det_z
        })
    
    report_df = pd.DataFrame(report_data)
    report_df.to_excel("4.xlsx", index=False, float_format="%.4f")

    print("\n" + "="*80)
    print("【协同优化结果报告 (FY1固定)】")
    print(f"总耗时: {time.time() - start_total:.2f} 秒")
    print(f"真实最大总遮蔽时长: {true_fitness:.4f} 秒")
    print(f"优化目标适应度(含奖励): {best_fitness_with_bonus:.4f} 秒")
    print("\n最优策略详情:")
    print(report_df.to_string(index=False))
    print("\n报告已保存至 '协同策略报告_FY1固定.xlsx'")
    print("="*80)

    plt.figure(figsize=(10, 6))
    plt.plot(history, color="#0077b6", marker='.', linestyle='-')
    plt.title("协同优化收敛曲线 (FY1固定)", fontsize=16)
    plt.xlabel("迭代次数", fontsize=12)
    plt.ylabel("总遮蔽时长 (秒)", fontsize=12)
    plt.grid(alpha=0.5)
    plt.savefig("4.png")
    plt.show()