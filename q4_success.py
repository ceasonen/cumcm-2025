import numpy as np
import pandas as pd
import time
import warnings
import multiprocessing as mp
from scipy.optimize import differential_evolution

# ========================= 1. 全局参数与环境配置 =========================
warnings.filterwarnings('ignore', category=RuntimeWarning)
mp.set_start_method('fork', force=True) if hasattr(os, 'fork') else None

# 物理常量
G = 9.8
# 烟幕参数
SMOKE_RADIUS = 10.0
SMOKE_EFFECTIVE_TIME = 20.0
SMOKE_SINK_SPEED = 3.0
# 导弹M1参数
MISSILE_INIT_POS = np.array([20000.0, 0.0, 2000.0])
MISSILE_SPEED = 300.0
# 目标参数
FAKE_TARGET_POS = np.array([0.0, 0.0, 0.0])
TRUE_TARGET_CENTER = np.array([0.0, 200.0, 0.0])
TRUE_TARGET_RADIUS = 7.0
TRUE_TARGET_HEIGHT = 10.0
# 无人机初始位置
DRONES_INIT_POS = {
    1: np.array([17800.0, 0.0, 1800.0]),
    2: np.array([12000.0, 1400.0, 1400.0]),
    3: np.array([6000.0, -3000.0, 700.0])
}
# 预计算关键物理量
MISSILE_DIR = (FAKE_TARGET_POS - MISSILE_INIT_POS) / np.linalg.norm(FAKE_TARGET_POS - MISSILE_INIT_POS)
MISSILE_TOTAL_FLIGHT_TIME = np.linalg.norm(FAKE_TARGET_POS - MISSILE_INIT_POS) / MISSILE_SPEED
print(f"【物理约束】导弹全程飞行时间: {MISSILE_TOTAL_FLIGHT_TIME:.2f} s。所有事件必须在此之前完成。")
# 仿真参数
TIME_STEP = 0.1
CPU_CORES = max(1, mp.cpu_count() - 1)

# 全局仿真引擎实例，避免重复初始化
SIM_ENGINE = None

# ========================= 2. 核心仿真引擎 (融合q1, q2, q3精华) =========================
class SimulationEngine:
    """封装所有物理和几何计算，确保一致性和效率"""
    def __init__(self):
        self.target_mesh = self._generate_target_mesh()
        print(f"仿真引擎初始化：生成目标采样点 {len(self.target_mesh)} 个。")

    def _generate_target_mesh(self, num_heights=6, num_angles=24):
        points = []
        cx, cy, cz = TRUE_TARGET_CENTER
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        heights = np.linspace(cz, cz + TRUE_TARGET_HEIGHT, num_heights)
        # 顶面/底面/侧面
        for z in [cz, cz + TRUE_TARGET_HEIGHT]:
            for r_factor in [0.5, 1.0]:
                for a in angles:
                    points.append([cx + TRUE_TARGET_RADIUS * r_factor * np.cos(a), cy + TRUE_TARGET_RADIUS * r_factor * np.sin(a), z])
        for z in heights:
            for a in angles:
                points.append([cx + TRUE_TARGET_RADIUS * np.cos(a), cy + TRUE_TARGET_RADIUS * np.sin(a), z])
        return np.unique(np.array(points), axis=0)

    def calculate_detonation_point(self, drone_id, theta, v, td, tau):
        uav_init_pos = DRONES_INIT_POS[drone_id]
        uav_dir = np.array([np.cos(theta), np.sin(theta)])
        # 烟幕弹在td时刻释放，继承无人机水平速度
        # 在tau秒内，水平匀速，竖直自由落体
        release_pos = uav_init_pos + np.append(v * td * uav_dir, 0)
        det_xy = release_pos[:2] + v * tau * uav_dir
        det_z = release_pos[2] - 0.5 * G * tau**2
        return np.array([det_xy[0], det_xy[1], det_z])

    def get_occlusion_intervals(self, drone_id, theta, v, td, tau):
        # 物理可行性检查
        t_detonation = td + tau
        if t_detonation >= MISSILE_TOTAL_FLIGHT_TIME: return []
        det_point = self.calculate_detonation_point(drone_id, theta, v, td, tau)
        if det_point[2] < -SMOKE_RADIUS: return []

        t_start = t_detonation
        t_end = min(t_start + SMOKE_EFFECTIVE_TIME, MISSILE_TOTAL_FLIGHT_TIME)
        
        intervals = []
        in_occlusion = False
        current_interval_start = 0

        for t in np.arange(t_start, t_end, TIME_STEP):
            missile_pos = MISSILE_INIT_POS + MISSILE_SPEED * t * MISSILE_DIR
            smoke_center = det_point - np.array([0, 0, SMOKE_SINK_SPEED * (t - t_start)])
            if smoke_center[2] < -SMOKE_RADIUS: break

            # 射线-球体相交判断
            is_occluded = all(self._is_line_segment_intersecting_sphere(missile_pos, p, smoke_center) for p in self.target_mesh)
            
            if is_occluded and not in_occlusion:
                in_occlusion = True
                current_interval_start = t
            elif not is_occluded and in_occlusion:
                in_occlusion = False
                intervals.append((current_interval_start, t))
                
        if in_occlusion: intervals.append((current_interval_start, t_end))
        return intervals

    def _is_line_segment_intersecting_sphere(self, p1, p2, c, r=SMOKE_RADIUS):
        v = p2 - p1
        a = np.dot(v, v)
        if a < 1e-9: return np.linalg.norm(p1 - c) <= r
        w = p1 - c
        b = 2 * np.dot(w, v)
        c_eq = np.dot(w, w) - r**2
        d = b**2 - 4 * a * c_eq
        return d >= 0

def merge_and_get_duration(intervals_list):
    all_intervals = [item for sublist in intervals_list for item in sublist]
    if not all_intervals: return 0.0
    all_intervals.sort()
    merged = [list(all_intervals[0])]
    for s, e in all_intervals[1:]:
        if s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
        else: merged.append([s, e])
    return sum(e - s for s, e in merged)

# ========================= 3. 目标函数 (必须在顶层定义) =========================
def objective_function(params_12d):
    """用于协同优化的12维目标函数"""
    intervals_all = []
    for i in range(3):
        p = params_12d[i*4 : (i+1)*4]
        intervals_all.append(SIM_ENGINE.get_occlusion_intervals(i + 1, *p))
    return -merge_and_get_duration(intervals_all)

# ========================= 4. 主执行流程 =========================
def main():
    global SIM_ENGINE
    SIM_ENGINE = SimulationEngine()
    
    total_start_time = time.time()
    
    # --- 阶段一：生成启发式种子解 (快速获得高质量起点) ---
    print("\n" + "="*20 + " 阶段一: 生成启发式种子解 " + "="*20)
    heuristic_seed_12d = []
    total_heuristic_duration = 0
    
    for i in range(1, 4):
        # 启发式方向：直指真目标
        uav_pos = DRONES_INIT_POS[i]
        vec_to_target = TRUE_TARGET_CENTER - uav_pos
        heuristic_theta = np.arctan2(vec_to_target[1], vec_to_target[0])
        
        # 启发式速度：取中间值
        heuristic_v = 105.0
        
        # 针对时间延迟进行快速寻优
        def time_objective(time_params):
            td, tau = time_params
            p = (heuristic_theta, heuristic_v, td, tau)
            return -merge_and_get_duration([SIM_ENGINE.get_occlusion_intervals(i, *p)])
        
        time_bounds = [(5.0, MISSILE_TOTAL_FLIGHT_TIME - 21), (1.0, 15.0)]
        time_res = differential_evolution(time_objective, time_bounds, maxiter=30, popsize=10, tol=0.1)
        
        heuristic_td, heuristic_tau = time_res.x
        solo_duration = -time_res.fun
        total_heuristic_duration += solo_duration
        
        drone_seed = [heuristic_theta, heuristic_v, heuristic_td, heuristic_tau]
        heuristic_seed_12d.extend(drone_seed)
        print(f"FY{i} 种子解生成: 独立时长 {solo_duration:.2f}s, params: [th={np.degrees(heuristic_theta):.1f}°, v={heuristic_v}, td={heuristic_td:.1f}, tau={heuristic_tau:.1f}]")

    heuristic_seed_12d = np.array(heuristic_seed_12d)
    
    # --- 阶段二：精英种群引导的协同精细优化 ---
    print("\n" + "="*20 + " 阶段二: 精英种群引导的协同优化 " + "="*20)
    
    bounds_12d = [(0, 2*np.pi), (70, 140), (1, MISSILE_TOTAL_FLIGHT_TIME - 21), (1, 20)] * 3
    popsize = 50
    
    # 构建精英初始种群
    initial_population = np.random.rand(popsize, 12)
    for i in range(12):
        initial_population[:, i] = initial_population[:, i] * (bounds_12d[i][1] - bounds_12d[i][0]) + bounds_12d[i][0]
    initial_population[0] = heuristic_seed_12d # 注入精英种子
    
    # 加入种子周围的扰动点
    for i in range(1, 10):
        noise_scale = np.array([0.2, 10, 5, 2] * 3) # 为时间和速度增加较大扰动
        mutated_seed = heuristic_seed_12d + np.random.normal(0, noise_scale)
        # 边界约束
        min_b = np.array([b[0] for b in bounds_12d])
        max_b = np.array([b[1] for b in bounds_12d])
        initial_population[i] = np.clip(mutated_seed, min_b, max_b)

    # 运行协同优化
    result = differential_evolution(
        objective_function, bounds_12d, 
        init=initial_population, 
        maxiter=150, popsize=popsize, strategy='best1bin', 
        updating='deferred', workers=CPU_CORES, tol=0.01
    )
    final_params = result.x
    final_duration = -result.fun

    # --- 结果输出 ---
    total_time = time.time() - total_start_time
    print("\n" + "="*30 + " 【最终优化结果】 " + "="*30)
    print(f"总计算耗时: {total_time:.2f} s")
    print(f"*** 最优总遮蔽时长: {final_duration:.4f} s ***")

    data = []
    for i in range(3):
        p = final_params[i*4:(i+1)*4]
        data.append({
            "无人机编号": f"FY{i+1}", "飞行方向(°)": np.degrees(p[0]), "飞行速度(m/s)": p[1],
            "投放延迟(s)": p[2], "起爆延迟(s)": p[3], "绝对起爆时刻(s)": p[2] + p[3]
        })

    df = pd.DataFrame(data).sort_values(by="绝对起爆时刻(s)").reset_index(drop=True)
    print("\n最终策略详情 (按起爆顺序排列):")
    print(df.round(2))

    df.to_excel("result2.xlsx", index=False)
    print(f"\n结果已成功保存至: result2.xlsx")

if __name__ == "__main__":
    main()