import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, linear_sum_assignment

# ============================ 1. 全局参数初始化（优化搜索范围） ============================

# 真目标参数
TRUE_TARGET = {
    "r": 7,  # 圆柱半径
    "h": 10,  # 圆柱高度
    "center": np.array([0, 200, 0]),  # 底面圆心
    "sample_points": None  # 采样点
}

# 导弹参数
MISSILES = {
    "M1": {"init_pos": np.array([20000, 0, 2000]), "dir": None, "flight_time": None},
    "M2": {"init_pos": np.array([19000, 600, 2100]), "dir": None, "flight_time": None},
    "M3": {"init_pos": np.array([18000, -600, 1900]), "dir": None, "flight_time": None}
}
MISSILE_SPEED = 300  # 导弹速度(m/s)
G = 9.8  # 重力加速度(m/s²)
SMOKE_RADIUS = 10  # 烟幕有效半径(m)
SMOKE_SINK_SPEED = 3  # 起爆后下沉速度(m/s)
SMOKE_EFFECTIVE_TIME = 20  # 起爆后有效时长(s)

# 无人机参数
DRONES = {
    "FY1": {"init_pos": np.array([17800, 0, 1800]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None},
    "FY2": {"init_pos": np.array([12000, 1400, 1400]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None},
    "FY3": {"init_pos": np.array([6000, -3000, 700]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None},
    "FY4": {"init_pos": np.array([11000, 2000, 1800]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None},
    "FY5": {"init_pos": np.array([13000, -2000, 1300]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None}
}

DROP_INTERVAL = 1  # 同一无人机投放间隔(s)
TIME_STEP = 0.1  # 时间采样步长(s)

# 生成真目标采样点
def generate_true_target_samples():
    samples = []
    r, h, center = TRUE_TARGET["r"], TRUE_TARGET["h"], TRUE_TARGET["center"]
    # 底面采样
    samples.append(center)
    for theta in np.linspace(0, 2 * np.pi, 15):
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, center[2]]))
    # 顶面采样
    top_center = center + np.array([0, 0, h])
    samples.append(top_center)
    for theta in np.linspace(0, 2 * np.pi, 15):
        x = top_center[0] + r * np.cos(theta)
        y = top_center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, top_center[2]]))
    # 侧面采样
    for z in np.linspace(center[2], top_center[2], 5):
        for theta in np.linspace(0, 2 * np.pi, 12):
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            samples.append(np.array([x, y, z]))
    TRUE_TARGET["sample_points"] = np.array(samples)

# 初始化导弹参数
def init_missiles():
    for m_name, m_data in MISSILES.items():
        init_pos = m_data["init_pos"]
        dir_vec = -init_pos / np.linalg.norm(init_pos)
        m_data["dir"] = dir_vec * MISSILE_SPEED
        m_data["flight_time"] = np.linalg.norm(init_pos) / MISSILE_SPEED

# 初始化所有参数
generate_true_target_samples()
init_missiles()

# ============================ 2. 核心工具函数（增加Z坐标容错） ============================

# 导弹位置计算
def get_missile_pos(m_name, t):
    m_data = MISSILES[m_name]
    if t > m_data["flight_time"]:
        return m_data["init_pos"] + m_data["dir"] * m_data["flight_time"]
    return m_data["init_pos"] + m_data["dir"] * t

# 无人机位置计算
def get_drone_pos(drone_name, t):
    drone = DRONES[drone_name]
    if drone["speed"] is None or drone["direction"] is None:
        return drone["init_pos"]

    v_vec = np.array([drone["speed"] * np.cos(drone["direction"]),
                      drone["speed"] * np.sin(drone["direction"]), 0])
    return drone["init_pos"] + v_vec * t

# 烟幕弹位置计算
def get_smoke_pos(drone_name, drop_time, det_delay, t):
    drone = DRONES[drone_name]

    if t < drop_time:
        return None

    drop_pos = get_drone_pos(drone_name, drop_time)

    # 投放后到起爆前
    if t < drop_time + det_delay:
        delta_t = t - drop_time
        v_vec = np.array([drone["speed"] * np.cos(drone["direction"]),
                          drone["speed"] * np.sin(drone["direction"]), 0])
        x = drop_pos[0] + v_vec[0] * delta_t
        y = drop_pos[1] + v_vec[1] * delta_t
        z = drop_pos[2] - 0.5 * G * delta_t ** 2
        return np.array([x, y, max(z, 0.1)])  # Z轴最小0.1，避免无效

    # 起爆后
    det_time = drop_time + det_delay
    if t > det_time + SMOKE_EFFECTIVE_TIME:
        return None

    # 直接计算起爆位置
    delta_t_det = det_delay
    v_vec = np.array([drone["speed"] * np.cos(drone["direction"]),
                      drone["speed"] * np.sin(drone["direction"]), 0])
    det_x = drop_pos[0] + v_vec[0] * delta_t_det
    det_y = drop_pos[1] + v_vec[1] * delta_t_det
    det_z = drop_pos[2] - 0.5 * G * delta_t_det ** 2

    if det_z < 0:
        det_z = 0.1  # 容错处理

    delta_t_after = t - det_time
    z = det_z - SMOKE_SINK_SPEED * delta_t_after
    return np.array([det_x, det_y, max(z, 0.1)])  # Z轴最小0.1

# 线段与球相交判定
def segment_sphere_intersect(p1, p2, center, radius):
    vec_p = p2 - p1
    vec_c = center - p1
    t = np.dot(vec_c, vec_p) / (np.dot(vec_p, vec_p) + 1e-8)

    if 0 <= t <= 1:
        nearest = p1 + t * vec_p
    else:
        nearest = p1 if t < 0 else p2

    return np.linalg.norm(nearest - center) <= radius + 1e-8

# 单烟幕弹遮蔽时长计算
def calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay):
    drone = DRONES[drone_name]
    v, theta = drone["speed"], drone["direction"]

    if v is None or theta is None:
        return -1000

    if not (drone["speed_range"][0] - 1e-3 <= v <= drone["speed_range"][1] + 1e-3):  # 增加容错
        return -1000

    # 检查起爆点有效性
    det_time = drop_time + det_delay
    drop_pos = get_drone_pos(drone_name, drop_time)
    delta_t_det = det_delay
    det_z = drop_pos[2] - 0.5 * G * delta_t_det ** 2
    if det_z < -0.5:  # 放宽Z轴判定
        return -1000

    # 检查投放间隔
    for smoke in drone["smokes"]:
        if abs(drop_time - smoke["drop_time"]) < DROP_INTERVAL - 0.1:
            return -1000

    # 计算有效时长
    max_t = min(det_time + SMOKE_EFFECTIVE_TIME, MISSILES[m_name]["flight_time"] + 1)
    min_t = max(det_time, 0)
    if min_t >= max_t - 1e-3:
        return 0

    effective_duration = 0
    for t in np.arange(min_t, max_t, TIME_STEP):
        m_pos = get_missile_pos(m_name, t)
        smoke_pos = get_smoke_pos(drone_name, drop_time, det_delay, t)
        if smoke_pos is None:
            continue

        all_intersect = True
        for sample in TRUE_TARGET["sample_points"]:
            if not segment_sphere_intersect(m_pos, sample, smoke_pos, SMOKE_RADIUS):
                all_intersect = False
                break
        if all_intersect:
            effective_duration += TIME_STEP
    return effective_duration

# ============================ 3. 优化函数（增强差分进化搜索能力） ============================

# 单弹优化（此函数在当前逻辑中未直接使用，但可用于辅助分析）
def optimize_single_smoke(drone_name, m_name):
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_flight_time = MISSILES[m_name]["flight_time"]

    bounds = [
        (v_min * 0.8, v_max * 1.2),
        (0, 2 * np.pi),
        (0, max_flight_time - 1),
        (0.1, 20)
    ]

    def objective(x):
        v, theta, drop_time, det_delay = x
        drone["speed"] = v
        drone["direction"] = theta
        return -calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay)

    result = differential_evolution(
        func=objective,
        bounds=bounds,
        mutation=0.8,
        recombination=0.9,
        popsize=60,
        maxiter=80,
        tol=1e-3,
        disp=False,
        polish=True
    )

    v_opt, theta_opt, drop_time_opt, det_delay_opt = result.x
    v_opt = np.clip(v_opt, v_min, v_max)
    effective_time = -result.fun
    return {
        "v": v_opt,
        "theta": theta_opt,
        "drop_time": drop_time_opt,
        "det_delay": det_delay_opt,
        "det_time": drop_time_opt + det_delay_opt,
        "det_pos": get_smoke_pos(drone_name, drop_time_opt, det_delay_opt, drop_time_opt + det_delay_opt),
        "effective_time": effective_time if effective_time > 1e-3 else 0,
        "missile": m_name
    }

# 无人机轨迹优化（为单个无人机添加一枚烟幕弹）
def optimize_drone_trajectory(drone_name, m_name):
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]

    # 如果无人机烟幕弹已满，则无法继续优化
    if len(drone["smokes"]) >= drone["max_smoke"]:
        return None

    v_candidates = np.linspace(v_min, v_max, 8)
    best_v = None
    best_smoke = None
    max_effective_time = 0

    for v in v_candidates:
        drone["speed"] = v
        
        # 关键：根据已有的烟幕弹设置最小投放时间
        min_drop_time = 0
        if drone["smokes"]:
            # 找到这架无人机已投放烟幕弹的最晚投放时间
            last_drop_time = max([s["drop_time"] for s in drone["smokes"]])
            min_drop_time = last_drop_time + DROP_INTERVAL
        
        max_drop_time = MISSILES[m_name]["flight_time"] - 0.1
        
        if min_drop_time >= max_drop_time - 1e-3:
            continue

        def objective(x):
            theta, drop_time, det_delay = x
            drone["direction"] = theta
            return -calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay)

        result = differential_evolution(
            func=objective,
            bounds=[(0, 2 * np.pi), (min_drop_time, max_drop_time), (0.1, 10)],
            mutation=0.7,
            recombination=0.8,
            popsize=50,
            maxiter=60,
            disp=False
        )
        
        # 检查优化是否成功
        if result.success and -result.fun > 0.1:
            effective_time = -result.fun
            if effective_time > max_effective_time:
                max_effective_time = effective_time
                theta_opt, drop_time_opt, det_delay_opt = result.x
                best_v = v
                best_smoke = {
                    "v": v,
                    "theta": theta_opt,
                    "drop_time": drop_time_opt,
                    "det_delay": det_delay_opt,
                    "det_time": drop_time_opt + det_delay_opt,
                    "effective_time": effective_time,
                    "missile": m_name
                }
    
    # 路径拟合与波动优化（在获得所有烟幕弹后统一进行，此处简化为返回最佳单弹）
    if best_smoke:
        # 临时设置无人机参数以计算det_pos
        drone["speed"] = best_smoke["v"]
        drone["direction"] = best_smoke["theta"]
        det_pos = get_smoke_pos(drone_name, best_smoke["drop_time"], best_smoke["det_delay"], best_smoke["det_time"])
        best_smoke["det_pos"] = det_pos
        return best_smoke
    
    return None

# ============================ 4. 任务分配与迭代优化（核心修改） ============================

# 任务分配
def assign_tasks():
    unoptimized_drones = [d_name for d_name, d_data in DRONES.items() if len(d_data["smokes"]) < d_data["max_smoke"]]
    
    if not unoptimized_drones:
        return {}

    missile_list = list(MISSILES.keys())
    n_drones = len(unoptimized_drones)
    n_missiles = len(missile_list)
    
    cost_matrix = np.zeros((n_drones, n_missiles))
    for i, d_name in enumerate(unoptimized_drones):
        d_init = DRONES[d_name]["init_pos"]
        d_avg_v = (DRONES[d_name]["speed_range"][0] + DRONES[d_name]["speed_range"][1]) / 2
        for j, m_name in enumerate(missile_list):
            m_init = MISSILES[m_name]["init_pos"]
            m_flight_time = MISSILES[m_name]["flight_time"]
            
            dist = np.linalg.norm(d_init - m_init)
            cost1 = dist / d_avg_v
            cost2 = 1000 / (m_flight_time + 1)
            cost3 = abs(d_avg_v - MISSILE_SPEED) / 100
            cost_matrix[i, j] = cost1 + cost2 + cost3

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = {m: [] for m in missile_list}
    
    assigned_drones_indices = set()
    for r, c in zip(row_ind, col_ind):
        drone_name = unoptimized_drones[r]
        missile_name = missile_list[c]
        assignments[missile_name].append(drone_name)
        assigned_drones_indices.add(r)
        
    # 将剩余的无人机（可能因为导弹数少于无人机数）分配给成本最低的任务
    for i in range(n_drones):
        if i not in assigned_drones_indices:
            drone_name = unoptimized_drones[i]
            min_cost_j = np.argmin(cost_matrix[i])
            missile_name = missile_list[min_cost_j]
            assignments[missile_name].append(drone_name)
            
    return assignments

# 【核心修改】迭代优化主函数
def iterative_optimization(max_iterations=20, improvement_threshold=0.3, max_stall_iter=3):
    # 重置无人机状态
    for d_name in DRONES:
        DRONES[d_name]["smokes"] = []
        DRONES[d_name]["speed"] = None
        DRONES[d_name]["direction"] = None

    prev_total_time = 0
    stall_count = 0

    for iteration in range(max_iterations):
        print(f"\n===== 迭代 {iteration + 1}/{max_iterations} =====")

        # 1. 检查是否所有无人机都已投满烟幕弹
        drones_can_still_drop = [d for d in DRONES if len(DRONES[d]["smokes"]) < DRONES[d]["max_smoke"]]
        if not drones_can_still_drop:
            print("所有无人机均已达到最大投放数量，停止迭代。")
            break

        # 2. 任务分配（为所有还能投放的无人机分配任务）
        assignments = assign_tasks()
        print(f"本轮任务分配: {assignments}")
        
        if not assignments:
             print("无任务可分配，迭代结束。")
             break

        # 3. 对分配到任务的无人机进行优化，尝试添加一枚烟幕弹
        optimized_this_iter = False
        for m_name, drone_names in assignments.items():
            for d_name in drone_names:
                # 如果该无人机还能投放
                if len(DRONES[d_name]["smokes"]) < DRONES[d_name]["max_smoke"]:
                    print(f"正在优化无人机 {d_name} 干扰 {m_name}...")
                    new_smoke = optimize_drone_trajectory(d_name, m_name)

                    if new_smoke:
                        optimized_this_iter = True
                        DRONES[d_name]["smokes"].append(new_smoke)
                        # 更新无人机的速度和方向为最新一次成功投放的参数
                        DRONES[d_name]["speed"] = new_smoke["v"]
                        DRONES[d_name]["direction"] = new_smoke["theta"]
                        print(f"[{d_name}] 成功添加一枚烟幕弹，当前共 {len(DRONES[d_name]['smokes'])} 枚。")
                    else:
                        print(f"[{d_name}] 未能为该无人机找到更优的投放方案。")

        # 4. 路径拟合与波动优化（在每次迭代后对所有无人机进行）
        for d_name, drone in DRONES.items():
            if len(drone["smokes"]) > 1: # 至少有2个点才能拟合
                # 加权拟合直线
                drop_points = []
                weights = []
                for smoke in drone["smokes"]:
                    v_vec = np.array([smoke["v"] * np.cos(smoke["theta"]), smoke["v"] * np.sin(smoke["theta"]), 0])
                    drop_pos = drone["init_pos"] + v_vec * smoke["drop_time"]
                    drop_points.append(drop_pos[:2])
                    weights.append(smoke["effective_time"])
                
                drop_points = np.array(drop_points)
                weights = np.array(weights)

                # 拟合方向角
                try:
                    X = np.column_stack([drop_points[:, 0], np.ones(len(drop_points))])
                    W = np.diag(weights)
                    k, b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ drop_points[:, 1])
                    ref_theta = np.arctan(k)
                except np.linalg.LinAlgError:
                    ref_theta = np.mean([s["theta"] for s in drone["smokes"]])

                # 将拟合后的方向更新给无人机
                drone["direction"] = ref_theta


        # 5. 计算当前总遮蔽时长并检查停止条件
        total_effective_time = sum(s["effective_time"] for d in DRONES.values() for s in d["smokes"])
        print(f"\n当前总遮蔽时长: {total_effective_time:.2f}s")

        improvement = total_effective_time - prev_total_time
        print(f"相比上一轮改进量: {improvement:.2f}s")

        if improvement < improvement_threshold:
            stall_count += 1
            print(f"连续无显著改进次数: {stall_count}/{max_stall_iter}")
            if stall_count >= max_stall_iter:
                print(f"连续{max_stall_iter}轮无显著改进，停止迭代。")
                break
        else:
            stall_count = 0

        prev_total_time = total_effective_time

    # 迭代结束后，整理最终结果
    all_smokes = []
    for d_name, d_data in DRONES.items():
        for smoke in d_data["smokes"]:
            # 确保每个烟幕弹信息都包含无人机编号
            smoke_with_drone = smoke.copy()
            smoke_with_drone["drone"] = d_name
            all_smokes.append(smoke_with_drone)
            
    return all_smokes

# ============================ 5. 结果输出与可视化（不变） ============================

def save_result(smokes, filename="smoke_optimization_result.xlsx"):
    data = []
    for i, smoke in enumerate(smokes, 1):
        det_pos = smoke.get("det_pos") 
        if det_pos is None:
            det_pos = np.array([0,0,0])
            
        data.append({
            "烟幕弹编号": f"S{i}",
            "无人机编号": smoke["drone"],
            "速度(m/s)": round(smoke["v"], 2),
            "方向(°)": round(np.degrees(smoke["theta"]), 2),
            "投放时刻(s)": round(smoke["drop_time"], 2),
            "起爆延迟(s)": round(smoke["det_delay"], 2),
            "起爆时刻(s)": round(smoke["det_time"], 2),
            "起爆点X(m)": round(det_pos[0], 2),
            "起爆点Y(m)": round(det_pos[1], 2),
            "起爆点Z(m)": round(det_pos[2], 2),
            "干扰导弹": smoke["missile"],
            "有效遮蔽时长(s)": round(smoke["effective_time"], 2)
        })
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False, engine="openpyxl")
    print(f"结果已保存到 {filename}")
    return df

def visualize_result(smokes):
    if not smokes:
        print("无有效数据可可视化")
        return

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 绘制真目标、导弹轨迹、无人机轨迹
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = TRUE_TARGET["center"][0] + TRUE_TARGET["r"] * np.cos(theta)
    y_circle = TRUE_TARGET["center"][1] + TRUE_TARGET["r"] * np.sin(theta)
    ax1.plot(x_circle, y_circle, "r-", label="真目标投影")
    ax1.scatter(TRUE_TARGET["center"][0], TRUE_TARGET["center"][1], c="r", marker="*", s=200, label="真目标中心")

    # 导弹轨迹
    colors = ["red", "green", "blue"]
    for i, (m_name, m_data) in enumerate(MISSILES.items()):
        t_range = np.linspace(0, m_data["flight_time"], 100)
        pos_list = [get_missile_pos(m_name, t)[:2] for t in t_range]
        pos_arr = np.array(pos_list)
        ax1.plot(pos_arr[:, 0], pos_arr[:, 1], f"{colors[i]}--", label=f"{m_name}轨迹")
        ax1.scatter(m_data["init_pos"][0], m_data["init_pos"][1], c=colors[i], s=100, label=f"{m_name}初始位置")

    # 无人机轨迹和烟幕
    drone_colors = ["orange", "purple", "cyan", "magenta", "brown"]
    for i, (d_name, d_data) in enumerate(DRONES.items()):
        if not d_data["smokes"]:
            continue
        
        # 确保无人机有速度和方向用于绘制轨迹
        if d_data["speed"] is not None and d_data["direction"] is not None:
            last_smoke = max(d_data["smokes"], key=lambda s: s["drop_time"])
            t_range = np.linspace(0, last_smoke["drop_time"], 50)
            v_vec = np.array([d_data["speed"] * np.cos(d_data["direction"]), d_data["speed"] * np.sin(d_data["direction"]), 0])
            pos_list = [d_data["init_pos"] + v_vec * t for t in t_range]
            pos_arr = np.array(pos_list)
            ax1.plot(pos_arr[:, 0], pos_arr[:, 1], f"{drone_colors[i]}-", label=f"{d_name}轨迹")
        
        ax1.scatter(d_data["init_pos"][0], d_data["init_pos"][1], c=drone_colors[i], s=100, marker="^", label=f"{d_name}初始位置")

        # 烟幕起爆点
        for smoke in d_data["smokes"]:
            det_pos = smoke.get("det_pos")
            if det_pos is not None:
                ax1.scatter(det_pos[0], det_pos[1], c=drone_colors[i], s=50, alpha=0.7)
                circle = plt.Circle((det_pos[0], det_pos[1]), SMOKE_RADIUS, color=drone_colors[i], alpha=0.2)
                ax1.add_patch(circle)

    ax1.set_xlabel("X(m)")
    ax1.set_ylabel("Y(m)")
    ax1.set_title("无人机、导弹轨迹及烟幕起爆点")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. 各导弹遮蔽时长
    missile_effect = {m: 0 for m in MISSILES.keys()}
    for smoke in smokes:
        missile_effect[smoke["missile"]] += smoke["effective_time"]
    ax2.bar(missile_effect.keys(), missile_effect.values(), color=colors)
    ax2.set_xlabel("导弹编号")
    ax2.set_ylabel("总遮蔽时长(s)")
    ax2.set_title("各导弹总遮蔽时长")
    for m, t in missile_effect.items():
        ax2.text(m, t + 0.5, f"{t:.1f}s", ha="center")

    # 3. 各无人机烟幕弹数量
    drone_smoke_count = {d: len(DRONES[d]["smokes"]) for d in DRONES.keys()}
    ax3.bar(drone_smoke_count.keys(), drone_smoke_count.values(), color=drone_colors)
    ax3.set_xlabel("无人机编号")
    ax3.set_ylabel("烟幕弹数量")
    ax3.set_title("各无人机烟幕弹投放数量")
    for d, cnt in drone_smoke_count.items():
        ax3.text(d, cnt + 0.05, str(cnt), ha="center")

    # 4. 遮蔽时长分布
    effect_times = [smoke["effective_time"] for smoke in smokes]
    if effect_times:
        ax4.hist(effect_times, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
    ax4.set_xlabel("单烟幕弹遮蔽时长(s)")
    ax4.set_ylabel("烟幕弹数量")
    ax4.set_title("单烟幕弹遮蔽时长分布")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("smoke_optimization_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()

# ============================ 主函数执行 ============================

if __name__ == "__main__":
    # 增加最大迭代次数，确保有足够机会为所有无人机找到解
    all_smokes = iterative_optimization(max_iterations=20, improvement_threshold=0.3, max_stall_iter=3)

    if all_smokes:
        result_df = save_result(all_smokes)
        visualize_result(all_smokes)
        print("\n" + "=" * 50)
        print("最终结果汇总：")
        print(f"总烟幕弹数量：{len(all_smokes)}")
        print(f"总遮蔽时长：{sum(s['effective_time'] for s in all_smokes):.2f}s")
        print("\n各无人机投放详情：")
        for d_name, d_data in DRONES.items():
            if d_data["smokes"]:
                total = sum([s["effective_time"] for s in d_data["smokes"]])
                print(f"{d_name}：{len(d_data['smokes'])}枚弹，总遮蔽时长{total:.2f}s")
            else:
                print(f"{d_name}：未找到有效投放方案")
        print("=" * 50)
    else:
        print("未找到有效的烟幕弹投放方案")