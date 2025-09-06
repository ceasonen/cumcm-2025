"""
问题4: 三架无人机协同烟幕弹投放策略优化
采用分而治之策略：先分别为3架无人机搜索多组候选策略，再通过组合筛选最优解
基于用户提供的正确思路和代码模板
Created on Wed Sep  6 2024
@author: Research Team Alpha
@version: 4.5.0 (基于正确思路重写)
"""

import numpy as np
import pandas as pd
import math
from itertools import product
import random

# 全局常量定义
g = 9.8  # 重力加速度(m/s²)
SMOKE_RADIUS = 10  # 烟幕球半径(m)
SMOKE_EFFECTIVE_TIME = 20  # 烟幕有效时间(s)
SMOKE_SINK_SPEED = 3  # 起爆后下沉速度(m/s)
MISSILE_SPEED = 300  # 导弹速度(m/s)
TRUE_TARGET_RADIUS = 7  # 真目标半径(m)
TRUE_TARGET_HEIGHT = 10  # 真目标高度(m)
TRUE_TARGET_CENTER = np.array([0, 200, 0])  # 真目标下底面圆心

# 无人机初始参数（FY1、FY2、FY3）
DRONES_INIT = {
    1: np.array([17800, 0, 1800]),  # FY1: (x0,y0,z0)
    2: np.array([12000, 1400, 1400]),  # FY2
    3: np.array([6000, -3000, 700])  # FY3
}

# 导弹初始参数
MISSILE_INIT = np.array([20000, 0, 2000])  # 初始位置
missile_dir = -MISSILE_INIT / np.linalg.norm(MISSILE_INIT)  # 运动方向

# 预生成真目标采样点（覆盖底面、侧面、顶面）
def generate_true_target_points():
    points = []
    cx, cy, cz = TRUE_TARGET_CENTER

    # 底面点（z=0）
    angles = np.linspace(0, 2*np.pi, 30)
    for alpha in angles:
        x = cx + TRUE_TARGET_RADIUS * math.cos(alpha)
        y = cy + TRUE_TARGET_RADIUS * math.sin(alpha)
        points.append([x, y, cz])

    # 侧面点（z从0到10）
    angles = np.linspace(0, 2*np.pi, 30)
    heights = np.linspace(0, TRUE_TARGET_HEIGHT, 10)
    for z in heights:
        for alpha in angles:
            x = cx + TRUE_TARGET_RADIUS * math.cos(alpha)
            y = cy + TRUE_TARGET_RADIUS * math.sin(alpha)
            points.append([x, y, z])

    # 顶面点（z=10）
    for alpha in angles:
        x = cx + TRUE_TARGET_RADIUS * math.cos(alpha)
        y = cy + TRUE_TARGET_RADIUS * math.sin(alpha)
        points.append([x, y, cz + TRUE_TARGET_HEIGHT])

    return np.array(points)

TRUE_TARGET_POINTS = generate_true_target_points()

# 无人机位置计算
def drone_position(drone_id, t, theta, v):
    x0, y0, z0 = DRONES_INIT[drone_id]
    x = x0 + v * math.cos(theta) * t
    y = y0 + v * math.sin(theta) * t
    return np.array([x, y, z0])

# 烟幕位置计算（修正递归问题）
def calculate_detonate_point(drone_id, theta, v, td, tau):
    xd, yd, zd = drone_position(drone_id, td, theta, v)
    delta_t = tau
    xe = xd + v * math.cos(theta) * delta_t
    ye = yd + v * math.sin(theta) * delta_t
    ze = zd - 0.5 * g * delta_t ** 2
    return np.array([xe, ye, ze])

def smoke_position(drone_id, t, theta, v, td, tau):
    if t < td - 1e-6:  # 未投放
        return drone_position(drone_id, t, theta, v)
    elif td - 1e-6 <= t < td + tau - 1e-6:  # 投放后未起爆
        xd, yd, zd = drone_position(drone_id, td, theta, v)
        delta_t = t - td
        x = xd + v * math.cos(theta) * delta_t
        y = yd + v * math.sin(theta) * delta_t
        z = zd - 0.5 * g * delta_t **2
        return np.array([x, y, z])
    elif td + tau - 1e-6 <= t <= td + tau + SMOKE_EFFECTIVE_TIME + 1e-6:  # 起爆后
        xe, ye, ze = calculate_detonate_point(drone_id, theta, v, td, tau)
        delta_t = t - (td + tau)
        z = ze - SMOKE_SINK_SPEED * delta_t
        return np.array([xe, ye, z])
    else:  # 失效
        return None

# 线段与球相交判定（基于Q2的成功算法）
def segment_sphere_intersection(P1, P2, C, r):
    """
    基于Q2成功验证的线段-球体相交算法
    返回相交比例，> 0 表示有相交
    """
    P1 = np.array(P1, dtype=np.float64)
    P2 = np.array(P2, dtype=np.float64)
    C = np.array(C, dtype=np.float64)
    
    segment_vector = P2 - P1
    segment_to_center = P1 - C
    
    a = np.dot(segment_vector, segment_vector)
    if a < 1e-15:  # 线段退化为点
        return np.linalg.norm(P1 - C) <= r
    
    b = 2 * np.dot(segment_to_center, segment_vector)
    c = np.dot(segment_to_center, segment_to_center) - r * r
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < -1e-15:
        return False
    if discriminant < 0:
        discriminant = 0
    
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)
    
    # 计算相交区间
    interval_start = max(0.0, min(t1, t2))
    interval_end = min(1.0, max(t1, t2))
    
    intersection_ratio = max(0.0, interval_end - interval_start)
    return intersection_ratio > 1e-15

# 计算单个策略的遮蔽区间（基于Q2成功逻辑）
def get_smoke_intervals(drone_id, theta, v, td, tau, time_step=0.1):
    """基于Q2验证成功的遮蔽时间计算逻辑"""
    intervals = []
    
    # 计算关键时间点
    abs_det_time = td + tau
    smoke_end_time = abs_det_time + SMOKE_EFFECTIVE_TIME
    
    # 计算起爆位置
    drone_pos = DRONES_INIT[drone_id]
    flight_dir = np.array([math.cos(theta), math.sin(theta), 0])
    
    drop_pos = drone_pos + v * td * flight_dir
    det_xy = drop_pos[:2] + v * tau * flight_dir[:2]
    det_z = drop_pos[2] - 0.5 * g * tau**2
    
    if det_z < 5.0:  # 起爆点太低
        return []
    
    det_pos = np.array([det_xy[0], det_xy[1], det_z])
    
    # 计算导弹到达目标的时间
    missile_flight_time = np.linalg.norm(MISSILE_INIT) / MISSILE_SPEED
    analysis_end = min(smoke_end_time, missile_flight_time)
    
    if abs_det_time >= analysis_end:
        return []
    
    # 时间序列分析
    current_start = None
    total_coverage = 0.0
    
    for t in np.arange(abs_det_time, analysis_end + time_step, time_step):
        # 烟幕当前位置
        sink_time = t - abs_det_time
        current_smoke_z = det_z - SMOKE_SINK_SPEED * sink_time
        
        if current_smoke_z < 2.0:  # 烟幕落地
            if current_start is not None:
                intervals.append((current_start, t))
                current_start = None
            continue
        
        smoke_center = np.array([det_pos[0], det_pos[1], current_smoke_z])
        
        # 导弹当前位置
        missile_pos = MISSILE_INIT + MISSILE_SPEED * t * missile_dir
        
        # 检查是否遮蔽所有目标点（使用Q2的检测逻辑）
        blocked_count = 0
        total_samples = len(TRUE_TARGET_POINTS)
        
        for target_point in TRUE_TARGET_POINTS:
            if segment_sphere_intersection(missile_pos, target_point, smoke_center, SMOKE_RADIUS):
                blocked_count += 1
        
        # 需要遮蔽所有采样点才算完全遮蔽
        if blocked_count == total_samples:
            if current_start is None:
                current_start = t
        else:
            if current_start is not None:
                intervals.append((current_start, t))
                current_start = None
    
    # 处理最后的区间
    if current_start is not None:
        intervals.append((current_start, analysis_end))
    
    # 过滤无效短区间
    valid_intervals = []
    for s, e in intervals:
        if e - s > 0.05:  # 最小有效时间
            valid_intervals.append((s, e))
    
    return valid_intervals

# 合并区间并计算总时长
def merge_and_calculate(intervals_list):
    all_intervals = []
    for intervals in intervals_list:
        all_intervals.extend(intervals)

    if not all_intervals:
        return [], 0.0

    # 排序并合并
    all_intervals.sort()
    merged = [list(all_intervals[0])]
    for s, e in all_intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + 0.1:
            merged[-1][1] = max(last_e, e)
        else:
            merged.append([s, e])

    total = sum(e - s for s, e in merged if e - s > 0.1)
    return merged, total

# 为单个无人机生成随机策略
def generate_drone_strategies(drone_id, num_strategies=5, min_effective=1.0):
    strategies = []
    x0, y0, z0 = DRONES_INIT[drone_id]

    # 参数范围（基于Q2成功经验调整）
    if drone_id == 1:  # 最远的无人机
        theta_range = (-0.05, 0.05)  # 更集中的方向角
        v_range = (95, 115)  # 更合理的速度范围
        td_range = (160, 180)  # 投放延迟
        tau_range = (15, 25)  # 起爆延迟
    elif drone_id == 2:  # 中等距离
        theta_range = (-0.1, 0.1)
        v_range = (90, 110)
        td_range = (100, 130)
        tau_range = (12, 20)
    else:  # drone_id == 3，最近的无人机
        theta_range = (-0.15, 0.15)
        v_range = (85, 105)
        td_range = (50, 80)
        tau_range = (8, 15)

    # 生成策略直到满足数量要求
    max_attempts = num_strategies * 20  # 避免无限循环
    attempts = 0
    
    while len(strategies) < num_strategies and attempts < max_attempts:
        attempts += 1
        theta = random.uniform(*theta_range)
        v = random.uniform(*v_range)
        td = random.uniform(*td_range)
        tau = random.uniform(*tau_range)

        # 计算该策略的有效遮蔽时间
        intervals = get_smoke_intervals(drone_id, theta, v, td, tau)
        _, total = merge_and_calculate([intervals])

        # 只保留有效时间足够的策略
        if total >= min_effective:
            strategies.append({
                'drone_id': drone_id,
                'theta': theta,
                'v': v,
                'td': td,
                'tau': tau,
                'intervals': intervals,
                'duration': total
            })
            print(f"生成无人机{drone_id}策略{len(strategies)}，有效时间：{total:.2f}s")

    # 如果策略不够，降低要求再生成
    while len(strategies) < num_strategies:
        theta = random.uniform(*theta_range)
        v = random.uniform(*v_range)
        td = random.uniform(*td_range)
        tau = random.uniform(*tau_range)

        intervals = get_smoke_intervals(drone_id, theta, v, td, tau)
        _, total = merge_and_calculate([intervals])

        strategies.append({
            'drone_id': drone_id,
            'theta': theta,
            'v': v,
            'td': td,
            'tau': tau,
            'intervals': intervals,
            'duration': total
        })
        print(f"生成无人机{drone_id}策略{len(strategies)}（降低要求），有效时间：{total:.2f}s")

    return strategies

# 组合策略并找到最优解
def find_best_combination(fy1_strats, fy2_strats, fy3_strats):
    best_total = 0
    best_combination = None
    best_intervals = []

    # 遍历所有组合（5×5×5=125种组合）
    total_combinations = len(fy1_strats) * len(fy2_strats) * len(fy3_strats)
    print(f"\n开始评估{total_combinations}种策略组合...")

    for i, (s1, s2, s3) in enumerate(product(fy1_strats, fy2_strats, fy3_strats)):
        # 合并三个策略的区间
        all_intervals = [s1['intervals'], s2['intervals'], s3['intervals']]
        merged, total = merge_and_calculate(all_intervals)

        # 计算重叠率（越低越好）
        individual_sum = s1['duration'] + s2['duration'] + s3['duration']
        overlap_rate = (individual_sum - total) / individual_sum if individual_sum > 0 else 1.0

        # 优先选择总时间长且重叠率低的组合
        if total > best_total or (total == best_total and overlap_rate < 0.5):
            best_total = total
            best_combination = (s1, s2, s3)
            best_intervals = merged
            print(f"找到更优组合（{i+1}/{total_combinations}）：总时间{total:.2f}s，重叠率{overlap_rate:.2f}")

    return best_combination, best_intervals, best_total

# 保存结果到Excel
def save_results(best_comb, best_intervals, best_total, save_path="result2.xlsx"):
    if best_comb is None:
        print("未找到有效组合")
        return
        
    s1, s2, s3 = best_comb
    data = []

    # 单个无人机策略详情
    for s in [s1, s2, s3]:
        intervals_str = "; ".join([f"[{s:.1f},{e:.1f}]" for s,e in s['intervals'] if e-s>0.1])
        data.append([
            f"FY{s['drone_id']}",
            f"{s['theta']:.4f}",
            f"{s['v']:.1f}",
            f"{s['td']:.1f}",
            f"{s['tau']:.1f}",
            f"{s['duration']:.2f}s",
            intervals_str
        ])

    # 总结果
    total_intervals_str = "; ".join([f"[{s:.1f},{e:.1f}]" for s,e in best_intervals if e-s>0.1])
    data.append([
        "总计", "", "", "", "",
        f"{best_total:.2f}s",
        total_intervals_str
    ])

    # 保存
    df = pd.DataFrame(data, columns=[
        "无人机", "方向角(rad)", "速度(m/s)", "投放延迟(s)",
        "起爆延迟(s)", "有效时间", "遮蔽区间"
    ])
    
    try:
        df.to_excel(save_path, index=False, engine="openpyxl")
        print(f"\n结果已保存至 {save_path}")
    except:
        df.to_csv(save_path.replace('.xlsx', '.csv'), index=False, encoding='utf-8-sig')
        print(f"\n结果已保存至 {save_path.replace('.xlsx', '.csv')}")

# 主程序
if __name__ == "__main__":
    print("="*50 + " 生成单个无人机策略 " + "="*50)
    
    # 多轮尝试，避免随机性导致的局部最优
    best_overall_total = 0
    best_overall_comb = None
    best_overall_intervals = None
    
    for round_num in range(3):  # 最多3轮
        print(f"\n第{round_num+1}轮尝试:")
        
        # 步骤1：为每个无人机生成5种有效策略
        fy1_strategies = generate_drone_strategies(1, num_strategies=5, min_effective=1.0)
        fy2_strategies = generate_drone_strategies(2, num_strategies=5, min_effective=1.0)
        fy3_strategies = generate_drone_strategies(3, num_strategies=5, min_effective=1.0)

        # 步骤2：组合策略并找到最优解
        print("\n" + "="*30 + " 寻找最优策略组合 " + "="*30)
        best_comb, best_intervals, best_total = find_best_combination(
            fy1_strategies, fy2_strategies, fy3_strategies
        )

        print(f"第{round_num+1}轮最优结果: {best_total:.2f}s")
        
        if best_total > best_overall_total:
            best_overall_total = best_total
            best_overall_comb = best_comb
            best_overall_intervals = best_intervals
        
        # 如果达到目标，提前结束
        if best_overall_total >= 10.0:
            print(f"已达到目标！提前结束")
            break

    # 步骤3：保存结果
    print("\n" + "="*50 + " 最终优化结果 " + "="*50)
    if best_overall_total < 10.0:
        print(f"最优组合总遮蔽时间为{best_overall_total:.2f}s，未达到10s目标")
        print("建议：多次运行程序尝试不同的随机种子")
    else:
        print(f"成功：最优组合总遮蔽时间为{best_overall_total:.2f}s，满足要求")

    if best_overall_comb:
        save_results(best_overall_comb, best_overall_intervals, best_overall_total)
    else:
        print("未找到任何有效策略组合")