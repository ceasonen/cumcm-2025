#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3：利用无人机FY1投放3枚烟幕干扰弹实施对M1的干扰
基于前两问的思路，采用环环相扣的优化策略
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class SmokeEvent:
    """单个烟幕事件的数据结构"""
    deploy_time: float
    fuse_delay: float
    deploy_pos: np.ndarray
    explode_pos: np.ndarray
    explode_time: float
    coverage_intervals: List[Tuple[float, float]]
    total_coverage: float

class TripleSmokeConfig:
    def __init__(self):
        self.G = 9.8
        self.TIME_PRECISION = 0.1  # 平衡精度和速度
        self.SPATIAL_PRECISION = 1e-12
        
        # 目标配置
        self.fake_target = np.array([0, 0, 0], dtype=np.float64)
        self.real_target_center = np.array([0, 200, 0], dtype=np.float64)
        self.target_radius = 7.0
        self.target_height = 10.0
        
        # 平台配置
        self.uav_initial = np.array([17800, 0, 1800], dtype=np.float64)
        self.missile_initial = np.array([20000, 0, 2000], dtype=np.float64)
        self.missile_speed = 300.0
        
        # 烟幕配置
        self.smoke_radius = 10.0
        self.smoke_descent_rate = 3.0
        self.smoke_lifetime = 20.0
        
        # 协调配置
        self.min_interval = 1.0
        self.mission_duration = 60.0
        
        # 计算导弹轨迹函数
        direction = self.fake_target - self.missile_initial
        self.missile_direction = direction / np.linalg.norm(direction)
        
        def missile_pos_func(t):
            return self.missile_initial + self.missile_direction * self.missile_speed * t
        
        self.missile_pos_func = missile_pos_func
        
        # 真目标位置
        self.real_target_pos = self.real_target_center

def generate_target_mesh(config, density=1.2):
    """生成目标区域网格，调整密度以获得几百个节点"""
    center = config.real_target_center
    radius = config.target_radius
    height = config.target_height
    
    # 根据密度调整网格间距
    base_spacing = 1.0
    spacing = base_spacing / density
    
    # 计算网格范围
    x_range = np.arange(center[0] - radius, center[0] + radius + spacing, spacing)
    y_range = np.arange(center[1] - radius, center[1] + radius + spacing, spacing)
    z_range = np.arange(center[2], center[2] + height + spacing, spacing)
    
    mesh_points = []
    
    for x in x_range:
        for y in y_range:
            for z in z_range:
                # 检查是否在圆柱体内
                distance_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if distance_from_center <= radius and center[2] <= z <= center[2] + height:
                    mesh_points.append([x, y, z])
    
    return np.array(mesh_points, dtype=np.float64)

def merge_time_intervals(intervals_list):
    """合并重叠的时间区间"""
    if not intervals_list:
        return [], 0.0
    
    # 展平所有区间
    all_intervals = []
    for intervals in intervals_list:
        all_intervals.extend(intervals)
    
    if not all_intervals:
        return [], 0.0
    
    # 排序并合并
    sorted_intervals = sorted(all_intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # 重叠或相邻
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    total_time = sum(end - start for start, end in merged)
    return merged, total_time

def compute_single_smoke_performance(config, velocity, deploy_time, fuse_delay, target_mesh):
    """计算单个烟幕弹的性能"""
    
    # 计算投放位置
    uav_direction = (config.fake_target - config.uav_initial) / np.linalg.norm(config.fake_target - config.uav_initial)
    deploy_pos = config.uav_initial + uav_direction * velocity * deploy_time
    
    # 计算爆炸时间和位置
    explode_time = deploy_time + fuse_delay
    
    # 烟幕弹的运动（重力影响）
    initial_velocity = uav_direction * velocity
    fall_time = fuse_delay
    
    explode_pos = deploy_pos + initial_velocity * fall_time + 0.5 * np.array([0, 0, -config.G]) * fall_time**2
    
    # 检查爆炸位置是否合理
    if explode_pos[2] < 0:
        # 落地了，无效
        return SmokeEvent(
            deploy_time=deploy_time,
            fuse_delay=fuse_delay,
            deploy_pos=deploy_pos,
            explode_pos=None,
            explode_time=explode_time,
            coverage_intervals=[],
            total_coverage=0.0
        )
    
    # 分析烟幕遮蔽效果
    smoke_center = explode_pos.copy()
    
    # 计算有效时间范围
    start_time = explode_time
    end_time = min(explode_time + config.smoke_lifetime, config.mission_duration)
    
    # 时间采样分析
    timeline = np.arange(start_time, end_time + config.TIME_PRECISION, config.TIME_PRECISION)
    
    coverage_indicators = []
    for t in timeline:
        # 烟幕随时间下沉
        current_smoke_pos = smoke_center - np.array([0, 0, config.smoke_descent_rate * (t - explode_time)])
        
        # 检查是否遮蔽导弹到目标的视线
        missile_pos = config.missile_pos_func(t)
        
        # 计算导弹到目标的直线是否与烟幕球相交
        blocked = sphere_intersects_line(current_smoke_pos, config.smoke_radius, 
                                       config.real_target_pos, missile_pos)
        coverage_indicators.append(blocked)
    
    # 找出连续的遮蔽区间
    intervals = []
    in_coverage = False
    interval_start = None
    
    for i, is_covered in enumerate(coverage_indicators):
        current_time = timeline[i]
        
        if is_covered and not in_coverage:
            interval_start = current_time
            in_coverage = True
        elif not is_covered and in_coverage:
            if interval_start is not None:
                intervals.append((interval_start, current_time))
            in_coverage = False
    
    # 处理最后一个区间
    if in_coverage and interval_start is not None:
        intervals.append((interval_start, timeline[-1]))
    
    total_coverage_time = sum(end - start for start, end in intervals)
    
    return SmokeEvent(
        deploy_time=deploy_time,
        fuse_delay=fuse_delay,
        deploy_pos=deploy_pos,
        explode_pos=explode_pos,
        explode_time=explode_time,
        coverage_intervals=intervals,
        total_coverage=total_coverage_time
    )

def sphere_intersects_line(sphere_center, sphere_radius, line_start, line_end):
    """检查球体是否与线段相交"""
    # 向量计算
    d = line_end - line_start
    f = line_start - sphere_center
    
    # 求解二次方程 t^2*(d.d) + 2*t*(f.d) + (f.f - r^2) = 0
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - sphere_radius**2
    
    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        return False
    
    # 计算交点参数
    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2*a)
    t2 = (-b + discriminant) / (2*a)
    
    # 检查交点是否在线段上
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)

def evaluate_triple_smoke_strategy(params, target_mesh, config):
    """评估三枚烟幕弹策略的总体性能"""
    velocity = params[0]
    deploy_times = params[1:4]
    fuse_delays = params[4:7]
    
    # 确保投放时间间隔约束
    sorted_times = sorted(deploy_times)
    for i in range(1, len(sorted_times)):
        if sorted_times[i] - sorted_times[i-1] < config.min_interval:
            return -1000.0  # 惩罚违反约束的解
    
    # 计算每个烟幕弹的性能
    smoke_events = []
    for i in range(3):
        smoke_event = compute_single_smoke_performance(
            config, velocity, deploy_times[i], fuse_delays[i], target_mesh
        )
        smoke_events.append(smoke_event)
    
    # 合并遮蔽时间
    all_intervals = [event.coverage_intervals for event in smoke_events]
    merged_intervals, total_merged_time = merge_time_intervals(all_intervals)
    
    # 策略奖励
    bonus = 0.0
    
    # 多段遮蔽奖励
    if len(merged_intervals) > 1:
        bonus += 0.1
    
    # 早期部署奖励
    if min(deploy_times) < 2.0:
        bonus += 0.05
    
    # 避免过度重叠惩罚
    individual_total = sum(event.total_coverage for event in smoke_events)
    if total_merged_time < individual_total:
        overlap_penalty = (individual_total - total_merged_time) * 0.05
        bonus -= overlap_penalty
    
    return total_merged_time + bonus

class TripleSmokeOptimizer:
    def __init__(self, objective_func, bounds, swarm_size=20, max_iter=40,
                 c1=2.0, c2=2.0, w_max=0.9, w_min=0.4):
        
        self.objective_func = objective_func
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        
        self.dim = len(bounds)
        
        # 粒子群状态
        self.positions = np.zeros((swarm_size, self.dim))
        self.velocities = np.zeros((swarm_size, self.dim))
        self.personal_best_pos = np.zeros((swarm_size, self.dim))
        self.personal_best_scores = np.full(swarm_size, -np.inf)
        
        self.global_best_pos = np.zeros(self.dim)
        self.global_best_score = -np.inf
        self.history = []
        
        self._initialize_swarm()
    
    def _initialize_swarm(self):
        """初始化粒子群"""
        for i in range(self.swarm_size):
            for j in range(self.dim):
                min_val, max_val = self.bounds[j]
                self.positions[i, j] = np.random.uniform(min_val, max_val)
                
                vel_range = max_val - min_val
                self.velocities[i, j] = np.random.uniform(-0.1*vel_range, 0.1*vel_range)
            
            # 特殊处理：确保投放时间间隔约束
            if self.dim >= 4:
                base_time = np.random.uniform(0, 5)
                for k in range(3):
                    self.positions[i, 1+k] = base_time + k * (1.0 + np.random.uniform(0, 1))
            
            # 评估初始适应度
            score = self.objective_func(self.positions[i])
            self.personal_best_pos[i] = self.positions[i].copy()
            self.personal_best_scores[i] = score
            
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_pos = self.positions[i].copy()
    
    def _apply_bounds(self, position, dim):
        """应用边界约束"""
        min_val, max_val = self.bounds[dim]
        return np.clip(position, min_val, max_val)
    
    def _apply_velocity_limit(self, velocity, dim):
        """应用速度限制"""
        min_val, max_val = self.bounds[dim]
        max_vel = 0.2 * (max_val - min_val)
        return np.clip(velocity, -max_vel, max_vel)
    
    def optimize(self):
        """执行优化"""
        for iteration in range(self.max_iter):
            # 动态惯性权重
            w = self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iter)
            
            # 评估所有粒子
            scores = []
            for pos in self.positions:
                scores.append(self.objective_func(pos))
            
            # 更新最优解
            for i in range(self.swarm_size):
                if scores[i] > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_pos[i] = self.positions[i].copy()
                
                if scores[i] > self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_pos = self.positions[i].copy()
                
                # 更新速度和位置
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                cognitive = self.c1 * r1 * (self.personal_best_pos[i] - self.positions[i])
                social = self.c2 * r2 * (self.global_best_pos - self.positions[i])
                
                new_velocity = w * self.velocities[i] + cognitive + social
                
                # 速度限制
                for d in range(self.dim):
                    new_velocity[d] = self._apply_velocity_limit(new_velocity[d], d)
                
                self.velocities[i] = new_velocity
                
                # 位置更新
                new_position = self.positions[i] + new_velocity
                
                # 边界约束
                for d in range(self.dim):
                    new_position[d] = self._apply_bounds(new_position[d], d)
                
                # 投放时间间隔约束修正
                if self.dim >= 4:
                    deploy_times = new_position[1:4].copy()
                    deploy_times.sort()
                    for k in range(1, 3):
                        if deploy_times[k] - deploy_times[k-1] < 1.0:
                            deploy_times[k] = deploy_times[k-1] + 1.0 + np.random.uniform(0, 0.2)
                    new_position[1:4] = deploy_times
                
                self.positions[i] = new_position
            
            self.history.append(self.global_best_score)
            
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"迭代 {iteration+1}/{self.max_iter}, 最优性能: {self.global_best_score:.6f}")
        
        return self.global_best_pos, self.global_best_score, self.history

def main_optimization():
    """主优化流程"""
    start_time = time.time()
    
    # 配置系统
    config = TripleSmokeConfig()
    
    # 生成目标网格（调整密度到几百个节点）
    print("生成真目标网格...")
    target_mesh = generate_target_mesh(config, density=1.2)  # 调整密度
    print(f"网格节点数量: {len(target_mesh)}")
    
    # 优化变量边界
    bounds = [
        (80.0, 120.0),      # 速度
        (0.0, 20.0),        # 投放时间1
        (1.0, 21.0),        # 投放时间2
        (2.0, 22.0),        # 投放时间3
        (1.0, 8.0),         # 引信延迟1
        (1.0, 8.0),         # 引信延迟2
        (1.0, 8.0)          # 引信延迟3
    ]
    
    # 目标函数
    def objective(params):
        return evaluate_triple_smoke_strategy(params, target_mesh, config)
    
    # 多次试验
    num_trials = 2
    results = []
    
    print(f"\n开始 {num_trials} 次优化试验...")
    
    np.random.seed(42)
    
    for trial in range(num_trials):
        print(f"\n=== 试验 {trial+1}/{num_trials} ===")
        
        optimizer = TripleSmokeOptimizer(
            objective_func=objective,
            bounds=bounds,
            swarm_size=20,
            max_iter=40,
            c1=1.5,
            c2=2.5,
            w_max=0.9,
            w_min=0.4
        )
        
        best_params, best_score, convergence = optimizer.optimize()
        
        # 验证结果
        verification_score = evaluate_triple_smoke_strategy(best_params, target_mesh, config)
        
        # 解析参数
        velocity_opt = best_params[0]
        deploy_times_opt = best_params[1:4]
        fuse_delays_opt = best_params[4:7]
        
        # 计算详细信息
        smoke_details = []
        for i in range(3):
            smoke_event = compute_single_smoke_performance(
                config, velocity_opt, deploy_times_opt[i], fuse_delays_opt[i], target_mesh
            )
            smoke_details.append(smoke_event)
        
        # 合并结果
        all_intervals = [event.coverage_intervals for event in smoke_details]
        merged_intervals, total_time = merge_time_intervals(all_intervals)
        
        result = {
            'trial': trial + 1,
            'velocity': velocity_opt,
            'deploy_times': deploy_times_opt,
            'fuse_delays': fuse_delays_opt,
            'smoke_events': smoke_details,
            'merged_intervals': merged_intervals,
            'total_coverage': verification_score,
            'convergence': convergence
        }
        results.append(result)
        
        print(f"试验 {trial+1} 完成，总遮蔽时间: {verification_score:.4f} 秒")
    
    # 找到最优结果
    best_idx = np.argmax([r['total_coverage'] for r in results])
    best_result = results[best_idx]
    
    execution_time = time.time() - start_time
    
    # 输出结果
    print("\n" + "="*80)
    print("【问题3：三枚烟幕干扰弹协同投放优化结果】")
    print(f"总执行时间: {execution_time:.2f} 秒")
    print(f"最优试验: 试验 {best_result['trial']}")
    print(f"总遮蔽时间: {best_result['total_coverage']:.6f} 秒")
    print(f"无人机速度: {best_result['velocity']:.4f} m/s")
    
    print("\n烟幕弹详细信息:")
    for i, event in enumerate(best_result['smoke_events'], 1):
        print(f"  烟幕弹 {i}:")
        print(f"    投放时间: {event.deploy_time:.4f} s")
        print(f"    引信延迟: {event.fuse_delay:.4f} s")
        print(f"    爆炸时间: {event.explode_time:.4f} s")
        print(f"    投放位置: {event.deploy_pos.round(2)}")
        if event.explode_pos is not None:
            print(f"    爆炸位置: {event.explode_pos.round(2)}")
        print(f"    单独遮蔽时间: {event.total_coverage:.4f} s")
    
    print(f"\n合并后遮蔽区间:")
    for i, (start, end) in enumerate(best_result['merged_intervals'], 1):
        print(f"  区间 {i}: {start:.4f}s ~ {end:.4f}s (持续 {end-start:.4f}s)")
    
    print("="*80)
    
    # 保存结果
    save_to_excel(best_result, config)
    plot_results(results, best_idx)
    
    print("\n结果已保存至 result1.xlsx")
    print("收敛图已保存至 q3_fixed_convergence.png")

def save_to_excel(result, config):
    """保存结果到Excel"""
    data_rows = []
    
    for i, event in enumerate(result['smoke_events'], 1):
        if event.explode_pos is not None:
            row = {
                '序号': i,
                '投放时间(s)': round(event.deploy_time, 4),
                '投放位置X(m)': round(event.deploy_pos[0], 2),
                '投放位置Y(m)': round(event.deploy_pos[1], 2), 
                '投放位置Z(m)': round(event.deploy_pos[2], 2),
                '引信延迟(s)': round(event.fuse_delay, 4),
                '爆炸时间(s)': round(event.explode_time, 4),
                '爆炸位置X(m)': round(event.explode_pos[0], 2),
                '爆炸位置Y(m)': round(event.explode_pos[1], 2),
                '爆炸位置Z(m)': round(event.explode_pos[2], 2),
                '单独遮蔽时间(s)': round(event.total_coverage, 4)
            }
        else:
            row = {
                '序号': i,
                '投放时间(s)': round(event.deploy_time, 4),
                '投放位置X(m)': round(event.deploy_pos[0], 2),
                '投放位置Y(m)': round(event.deploy_pos[1], 2),
                '投放位置Z(m)': round(event.deploy_pos[2], 2),
                '引信延迟(s)': round(event.fuse_delay, 4),
                '爆炸时间(s)': 'N/A',
                '爆炸位置X(m)': 'N/A',
                '爆炸位置Y(m)': 'N/A',
                '爆炸位置Z(m)': 'N/A',
                '单独遮蔽时间(s)': 0
            }
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # 摘要
    summary = {
        '参数': ['无人机速度(m/s)', '总遮蔽时间(s)', '遮蔽区间数量'],
        '数值': [
            round(result['velocity'], 4),
            round(result['total_coverage'], 6),
            len(result['merged_intervals'])
        ]
    }
    summary_df = pd.DataFrame(summary)
    
    # 保存到Excel
    try:
        with pd.ExcelWriter('result1.xlsx', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='烟幕弹投放详情', index=False)
            summary_df.to_excel(writer, sheet_name='优化结果摘要', index=False)
        print("结果已保存至 result1.xlsx")
    except (ImportError, ModuleNotFoundError):
        print("\n警告：未安装openpyxl模块，结果将保存为CSV文件")
        df.to_csv('result1_details.csv', index=False, encoding='utf-8-sig')
        summary_df.to_csv('result1_summary.csv', index=False, encoding='utf-8-sig')
        print("详细结果已保存至 result1_details.csv")
        print("摘要结果已保存至 result1_summary.csv")

def plot_results(results, best_idx):
    """绘制收敛曲线"""
    plt.figure(figsize=(12, 8))
    
    for i, result in enumerate(results):
        if i == best_idx:
            plt.plot(result['convergence'], 'r-', linewidth=3, 
                    label=f'试验 {result["trial"]} (最优)', alpha=0.9)
        else:
            plt.plot(result['convergence'], '--', linewidth=1.5, 
                    label=f'试验 {result["trial"]}', alpha=0.7)
    
    plt.title('问题3：三枚烟幕弹协同投放PSO优化收敛曲线', fontsize=16, fontweight='bold')
    plt.xlabel('迭代次数', fontsize=14)
    plt.ylabel('适应度值（总遮蔽时间 秒）', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig('q3_fixed_convergence.png', dpi=300, bbox_inches='tight')
    # 移除plt.show()避免阻塞
    print("收敛图已保存")

if __name__ == "__main__":
    main_optimization()
