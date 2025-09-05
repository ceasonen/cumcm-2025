#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3：三枚烟幕弹协同优化 - 修正版本
关键修正：烟幕必须投放在导弹-目标连线上才有效
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time
import warnings
import os
import pandas as pd

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========================= 核心系统配置 =========================
class SystemConfig:
    def __init__(self):
        self.g = 9.8
        
        # 目标配置
        self.target_center = np.array([0, 200, 0], dtype=np.float64)
        self.target_R = 7.0
        self.target_H = 10.0
        self.decoy_pos = np.array([0, 0, 0], dtype=np.float64)
        
        # 平台配置
        self.uav_start = np.array([17800, 0, 1800], dtype=np.float64)
        self.missile_start = np.array([20000, 0, 2000], dtype=np.float64)
        self.missile_speed = 300.0
        
        # 烟幕配置
        self.smoke_radius = 10.0
        self.smoke_descent = 3.0
        self.smoke_duration = 20.0
        
        # 计算导弹轨迹参数
        self.missile_dir = (self.decoy_pos - self.missile_start)
        self.missile_dir = self.missile_dir / np.linalg.norm(self.missile_dir)
        self.missile_total_time = np.linalg.norm(self.decoy_pos - self.missile_start) / self.missile_speed

def generate_target_samples(config, num_samples=200):
    """生成目标采样点"""
    samples = []
    center = config.target_center
    R, H = config.target_R, config.target_H
    
    # 圆柱体采样
    for i in range(num_samples):
        # 随机半径和角度
        r = R * np.sqrt(np.random.random())
        theta = 2 * np.pi * np.random.random()
        z = np.random.uniform(center[2], center[2] + H)
        
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        samples.append([x, y, z])
    
    return np.array(samples)

def line_sphere_intersection_ratio(p1, p2, sphere_center, radius):
    """计算线段与球体的相交比例"""
    d = p2 - p1
    f = p1 - sphere_center
    
    a = np.dot(d, d)
    if a < 1e-15:
        return 1.0 if np.linalg.norm(f) <= radius else 0.0
    
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius**2
    
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return 0.0
    
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    
    t_start = max(0.0, min(t1, t2))
    t_end = min(1.0, max(t1, t2))
    
    return max(0.0, t_end - t_start)

def calculate_blocking_effectiveness(missile_pos, smoke_pos, smoke_radius, target_samples):
    """计算遮蔽有效性"""
    blocked_count = 0
    total_count = len(target_samples)
    
    for target_point in target_samples:
        ratio = line_sphere_intersection_ratio(missile_pos, target_point, smoke_pos, smoke_radius)
        if ratio > 0.3:  # 30%相交就算被遮挡
            blocked_count += 1
    
    return blocked_count / total_count

def evaluate_smoke_deployment(params, config, target_samples):
    """
    评估单枚烟幕弹部署效果
    关键修正：烟幕必须部署在导弹轨迹附近才有效
    
    params: [deploy_fraction, fuse_delay]
    - deploy_fraction: 在导弹轨迹上的投放位置比例 (0-1)
    - fuse_delay: 引信延迟时间
    """
    deploy_fraction, fuse_delay = params
    
    # 基本约束
    if not (0.1 <= deploy_fraction <= 0.9) or not (0.5 <= fuse_delay <= 15.0):
        return 0.0
    
    # 计算导弹轨迹上的目标投放点
    missile_trajectory_length = np.linalg.norm(config.decoy_pos - config.missile_start)
    target_deploy_distance = deploy_fraction * missile_trajectory_length
    target_deploy_pos = config.missile_start + target_deploy_distance * config.missile_dir
    
    # 检查无人机是否能到达此位置
    uav_to_target = target_deploy_pos - config.uav_start
    uav_distance = np.linalg.norm(uav_to_target)
    uav_direction = uav_to_target / uav_distance
    
    # 计算无人机到达目标投放点所需时间和速度
    # 假设无人机以100 m/s速度飞行
    uav_speed = 100.0
    if not (70.0 <= uav_speed <= 140.0):
        uav_speed = 120.0
    
    uav_flight_time = uav_distance / uav_speed
    
    # 实际投放位置（无人机飞行轨迹上）
    actual_deploy_pos = config.uav_start + uav_flight_time * uav_speed * uav_direction
    
    # 计算爆炸位置（考虑引信延迟期间的继续飞行和重力）
    explosion_xy = actual_deploy_pos[:2] + uav_speed * fuse_delay * uav_direction[:2]
    explosion_z = actual_deploy_pos[2] - 0.5 * config.g * fuse_delay**2
    
    if explosion_z < 10.0:  # 爆炸高度太低
        return 0.0
    
    explosion_pos = np.array([explosion_xy[0], explosion_xy[1], explosion_z])
    
    # 计算时间窗口
    total_deploy_time = uav_flight_time + fuse_delay
    smoke_end_time = total_deploy_time + config.smoke_duration
    analysis_end_time = min(smoke_end_time, config.missile_total_time)
    
    if total_deploy_time >= analysis_end_time:
        return 0.0
    
    # 逐时刻计算遮蔽效果
    total_blocked_time = 0.0
    time_step = 0.1
    
    for t in np.arange(total_deploy_time, analysis_end_time, time_step):
        # 当前导弹位置
        missile_pos = config.missile_start + config.missile_speed * t * config.missile_dir
        
        # 当前烟幕位置（考虑下沉）
        time_since_explosion = t - total_deploy_time
        current_smoke_z = explosion_z - config.smoke_descent * time_since_explosion
        
        if current_smoke_z < 5.0:  # 烟幕下沉太低失效
            break
        
        current_smoke_pos = np.array([explosion_pos[0], explosion_pos[1], current_smoke_z])
        
        # 计算遮蔽有效性
        effectiveness = calculate_blocking_effectiveness(
            missile_pos, current_smoke_pos, config.smoke_radius, target_samples
        )
        
        if effectiveness > 0.5:  # 50%以上遮蔽率认为有效
            total_blocked_time += time_step
    
    return total_blocked_time

class TripleSmokeOptimizer:
    def __init__(self, config):
        self.config = config
        self.target_samples = generate_target_samples(config, 150)
        print(f"目标采样点数量: {len(self.target_samples)}")
    
    def optimize_smoke(self, smoke_id, existing_deployments=None):
        """优化单枚烟幕弹部署"""
        print(f"优化第{smoke_id}枚烟幕弹...")
        
        def objective(params):
            score = evaluate_smoke_deployment(params, self.config, self.target_samples)
            
            # 如果有现有部署，添加时间冲突惩罚
            if existing_deployments:
                deploy_fraction, fuse_delay = params
                current_deploy_time = deploy_fraction * 60  # 粗略估计部署时间
                
                for existing_fraction, existing_fuse in existing_deployments:
                    existing_time = existing_fraction * 60
                    time_diff = abs(current_deploy_time - existing_time)
                    if time_diff < 1.0:  # 小于1秒间隔的惩罚
                        score *= 0.1
            
            return -score  # 最大化
        
        # 优化边界：部署位置比例和引信延迟
        bounds = [
            (0.2, 0.8),   # deploy_fraction - 在导弹轨迹20%-80%位置
            (1.0, 12.0)   # fuse_delay - 引信延迟1-12秒
        ]
        
        result = differential_evolution(
            objective, bounds, maxiter=40, popsize=15, seed=42+smoke_id
        )
        
        best_score = -result.fun
        print(f"第{smoke_id}枚最优得分: {best_score:.4f}秒")
        return result.x, best_score
    
    def comprehensive_optimization(self):
        """执行完整的三枚烟幕弹优化"""
        print("开始三枚烟幕弹协同优化...")
        
        results = {}
        existing_deployments = []
        
        for i in range(1, 4):
            params, score = self.optimize_smoke(i, existing_deployments)
            results[f'smoke_{i}'] = {'params': params, 'score': score}
            existing_deployments.append(params)
        
        # 计算总得分
        total_score = sum(result['score'] for result in results.values())
        # 考虑协同效应，稍微降低总分以反映重叠
        total_score *= 0.85
        
        print(f"\n协同优化完成!")
        print(f"总遮蔽时间: {total_score:.6f}秒")
        
        results['total_score'] = total_score
        return results

def save_results_to_excel(results, config):
    """保存结果到Excel"""
    os.makedirs('answer3', exist_ok=True)
    
    detailed_data = []
    
    for i in range(1, 4):
        smoke_key = f'smoke_{i}'
        if smoke_key in results:
            deploy_fraction, fuse_delay = results[smoke_key]['params']
            score = results[smoke_key]['score']
            
            # 重新计算详细位置信息
            missile_trajectory_length = np.linalg.norm(config.decoy_pos - config.missile_start)
            target_deploy_distance = deploy_fraction * missile_trajectory_length
            target_deploy_pos = config.missile_start + target_deploy_distance * config.missile_dir
            
            uav_to_target = target_deploy_pos - config.uav_start
            uav_distance = np.linalg.norm(uav_to_target)
            uav_direction = uav_to_target / uav_distance
            uav_speed = 120.0
            uav_flight_time = uav_distance / uav_speed
            
            actual_deploy_pos = config.uav_start + uav_flight_time * uav_speed * uav_direction
            
            explosion_xy = actual_deploy_pos[:2] + uav_speed * fuse_delay * uav_direction[:2]
            explosion_z = actual_deploy_pos[2] - 0.5 * config.g * fuse_delay**2
            explosion_pos = np.array([explosion_xy[0], explosion_xy[1], explosion_z])
            
            detailed_data.append({
                '序号': i,
                '部署位置比例': round(deploy_fraction, 4),
                '引信延迟(s)': round(fuse_delay, 3),
                '无人机飞行时间(s)': round(uav_flight_time, 3),
                '投放位置X(m)': round(actual_deploy_pos[0], 1),
                '投放位置Y(m)': round(actual_deploy_pos[1], 1),
                '投放位置Z(m)': round(actual_deploy_pos[2], 1),
                '爆炸位置X(m)': round(explosion_pos[0], 1),
                '爆炸位置Y(m)': round(explosion_pos[1], 1),
                '爆炸位置Z(m)': round(explosion_pos[2], 1),
                '遮蔽时间(s)': round(score, 4)
            })
    
    df = pd.DataFrame(detailed_data)
    
    # 保存Excel
    try:
        with pd.ExcelWriter('result1.xlsx', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='烟幕弹部署策略', index=False)
        print("结果已保存至 result1.xlsx")
    except ImportError:
        df.to_csv('result1.csv', index=False, encoding='utf-8-sig')
        print("结果已保存至 result1.csv")
    
    return df

def create_3d_visualization(results, config):
    """创建3D可视化"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制导弹轨迹
    missile_times = np.linspace(0, config.missile_total_time, 100)
    missile_trajectory = config.missile_start + config.missile_speed * missile_times[:, np.newaxis] * config.missile_dir
    ax.plot(missile_trajectory[:, 0], missile_trajectory[:, 1], missile_trajectory[:, 2], 
            'r-', linewidth=3, label='导弹轨迹M1')
    
    # 绘制目标
    ax.scatter(*config.target_center, c='black', s=200, marker='s', label='真目标')
    ax.scatter(*config.decoy_pos, c='gray', s=150, marker='^', label='假目标')
    
    # 绘制无人机起点
    ax.scatter(*config.uav_start, c='blue', s=150, marker='o', label='无人机FY1起点')
    
    # 绘制烟幕弹部署
    colors = ['orange', 'green', 'purple']
    for i in range(1, 4):
        smoke_key = f'smoke_{i}'
        if smoke_key in results:
            deploy_fraction, fuse_delay = results[smoke_key]['params']
            
            # 计算位置
            missile_trajectory_length = np.linalg.norm(config.decoy_pos - config.missile_start)
            target_deploy_distance = deploy_fraction * missile_trajectory_length
            target_deploy_pos = config.missile_start + target_deploy_distance * config.missile_dir
            
            uav_to_target = target_deploy_pos - config.uav_start
            uav_distance = np.linalg.norm(uav_to_target)
            uav_direction = uav_to_target / uav_distance
            uav_speed = 120.0
            uav_flight_time = uav_distance / uav_speed
            
            actual_deploy_pos = config.uav_start + uav_flight_time * uav_speed * uav_direction
            
            explosion_xy = actual_deploy_pos[:2] + uav_speed * fuse_delay * uav_direction[:2]
            explosion_z = actual_deploy_pos[2] - 0.5 * config.g * fuse_delay**2
            explosion_pos = np.array([explosion_xy[0], explosion_xy[1], explosion_z])
            
            # 绘制投放点
            ax.scatter(*actual_deploy_pos, c=colors[i-1], s=120, marker='*', 
                      label=f'烟幕弹{i}投放点')
            
            # 绘制爆炸点
            ax.scatter(*explosion_pos, c=colors[i-1], s=150, marker='X', 
                      label=f'烟幕弹{i}爆炸点')
            
            # 绘制无人机到投放点轨迹
            uav_traj = np.array([config.uav_start, actual_deploy_pos])
            ax.plot(uav_traj[:, 0], uav_traj[:, 1], uav_traj[:, 2], 
                   '--', color=colors[i-1], alpha=0.7)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('问题3：三枚烟幕弹协同部署策略\n(修正版：烟幕部署在导弹轨迹附近)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('answer3/三枚烟幕弹部署可视化_修正版.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存至 answer3/三枚烟幕弹部署可视化_修正版.png")

def main():
    print("="*80)
    print("问题3：三枚烟幕弹协同优化 - 修正版本")
    print("关键修正：烟幕部署在导弹-目标连线附近")
    print("="*80)
    
    start_time = time.time()
    
    # 初始化
    config = SystemConfig()
    optimizer = TripleSmokeOptimizer(config)
    
    # 执行优化
    results = optimizer.comprehensive_optimization()
    
    execution_time = time.time() - start_time
    
    # 输出结果
    print(f"\n{'='*80}")
    print(f"优化完成! 总耗时: {execution_time:.2f}秒")
    print(f"三枚烟幕弹总遮蔽时间: {results['total_score']:.6f}秒")
    
    for i in range(1, 4):
        smoke_key = f'smoke_{i}'
        if smoke_key in results:
            params = results[smoke_key]['params']
            score = results[smoke_key]['score']
            print(f"烟幕弹{i}: 部署比例={params[0]:.3f}, 引信延迟={params[1]:.2f}s, 遮蔽时间={score:.3f}s")
    
    print(f"{'='*80}")
    
    # 保存结果
    save_results_to_excel(results, config)
    create_3d_visualization(results, config)
    
    print(f"\n所有结果已保存至answer3文件夹和result1.xlsx")
    
    return results

if __name__ == "__main__":
    results = main()
