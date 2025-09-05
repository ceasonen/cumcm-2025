#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试版本：验证基本遮蔽逻辑
"""

import numpy as np
import matplotlib.pyplot as plt

# 基本参数
g = 9.8
uav_start = np.array([17800, 0, 1800])
missile_start = np.array([20000, 0, 2000])
target_center = np.array([0, 200, 0])
decoy_pos = np.array([0, 0, 0])
missile_speed = 300.0
smoke_radius = 10.0

# 导弹方向
missile_dir = (decoy_pos - missile_start) / np.linalg.norm(decoy_pos - missile_start)
print(f"导弹方向: {missile_dir}")

def simple_intersection_test(p1, p2, center, radius):
    """简化的线段球体相交测试"""
    # 向量计算
    d = p2 - p1
    f = p1 - center
    
    a = np.dot(d, d)
    if a < 1e-15:
        return np.linalg.norm(f) <= radius
    
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius**2
    
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return False
    
    sqrt_d = np.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2*a)
    t2 = (-b + sqrt_d) / (2*a)
    
    return (t1 <= 1.0) and (t2 >= 0.0)

def test_single_scenario():
    """测试一个具体的烟幕投放场景"""
    # 测试参数
    theta = 3.14  # 180度，向左飞
    velocity = 100.0
    t1 = 10.0  # 10秒后投放
    t2 = 5.0   # 5秒引信延迟
    
    print(f"\n=== 测试场景 ===")
    print(f"无人机方向: {theta:.2f}rad ({np.degrees(theta):.1f}度)")
    print(f"无人机速度: {velocity}m/s")
    print(f"投放时间: {t1}s")
    print(f"引信延迟: {t2}s")
    
    # 计算投放位置
    direction = np.array([np.cos(theta), np.sin(theta), 0])
    deploy_pos = uav_start + velocity * t1 * direction
    print(f"投放位置: {deploy_pos}")
    
    # 计算爆炸位置
    explode_xy = deploy_pos[:2] + velocity * t2 * direction[:2]
    explode_z = deploy_pos[2] - 0.5 * g * t2**2
    explode_pos = np.array([explode_xy[0], explode_xy[1], explode_z])
    print(f"爆炸位置: {explode_pos}")
    
    explode_time = t1 + t2
    print(f"爆炸时间: {explode_time}s")
    
    # 检查几个关键时刻的遮蔽情况
    test_times = [explode_time + i for i in range(0, 10, 2)]
    
    for t in test_times:
        # 导弹位置
        missile_pos = missile_start + missile_speed * t * missile_dir
        
        # 烟幕位置（考虑下沉）
        time_since_explode = t - explode_time
        smoke_z = explode_z - 3.0 * time_since_explode  # 3m/s下沉
        smoke_pos = np.array([explode_pos[0], explode_pos[1], smoke_z])
        
        print(f"\n时间 {t:.1f}s:")
        print(f"  导弹位置: {missile_pos}")
        print(f"  烟幕位置: {smoke_pos}")
        
        # 测试与目标中心的遮蔽
        is_blocked = simple_intersection_test(missile_pos, target_center, smoke_pos, smoke_radius)
        print(f"  遮蔽目标中心: {is_blocked}")
        
        # 计算距离信息
        missile_to_target_dist = np.linalg.norm(target_center - missile_pos)
        smoke_to_target_dist = np.linalg.norm(target_center - smoke_pos)
        missile_to_smoke_dist = np.linalg.norm(smoke_pos - missile_pos)
        
        print(f"  导弹到目标距离: {missile_to_target_dist:.1f}m")
        print(f"  烟幕到目标距离: {smoke_to_target_dist:.1f}m")
        print(f"  导弹到烟幕距离: {missile_to_smoke_dist:.1f}m")
        
        if smoke_z < 0:
            print("  烟幕已落地")
            break

def test_optimal_positioning():
    """测试最优投放位置策略"""
    print(f"\n=== 寻找最优投放策略 ===")
    
    # 计算导弹在目标附近的时间
    target_vec = target_center - missile_start
    target_distance = np.dot(target_vec, missile_dir)
    target_time = target_distance / missile_speed
    print(f"导弹到达目标区域时间: {target_time:.1f}s")
    
    # 在目标前方布置烟幕
    # 目标是在导弹到达目标前2-3秒开始遮蔽
    desired_coverage_start = target_time - 5.0
    desired_coverage_end = target_time + 2.0
    
    print(f"期望遮蔽时间窗口: {desired_coverage_start:.1f}s - {desired_coverage_end:.1f}s")
    
    # 在目标与导弹之间选择烟幕位置
    # 烟幕应该在导弹-目标连线上，距离目标适当距离
    missile_pos_at_target_time = missile_start + missile_speed * target_time * missile_dir
    print(f"导弹在目标时间的位置: {missile_pos_at_target_time}")
    
    # 在导弹-目标连线的中点附近放置烟幕
    optimal_smoke_pos = (missile_pos_at_target_time + target_center) / 2
    optimal_smoke_pos[2] = target_center[2] + 50  # 比目标高50米
    print(f"理论最优烟幕位置: {optimal_smoke_pos}")
    
    # 反推所需的无人机参数
    # 假设无人机朝目标方向飞行
    target_direction = (target_center - uav_start) / np.linalg.norm(target_center - uav_start)
    
    print(f"建议无人机飞行方向: {target_direction}")
    print(f"建议无人机方向角: {np.degrees(np.arctan2(target_direction[1], target_direction[0])):.1f}度")

if __name__ == "__main__":
    print("=== 基本遮蔽逻辑调试 ===")
    test_single_scenario()
    test_optimal_positioning()
