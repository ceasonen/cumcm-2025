#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题4: 三架无人机协同烟幕弹投放策略优化
调试版本 - 找出为什么遮蔽时间总是0
"""

import numpy as np
import math

# 基本参数
g = 9.8
SMOKE_RADIUS = 10
SMOKE_TIME = 20
SMOKE_SINK_SPEED = 3
MISSILE_SPEED = 300
TARGET_CENTER = np.array([0, 200, 0])
TARGET_RADIUS = 7
TARGET_HEIGHT = 10
MISSILE_START = np.array([20000, 0, 2000])

# 无人机位置
DRONES = {
    1: np.array([17800, 0, 1800]),
    2: np.array([12000, 1400, 1400]),
    3: np.array([6000, -3000, 700])
}

# 导弹方向
missile_dir = -MISSILE_START / np.linalg.norm(MISSILE_START)

def test_single_strategy():
    """测试一个简单的策略"""
    print("="*50)
    print("调试单个策略")
    print("="*50)
    
    # 测试FY3的一个简单策略
    drone_id = 3
    drone_pos = DRONES[drone_id]
    target_pos = TARGET_CENTER
    
    # 计算更优的参数 - 让烟幕在导弹-目标线上
    # 选择一个合适的时间点进行拦截（比如40秒）
    intercept_time = 40.0
    intercept_point = MISSILE_START + MISSILE_SPEED * intercept_time * missile_dir
    intercept_point[2] = max(intercept_point[2], 150)  # 确保高度足够
    
    print(f"  拦截时间: {intercept_time}s")
    print(f"  拦截点: {intercept_point}")
    
    # 计算无人机需要的参数来到达这个拦截点
    to_intercept = intercept_point - drone_pos
    theta = math.atan2(to_intercept[1], to_intercept[0])
    v = 100.0
    
    # 计算投放时间（提前投放，让烟幕弹能及时到达）
    horizontal_distance = np.linalg.norm(to_intercept[:2])
    flight_time_needed = horizontal_distance / v
    
    # 投放延迟设置为让烟幕在拦截时间起爆
    t_deploy = max(5, intercept_time - flight_time_needed - 8)  # 提前8秒起爆
    t_detonate = max(5, min(15, intercept_time - t_deploy - flight_time_needed + 8))
    
    # 重新计算实际的起爆位置
    flight_dir = np.array([math.cos(theta), math.sin(theta), 0])
    drop_pos = drone_pos + v * t_deploy * flight_dir
    det_xy = drop_pos[:2] + v * t_detonate * flight_dir[:2]
    det_z = drop_pos[2] - 0.5 * g * t_detonate**2
    det_pos = np.array([det_xy[0], det_xy[1], det_z])
    abs_det_time = t_deploy + t_detonate
    
    print(f"测试参数:")
    print(f"  无人机: FY{drone_id}")
    print(f"  初始位置: {drone_pos}")
    print(f"  目标位置: {target_pos}")
    print(f"  方向角: {math.degrees(theta):.2f}°")
    print(f"  速度: {v} m/s")
    print(f"  投放延迟: {t_deploy} s")
    print(f"  起爆延迟: {t_detonate} s")
    print(f"  投放点: {drop_pos}")
    print(f"  起爆点: {det_pos}")
    print(f"  起爆时间: {abs_det_time} s")
    
    if det_z < 5:
        print(f"  ❌ 起爆点太低: {det_z} m")
        return 0
    
    # 检查在关键时刻的遮蔽效果
    # 选择几个测试时间点
    test_times = [abs_det_time + 1, abs_det_time + 5, abs_det_time + 10]
    
    total_coverage = 0.0
    
    for t in test_times:
        print(f"\n  时间 {t:.1f}s:")
        
        # 导弹位置
        missile_pos = MISSILE_START + MISSILE_SPEED * t * missile_dir
        print(f"    导弹位置: {missile_pos}")
        
        # 烟幕位置
        sink_time = t - abs_det_time
        smoke_z = det_z - SMOKE_SINK_SPEED * sink_time
        
        if smoke_z < 2:
            print(f"    ❌ 烟幕已落地: {smoke_z} m")
            continue
            
        smoke_center = np.array([det_pos[0], det_pos[1], smoke_z])
        print(f"    烟幕位置: {smoke_center}")
        
        # 检查几个关键目标点
        test_points = [
            TARGET_CENTER,  # 目标底部中心
            TARGET_CENTER + np.array([0, 0, TARGET_HEIGHT]),  # 目标顶部中心
            TARGET_CENTER + np.array([TARGET_RADIUS, 0, TARGET_HEIGHT/2])  # 目标侧面
        ]
        
        blocked_count = 0
        for point in test_points:
            # 简单的距离检查
            to_point = point - missile_pos
            to_smoke = smoke_center - missile_pos
            
            # 检查烟幕是否在导弹到目标的路径上
            point_dist = np.linalg.norm(to_point)
            smoke_dist = np.linalg.norm(to_smoke)
            
            if smoke_dist < point_dist:  # 烟幕在前方
                # 计算距离线的距离
                cross = np.cross(to_point[:2], to_smoke[:2])
                distance_to_line = abs(cross) / np.linalg.norm(to_point[:2])
                
                if distance_to_line < SMOKE_RADIUS:
                    blocked_count += 1
                    print(f"      ✓ 遮蔽点 {point}: 距离线{distance_to_line:.1f}m")
                else:
                    print(f"      ❌ 未遮蔽点 {point}: 距离线{distance_to_line:.1f}m")
            else:
                print(f"      ❌ 烟幕在目标后方")
        
        if blocked_count >= len(test_points) * 0.7:
            total_coverage += 1.0
            print(f"    ✓ 该时刻有效遮蔽")
        else:
            print(f"    ❌ 该时刻遮蔽不足")
    
    print(f"\n  总遮蔽评分: {total_coverage}")
    return total_coverage

def test_missile_trajectory():
    """测试导弹轨迹"""
    print("="*50)
    print("调试导弹轨迹")
    print("="*50)
    
    print(f"导弹起点: {MISSILE_START}")
    print(f"目标位置: {TARGET_CENTER}")
    print(f"导弹方向: {missile_dir}")
    
    # 计算导弹到达目标的时间
    to_target = TARGET_CENTER - MISSILE_START
    distance = np.linalg.norm(to_target)
    arrival_time = distance / MISSILE_SPEED
    
    print(f"导弹距离: {distance:.1f} m")
    print(f"导弹到达时间: {arrival_time:.1f} s")
    
    # 检查几个时间点的导弹位置
    for t in [0, 10, 20, 30, 40, 50, 60]:
        pos = MISSILE_START + MISSILE_SPEED * t * missile_dir
        dist_to_target = np.linalg.norm(pos - TARGET_CENTER)
        print(f"  时间 {t}s: 位置{pos}, 距目标{dist_to_target:.1f}m")

if __name__ == "__main__":
    print("问题4调试 - 找出算法问题")
    
    # 先测试导弹轨迹
    test_missile_trajectory()
    
    print()
    
    # 测试单个策略
    score = test_single_strategy()
    
    print(f"\n最终结果: {score}")
