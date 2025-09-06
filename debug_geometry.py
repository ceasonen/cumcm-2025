"""
几何分析版本 - 找出为什么遮蔽时间为0
"""

import numpy as np
import math

# 基础参数
MISSILE_INIT = np.array([20000, 0, 2000])
MISSILE_SPEED = 300
TARGET_CENTER = np.array([0, 200, 0])
DRONES = {
    1: np.array([17800, 0, 1800]),
    2: np.array([12000, 1400, 1400]),
    3: np.array([6000, -3000, 700])
}

# 导弹方向
missile_dir = (TARGET_CENTER - MISSILE_INIT) / np.linalg.norm(TARGET_CENTER - MISSILE_INIT)

def analyze_geometry():
    """分析几何关系"""
    print("几何关系分析")
    print("="*50)
    
    print(f"导弹初始位置: {MISSILE_INIT}")
    print(f"目标位置: {TARGET_CENTER}")
    print(f"导弹方向: {missile_dir}")
    
    # 导弹飞行距离和时间
    missile_dist = np.linalg.norm(TARGET_CENTER - MISSILE_INIT)
    missile_time = missile_dist / MISSILE_SPEED
    print(f"导弹飞行距离: {missile_dist:.1f}m")
    print(f"导弹飞行时间: {missile_time:.1f}s")
    
    # 分析每架无人机
    for drone_id, drone_pos in DRONES.items():
        print(f"\n无人机FY{drone_id}:")
        print(f"  位置: {drone_pos}")
        
        # 到目标的距离
        to_target = TARGET_CENTER - drone_pos
        dist_to_target = np.linalg.norm(to_target[:2])  # 水平距离
        height_diff = drone_pos[2] - TARGET_CENTER[2]
        
        print(f"  到目标水平距离: {dist_to_target:.1f}m")
        print(f"  高度差: {height_diff:.1f}m")
        
        # 理想飞行角度（指向目标）
        ideal_theta = math.atan2(to_target[1], to_target[0])
        print(f"  理想飞行角度: {math.degrees(ideal_theta):.1f}°")
        
        # 如果以100m/s飞行，到达目标时间
        fly_time = dist_to_target / 100
        print(f"  以100m/s到达目标时间: {fly_time:.1f}s")
        
        # 烟幕弹下降时间
        fall_time = math.sqrt(2 * height_diff / 9.8) if height_diff > 0 else 0
        print(f"  下降到目标高度时间: {fall_time:.1f}s")

def test_simple_case():
    """测试一个简单情况"""
    print("\n\n简单情况测试")
    print("="*50)
    
    # 测试FY3（最近的无人机）
    drone_pos = DRONES[3]
    
    # 直接朝目标飞行
    to_target = TARGET_CENTER - drone_pos
    theta = math.atan2(to_target[1], to_target[0])
    v = 100
    
    # 飞行时间：到达目标上空
    dist = np.linalg.norm(to_target[:2])
    t_deploy = dist / v
    
    # 起爆延迟：让烟幕在目标高度起爆
    height_diff = drone_pos[2] - TARGET_CENTER[2]
    t_detonate = math.sqrt(2 * height_diff / 9.8) if height_diff > 0 else 5
    
    print(f"测试参数:")
    print(f"  θ = {math.degrees(theta):.1f}°")
    print(f"  v = {v} m/s")
    print(f"  t_deploy = {t_deploy:.1f}s")
    print(f"  t_detonate = {t_detonate:.1f}s")
    
    # 计算起爆位置
    flight_dir = np.array([math.cos(theta), math.sin(theta), 0])
    drop_pos = drone_pos + v * t_deploy * flight_dir
    det_xy = drop_pos[:2] + v * t_detonate * flight_dir[:2]
    det_z = drop_pos[2] - 0.5 * 9.8 * t_detonate**2
    
    print(f"  投放位置: {drop_pos}")
    print(f"  起爆位置: [{det_xy[0]:.1f}, {det_xy[1]:.1f}, {det_z:.1f}]")
    
    # 检查起爆位置到目标的距离
    det_pos = np.array([det_xy[0], det_xy[1], det_z])
    dist_to_target = np.linalg.norm(det_pos[:2] - TARGET_CENTER[:2])
    print(f"  起爆点到目标水平距离: {dist_to_target:.1f}m")
    
    # 起爆时间
    abs_det_time = t_deploy + t_detonate
    print(f"  绝对起爆时间: {abs_det_time:.1f}s")
    print(f"  导弹此时位置: {MISSILE_INIT + MISSILE_SPEED * abs_det_time * missile_dir}")
    
    # 检查导弹到目标的剩余时间
    missile_total_time = np.linalg.norm(TARGET_CENTER - MISSILE_INIT) / MISSILE_SPEED
    remaining_time = missile_total_time - abs_det_time
    print(f"  导弹剩余飞行时间: {remaining_time:.1f}s")
    
    if remaining_time > 0:
        print(f"  ✓ 有机会拦截")
    else:
        print(f"  ✗ 起爆太晚，导弹已过")

def check_interception_window():
    """检查拦截窗口"""
    print("\n\n拦截窗口分析")
    print("="*50)
    
    # 导弹飞行轨迹关键点
    missile_total_time = np.linalg.norm(TARGET_CENTER - MISSILE_INIT) / MISSILE_SPEED
    
    print(f"导弹接近目标的时间窗口: 0 - {missile_total_time:.1f}s")
    
    # 在不同时刻，导弹的位置
    for t in [0, 20, 40, 60, missile_total_time]:
        if t <= missile_total_time:
            missile_pos = MISSILE_INIT + MISSILE_SPEED * t * missile_dir
            dist_to_target = np.linalg.norm(missile_pos - TARGET_CENTER)
            print(f"  t={t:4.1f}s: 导弹位置={missile_pos}, 距目标{dist_to_target:.1f}m")

if __name__ == "__main__":
    analyze_geometry()
    test_simple_case()
    check_interception_window()
