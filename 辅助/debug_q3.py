#!/usr/bin/env python3
"""
调试q3.py的核心问题
"""
import numpy as np

# 重新定义配置
class TestConfig:
    def __init__(self):
        self.g = 9.8
        
        # 位置信息
        self.target_center = np.array([0, 200, 0], dtype=np.float64)
        self.target_R = 7.0
        self.target_H = 10.0
        self.decoy_pos = np.array([0, 0, 0], dtype=np.float64)
        
        self.uav_start = np.array([17800, 0, 1800], dtype=np.float64)
        self.missile_start = np.array([20000, 0, 2000], dtype=np.float64)
        self.missile_speed = 300.0
        
        # 烟幕参数
        self.smoke_radius = 10.0
        self.smoke_descent = 3.0
        self.smoke_duration = 20.0
        
        # 计算导弹方向
        self.missile_dir = (self.decoy_pos - self.missile_start)
        self.missile_dir = self.missile_dir / np.linalg.norm(self.missile_dir)
        
        print(f"导弹起点: {self.missile_start}")
        print(f"假目标: {self.decoy_pos}")
        print(f"导弹方向: {self.missile_dir}")
        print(f"导弹速度: {self.missile_speed}")

def test_basic_geometry():
    """测试基本几何计算"""
    config = TestConfig()
    
    # 测试导弹轨迹
    print("\n=== 导弹轨迹测试 ===")
    for t in [0, 10, 20, 30, 40, 50, 60]:
        missile_pos = config.missile_start + config.missile_speed * t * config.missile_dir
        print(f"t={t}s: 导弹位置 {missile_pos}")
        
        # 距离目标的距离
        dist_to_target = np.linalg.norm(missile_pos - config.target_center)
        print(f"      距真目标距离: {dist_to_target:.1f}m")
    
    # 测试烟幕位置
    print("\n=== 烟幕测试 ===")
    # 假设在t=5s投放，t2=3s引信延迟
    theta = 0.0  # 朝向假目标
    v = 100.0    # 速度100m/s
    t1 = 5.0     # 投放时间
    t2 = 3.0     # 引信延迟
    
    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    deploy_pos = config.uav_start + v * t1 * direction
    
    explode_xy = deploy_pos[:2] + v * t2 * direction[:2]
    explode_z = deploy_pos[2] - 0.5 * config.g * t2**2
    explode_pos = np.array([explode_xy[0], explode_xy[1], explode_z])
    
    print(f"投放位置: {deploy_pos}")
    print(f"爆炸位置: {explode_pos}")
    print(f"爆炸时间: {t1 + t2}s")
    
    # 测试关键时刻的遮蔽
    explode_time = t1 + t2
    for dt in [0, 1, 2, 3, 5, 10]:
        current_time = explode_time + dt
        
        # 导弹位置
        missile_pos = config.missile_start + config.missile_speed * current_time * config.missile_dir
        
        # 烟幕位置（下沉）
        current_smoke_z = explode_z - config.smoke_descent * dt
        current_smoke_pos = np.array([explode_pos[0], explode_pos[1], current_smoke_z])
        
        print(f"\nt={current_time}s:")
        print(f"  导弹位置: {missile_pos}")
        print(f"  烟幕位置: {current_smoke_pos}")
        
        # 距离计算
        smoke_to_target = np.linalg.norm(current_smoke_pos - config.target_center)
        missile_to_target = np.linalg.norm(missile_pos - config.target_center)
        
        print(f"  烟幕到目标距离: {smoke_to_target:.1f}m")
        print(f"  导弹到目标距离: {missile_to_target:.1f}m")
        
        # 简单遮蔽判断：烟幕是否在导弹和目标之间
        # 计算导弹-目标连线上离烟幕最近的点
        missile_to_target_vec = config.target_center - missile_pos
        missile_to_smoke_vec = current_smoke_pos - missile_pos
        
        if np.linalg.norm(missile_to_target_vec) > 1e-10:
            proj_length = np.dot(missile_to_smoke_vec, missile_to_target_vec) / np.linalg.norm(missile_to_target_vec)
            proj_ratio = proj_length / np.linalg.norm(missile_to_target_vec)
            
            if 0 <= proj_ratio <= 1:
                # 烟幕投影在导弹-目标连线上
                closest_point = missile_pos + proj_ratio * missile_to_target_vec
                dist_to_line = np.linalg.norm(current_smoke_pos - closest_point)
                
                print(f"  烟幕到连线距离: {dist_to_line:.1f}m")
                print(f"  投影比例: {proj_ratio:.3f}")
                
                if dist_to_line <= config.smoke_radius:
                    print(f"  >>> 有效遮蔽! <<<")
                else:
                    print(f"  >>> 未遮蔽 (距离太远) <<<")
            else:
                print(f"  >>> 未遮蔽 (不在连线上) <<<")

def test_segment_sphere_intersection():
    """测试线段球体相交函数"""
    print("\n=== 线段球体相交测试 ===")
    
    def segment_sphere_intersection(p1, p2, sphere_center, sphere_radius):
        """线段-球体相交计算"""
        d = p2 - p1
        f = p1 - sphere_center
        a = np.dot(d, d)
        
        if a < 1e-15:
            return 1.0 if np.linalg.norm(f) <= sphere_radius + 1e-15 else 0.0
        
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - sphere_radius**2
        discriminant = b*b - 4*a*c
        
        if discriminant < -1e-15:
            return 0.0
        if discriminant < 0:
            discriminant = 0.0
        
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2*a)
        t2 = (-b + sqrt_discriminant) / (2*a)
        
        t_start = max(0.0, min(t1, t2))
        t_end = min(1.0, max(t1, t2))
        
        return max(0.0, t_end - t_start)
    
    # 测试案例
    # 案例1：线段完全穿过球体
    p1 = np.array([0, 0, 0])
    p2 = np.array([10, 0, 0])
    sphere_center = np.array([5, 0, 0])
    sphere_radius = 2.0
    
    ratio = segment_sphere_intersection(p1, p2, sphere_center, sphere_radius)
    print(f"案例1 (穿透): 相交比例 = {ratio:.3f}")
    
    # 案例2：线段与球体相切
    p1 = np.array([0, 2, 0])
    p2 = np.array([10, 2, 0])
    ratio = segment_sphere_intersection(p1, p2, sphere_center, sphere_radius)
    print(f"案例2 (相切): 相交比例 = {ratio:.3f}")
    
    # 案例3：线段完全错过球体
    p1 = np.array([0, 5, 0])
    p2 = np.array([10, 5, 0])
    ratio = segment_sphere_intersection(p1, p2, sphere_center, sphere_radius)
    print(f"案例3 (错过): 相交比例 = {ratio:.3f}")

if __name__ == "__main__":
    print("开始调试q3.py核心问题...")
    test_basic_geometry()
    test_segment_sphere_intersection()
