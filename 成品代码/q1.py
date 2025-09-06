import numpy as np
import time
# 计算导弹轨迹与烟雾遮挡的时间段
class Parameters:
    def __init__(self):
        self.g = 9.8
        self.dt = 0.001
        self.eps = 1e-15
        # 目标设置
        self.fake_target = np.array([0, 0, 0])
        self.real_target_base = np.array([0, 200, 0])
        self.target_R = 7.0
        self.target_H = 10.0
        # 飞行器参数
        self.uav_start = np.array([17800, 0, 1800])
        self.uav_v = 120.0
        self.delay1 = 1.5
        self.delay2 = 3.6
        # 烟雾参数
        self.smoke_R = 10.0
        self.smoke_fall_v = 3.0
        self.smoke_life = 20.0
        # 导弹参数
        self.missile_start = np.array([20000, 0, 2000])
        self.missile_v = 300.0

# 其他参数和计算函数定义
def get_release_pos(params):
    p1 = params.uav_start
    target = params.fake_target
    vec_h = target[:2] - p1[:2]
    d_h = np.linalg.norm(vec_h)
    if d_h > params.eps:
        dir_h = vec_h / d_h
    else:
        dir_h = np.zeros(2)
    dist = params.uav_v * params.delay1
    xy = p1[:2] + dir_h * dist
    return np.array([xy[0], xy[1], p1[2]])

# 计算弹药起爆位置，基于释放位置和飞行时间
def get_explode_pos(params, release_pt):
    target = params.fake_target
    vec_h = target[:2] - release_pt[:2]
    d_h = np.linalg.norm(vec_h)
    
    if d_h > params.eps:
        dir_h = vec_h / d_h
    else:
        dir_h = np.zeros(2)

    dist = params.uav_v * params.delay2
    xy = release_pt[:2] + dir_h * dist
    fall_h = 0.5 * params.g * params.delay2**2
    z = release_pt[2] - fall_h
    
    return np.array([xy[0], xy[1], z])

def create_mesh_points(params):
    base = params.real_target_base
    R = params.target_R
    H = params.target_H
    pts = []

    # 对圆周进行采样
    n_circle = 20
    n_height = 6
    angles = np.linspace(0, 2*np.pi, n_circle, endpoint=False)
    heights = np.linspace(base[2], base[2] + H, n_height)
    
    # 外表面点
    for theta in angles:
        x = base[0] + R * np.cos(theta)
        y = base[1] + R * np.sin(theta)
        pts.append([x, y, base[2]])
        pts.append([x, y, base[2] + H])
    
    for h in heights:
        for theta in angles:
            x = base[0] + R * np.cos(theta)
            y = base[1] + R * np.sin(theta)
            pts.append([x, y, h])
    
    # 内部的点
    for r in [0, R*0.3, R*0.7]:
        for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
            for h in np.linspace(base[2], base[2] + H, 3):
                x = base[0] + r * np.cos(theta)
                y = base[1] + r * np.sin(theta)
                pts.append([x, y, h])
    
    return np.unique(np.array(pts), axis=0)

def ray_sphere_hit(start, end, center, radius):
    ray = end - start
    to_center = center - start
    
    a = np.dot(ray, ray)
    if a < 1e-12:
        return np.linalg.norm(to_center) <= radius
    
    b = -2 * np.dot(ray, to_center)
    c = np.dot(to_center, to_center) - radius**2
    
    disc = b*b - 4*a*c
    if disc < 0:
        return False
    
    t1 = (-b - np.sqrt(disc)) / (2*a)
    t2 = (-b + np.sqrt(disc)) / (2*a)
    
    return t1 <= 1.0 and t2 >= 0.0

def is_blocked(missile_pt, smoke_center, smoke_r, mesh_pts):
    blocked_count = 0
    for pt in mesh_pts:
        if ray_sphere_hit(missile_pt, pt, smoke_center, smoke_r):
            blocked_count += 1
    return blocked_count == len(mesh_pts)

def run_analysis():
    t0 = time.time()
    params = Parameters()
    
    # 关键点计算
    release_pt = get_release_pos(params)
    explode_pt = get_explode_pos(params, release_pt)
    mesh = create_mesh_points(params)
    
    print(f"UAV初始: {params.uav_start}")
    print(f"释放位置: {release_pt}")
    print(f"爆炸位置: {explode_pt}")
    print(f"网格点数: {len(mesh)}")
    
    # 导弹轨迹向量
    aim_vec = params.fake_target - params.missile_start
    aim_dist = np.linalg.norm(aim_vec)
    aim_dir = aim_vec / aim_dist if aim_dist > params.eps else np.zeros(3)
    
    # 时间序列
    t_explode = params.delay1 + params.delay2
    t_begin = t_explode
    t_finish = t_explode + params.smoke_life
    time_seq = np.arange(t_begin, t_finish + params.dt, params.dt)
    
    print(f"\n爆炸时刻: {t_explode:.2f}s")
    print(f"分析区间: [{t_begin:.2f}, {t_finish:.2f}]s")
    
    # 主循环
    block_duration = 0.0
    intervals = []
    prev_blocked = False
    
    for t in time_seq:
        # 导弹当前位置
        missile_now = params.missile_start + aim_dir * params.missile_v * t
        
        # 烟雾当前位置
        drop_time = t - t_explode
        smoke_now = np.array([
            explode_pt[0],
            explode_pt[1],
            explode_pt[2] - params.smoke_fall_v * drop_time
        ])
        
        # 遮挡检测
        blocked = is_blocked(missile_now, smoke_now, params.smoke_R, mesh)
        
        if blocked:
            block_duration += params.dt
        
        # 记录时间段
        if blocked and not prev_blocked:
            intervals.append({"t1": t})
        elif not blocked and prev_blocked and intervals:
            intervals[-1]["t2"] = t - params.dt
        
        prev_blocked = blocked
    
    # 收尾处理
    if intervals and "t2" not in intervals[-1]:
        intervals[-1]["t2"] = t_finish
    
    # 结果输出
    print("\n" + "="*60)
    print(f"总遮挡时长: {block_duration:.4f} 秒")
    print("="*60)
    
    if intervals:
        print("\n遮挡区间:")
        for i, seg in enumerate(intervals, 1):
            span = seg["t2"] - seg["t1"]
            print(f"  #{i}: {seg['t1']:.3f}s ~ {seg['t2']:.3f}s (持续{span:.3f}s)")
    else:
        print("\n无有效遮挡")
    
    print(f"\n运行时间: {time.time() - t0:.3f}s")

if __name__ == "__main__":
    run_analysis()