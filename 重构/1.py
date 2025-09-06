import numpy as np
import time

P_UAV_INIT = np.array([17800.0, 0.0, 1800.0])   # 无人机初始位置
V_UAV = 120.0                                   # 无人机速度
P_M_INIT = np.array([20000.0, 0.0, 2000.0])     # 拦截器初始位置
V_M = 300.0                                     # 拦截器速度
P_AIM = np.array([0, 0, 0])                     # 瞄准点

# 目标区域: 底面圆心(0, 200, 0), 半径7, 高10
T_GEOM = {"base": np.array([0, 200, 0]), "r": 7.0, "h": 10.0}

# 烟雾弹
S_PARAMS = {"r": 10.0, "v_fall": 3.0, "life": 20.0, "d1": 1.5, "d2": 3.6}

G = 9.8
DT = 0.001
EPS = 1e-15

def calculate_key_events():
    vec_h = P_AIM[:2] - P_UAV_INIT[:2]
    dir_h = vec_h / np.linalg.norm(vec_h)
    
    # 释放点
    p_release = P_UAV_INIT + np.hstack([dir_h * V_UAV * S_PARAMS["d1"], 0])
    
    # 起爆点
    horizontal_travel = V_UAV * S_PARAMS["d2"]
    vertical_drop = 0.5 * G * S_PARAMS["d2"]**2
    p_detonation = p_release + np.hstack([dir_h * horizontal_travel, -vertical_drop])
    
    t_detonation = S_PARAMS["d1"] + S_PARAMS["d2"]
    return p_detonation, t_detonation

# def generate_target_cloud(geom, n_h=6, n_a=20):
#     base, r, h = geom["base"], geom["r"], geom["h"]
#     heights = np.linspace(base[2], base[2] + h, n_h)
#     angles = np.linspace(0, 2 * np.pi, n_a, endpoint=False)
    
#    
#     cloud = [(base[0] + r * np.cos(a), base[1] + r * np.sin(a), z) 
#               for z in heights for a in angles]
#
#     return np.array(list(set(cloud)))
# def check_los_block(m_pos, t_pos, s_center, s_r):

#     line_vec = t_pos - m_pos
#     oc_vec = s_center - m_pos
    
#     # 避免除以零
#     line_len_sq = np.dot(line_vec, line_vec)
#     if line_len_sq < 1e-12:
#         return np.linalg.norm(oc_vec) <= s_r

#     proj = np.dot(oc_vec, line_vec) / line_len_sq
    
#     closest_pt = m_pos + line_vec * proj
#     return np.linalg.norm(closest_pt - s_center) <= s_r

def generate_target_cloud(geom, n_h=6, n_a=20):
    base, r, h = geom["base"], geom["r"], geom["h"]
    heights = np.linspace(base[2], base[2] + h, n_h)
    angles = np.linspace(0, 2 * np.pi, n_a, endpoint=False)
    
    # 侧面点
    cloud = [(base[0] + r * np.cos(a), base[1] + r * np.sin(a), z) for z in heights for a in angles]
    # 顶面和底面点
    cloud.extend([(base[0] + r * np.cos(a), base[1] + r * np.sin(a), bz) for a in angles for bz in [base[2], base[2]+h]])
    
    return np.array(list(set(cloud))) 

def check_los_block(m_pos, t_pos, s_center, s_r):
    line_vec = t_pos - m_pos
    oc_vec = s_center - m_pos
    
    # 投影视线向量在线段上的长度比例
    proj = np.dot(oc_vec, line_vec) / np.dot(line_vec, line_vec)
    
    # 找到离烟幕中心最近的点
    closest_pt = m_pos + line_vec * np.clip(proj, 0, 1)
        
    return np.linalg.norm(closest_pt - s_center) <= s_r

def main_process():
    t0 = time.perf_counter()
    
    p_det, t_det = calculate_key_events()
    target_points = generate_target_cloud(T_GEOM)
    
    time_axis = np.arange(t_det, t_det + S_PARAMS["life"], DT)
    
    # 拦截器飞行方向
    m_dir = (P_AIM - P_M_INIT) / np.linalg.norm(P_AIM - P_M_INIT)

    m_trajectory = P_M_INIT + m_dir * V_M * time_axis[:, np.newaxis]
    time_since_det = time_axis - t_det
    s_trajectory = p_det - np.array([0, 0, S_PARAMS["v_fall"]]) * time_since_det[:, np.newaxis]

    # 迭代
    is_blocked_timeline = np.zeros(len(time_axis), dtype=bool)
    for i in range(len(time_axis)):
        # 只有当所有目标点都被遮挡时，才算完全遮蔽
        all_blocked = all(check_los_block(m_trajectory[i], pt, s_trajectory[i], S_PARAMS["r"]) for pt in target_points)
        if all_blocked:
            is_blocked_timeline[i] = True

    total_blocked_duration = np.sum(is_blocked_timeline) * DT
    intervals = []
    
    diff = np.diff(is_blocked_timeline.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    if is_blocked_timeline[0]: 
        starts = np.insert(starts, 0, 0)
    if is_blocked_timeline[-1]: 
        ends = np.append(ends, len(time_axis) - 1)
        
    for s_idx, e_idx in zip(starts, ends):
        intervals.append((time_axis[s_idx], time_axis[e_idx]))

    # 结果输出 
    print(f"烟幕起爆于: {t_det:.2f}秒")
    print(f"目标离散化点数: {len(target_points)}")
    print("\n" + "="*50)
    print(f"总有效遮蔽时长: {total_blocked_duration:.4f} 秒")
    print("="*50)
    
    if intervals:
        print("\n遮蔽时间段:")
        for i, (t_start, t_end) in enumerate(intervals, 1):
            print(f"  时段{i}: {t_start:.3f}s -> {t_end:.3f}s ")
    else:
        print("\n无有效遮蔽发生")
        


if __name__ == "__main__":
    main_process()