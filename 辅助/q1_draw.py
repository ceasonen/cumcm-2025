import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 根据问题1的参数定义常量 ---

# 目标 (真实目标)
TARGET_CENTER_BASE = np.array([0, 200, 0])
TARGET_RADIUS = 7
TARGET_HEIGHT = 10

# 假目标 (原点)
FAKE_TARGET_POS = np.array([0, 0, 0])

# 导弹 M1
M1_INITIAL_POS = np.array([20000, 0, 2000])
MISSILE_SPEED = 300

# 无人机 FY1
FY1_INITIAL_POS = np.array([17800, 0, 1800])
DRONE_SPEED = 120

# 事件时间
T_DRONE_FLY = 1.5  # 无人机飞行1.5s后投放
T_SMOKE_FALL = 3.6 # 烟幕弹投放3.6s后起爆

# 物理和烟幕参数
G = 9.8
SMOKE_RADIUS = 10 # 烟幕云团半径

# --- 2. 计算所有关键点的精确坐标 ---

# 无人机飞行方向 (朝向假目标)
drone_direction_vec = FAKE_TARGET_POS - FY1_INITIAL_POS
drone_direction_normalized = drone_direction_vec / np.linalg.norm(drone_direction_vec)

# 无人机投放烟幕弹时的位置
drone_drop_pos = FY1_INITIAL_POS + drone_direction_normalized * DRONE_SPEED * T_DRONE_FLY

# 烟幕弹起爆时的位置
# 初始水平速度 = 无人机速度
smoke_initial_velocity_xy = drone_direction_normalized[:2] * DRONE_SPEED
# 水平位移
smoke_displacement_xy = smoke_initial_velocity_xy * T_SMOKE_FALL
# 垂直位移 (自由落体)
smoke_displacement_z = -0.5 * G * T_SMOKE_FALL**2
# 起爆点坐标
smoke_detonation_pos = np.array([
    drone_drop_pos[0] + smoke_displacement_xy[0],
    drone_drop_pos[1] + smoke_displacement_xy[1],
    drone_drop_pos[2] + smoke_displacement_z
])

# 导弹飞行方向 (朝向假目标)
missile_direction_vec = FAKE_TARGET_POS - M1_INITIAL_POS
missile_direction_normalized = missile_direction_vec / np.linalg.norm(missile_direction_vec)

# 烟幕弹起爆的时刻
total_time_at_detonation = T_DRONE_FLY + T_SMOKE_FALL

# 导弹在烟幕弹起爆时的位置
missile_pos_at_detonation = M1_INITIAL_POS + missile_direction_normalized * MISSILE_SPEED * total_time_at_detonation


# --- 3. 绘图 ---

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# A. 绘制真实目标 (圆柱体)
z = np.linspace(0, TARGET_HEIGHT, 50)
theta = np.linspace(0, 2 * np.pi, 50)
theta_grid, z_grid = np.meshgrid(theta, z)
x_grid = TARGET_RADIUS * np.cos(theta_grid) + TARGET_CENTER_BASE[0]
y_grid = TARGET_RADIUS * np.sin(theta_grid) + TARGET_CENTER_BASE[1]
ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.8, color='royalblue', label='真目标')
# 为label添加一个代理点
ax.scatter([], [], [], color='royalblue', s=100, label='真目标 (圆柱)')


# B. 【新增】绘制包裹真目标的外切球
sphere_center = TARGET_CENTER_BASE + np.array([0, 0, TARGET_HEIGHT / 2])
sphere_radius = np.sqrt(TARGET_RADIUS**2 + (TARGET_HEIGHT / 2)**2)
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
x_sphere = sphere_center[0] + sphere_radius * np.cos(u) * np.sin(v)
y_sphere = sphere_center[1] + sphere_radius * np.sin(u) * np.sin(v)
z_sphere = sphere_center[2] + sphere_radius * np.cos(v)
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="c", alpha=0.3, linewidth=1, label='目标外切球')

# C. 绘制假目标
ax.scatter(FAKE_TARGET_POS[0], FAKE_TARGET_POS[1], FAKE_TARGET_POS[2],
           c='blue', marker='s', s=100, label='假目标 (原点)')

# D. 绘制导弹和其轨迹
ax.scatter(M1_INITIAL_POS[0], M1_INITIAL_POS[1], M1_INITIAL_POS[2],
           c='red', marker='^', s=150, label='导弹M1初始位置')
ax.plot([M1_INITIAL_POS[0], missile_pos_at_detonation[0]],
        [M1_INITIAL_POS[1], missile_pos_at_detonation[1]],
        [M1_INITIAL_POS[2], missile_pos_at_detonation[2]], 'r--', label='导弹M1轨迹')
ax.scatter(missile_pos_at_detonation[0], missile_pos_at_detonation[1], missile_pos_at_detonation[2],
           c='red', marker='^', s=100, label='导弹 (起爆瞬间)')


# E. 绘制无人机和其轨迹
ax.scatter(FY1_INITIAL_POS[0], FY1_INITIAL_POS[1], FY1_INITIAL_POS[2],
           c='orange', marker='s', s=150, label='无人机FY1初始位置')
ax.plot([FY1_INITIAL_POS[0], drone_drop_pos[0]],
        [FY1_INITIAL_POS[1], drone_drop_pos[1]],
        [FY1_INITIAL_POS[2], drone_drop_pos[2]], 'k--', label='无人机FY1轨迹')
ax.scatter(drone_drop_pos[0], drone_drop_pos[1], drone_drop_pos[2],
           c='orange', marker='s', s=100, label='投放点')


# F. 绘制烟幕弹
u_smoke, v_smoke = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
x_smoke = smoke_detonation_pos[0] + SMOKE_RADIUS * np.cos(u_smoke) * np.sin(v_smoke)
y_smoke = smoke_detonation_pos[1] + SMOKE_RADIUS * np.sin(u_smoke) * np.sin(v_smoke)
z_smoke = smoke_detonation_pos[2] + SMOKE_RADIUS * np.cos(v_smoke)
ax.plot_surface(x_smoke, y_smoke, z_smoke, color='green', alpha=0.5, label='烟幕弹云团')

# G. 【新增】绘制多条被遮蔽的视线轨迹 (从导弹到目标外切球)
num_lines = 10
# 在外切球表面选择一些点
phi_points = np.linspace(np.pi/4, 3*np.pi/4, num_lines) # 垂直角度
theta_points = np.linspace(np.pi/2, 3*np.pi/2, num_lines) # 水平角度 (取半边)

for i in range(num_lines):
    # 在球面上取点
    target_point = np.array([
        sphere_center[0] + sphere_radius * np.cos(theta_points[i]) * np.sin(phi_points[i]),
        sphere_center[1] + sphere_radius * np.sin(theta_points[i]) * np.sin(phi_points[i]),
        sphere_center[2] + sphere_radius * np.cos(phi_points[i])
    ])
    
    # 绘制从导弹到该点的虚线
    ax.plot([missile_pos_at_detonation[0], target_point[0]],
            [missile_pos_at_detonation[1], target_point[1]],
            [missile_pos_at_detonation[2], target_point[2]], 'b:', alpha=0.6, linewidth=1)

# 添加一个代理线条用于图例
ax.plot([], [], [], 'b:', alpha=0.6, label='遮蔽视线')


# --- 4. 设置图像格式 ---
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.set_title('问题1 - 精确空间布局与视线遮蔽示意图', fontsize=16)

# 调整坐标轴范围以获得更好的视角
ax.set_xlim([0, 21000])
ax.set_ylim([-2000, 2000])
ax.set_zlim([0, 2500])

# 设置合适的视角
ax.view_init(elev=20, azim=240)

# 显示图例
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()