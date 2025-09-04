import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# --- 1. 参数设置 (请在此处修改为您自己的数据) ---

# 保护目标 (圆柱体)
target_radius = 7  # 目标半径 (米)
target_height = 10 # 目标高度 (米)
target_pos = [0, 0, 0] # 目标中心坐标 (x, y, z)

# 无人机位置
uav_pos = [-800, 500, 1000] # 无人机坐标 (x, y, z)

# 烟幕弹爆炸形成的烟幕球
smoke_cloud_1_pos = [-200, 150, 200] # 第一个烟幕球中心坐标
smoke_cloud_2_pos = [-400, 250, 300] # 第二个烟幕球中心坐标
smoke_radius = 25 # 烟幕有效半径 (为了可视化效果，可以适当调整)

# 来袭导弹位置 (示例数据)
missile_positions = np.array([
    [1500, -800, 400],
    [1800, -600, 450],
    [1600, -1000, 350]
])
# 假设导弹都朝向目标原点
missile_directions = -missile_positions / np.linalg.norm(missile_positions, axis=1)[:, np.newaxis]


# --- 2. 3D 场景与样式设置 ---

# 创建 3D 图形
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 设置背景色
fig.patch.set_facecolor('#f0f4f8')
ax.set_facecolor('#f0f4f8')

# 设置坐标轴范围
ax.set_xlim([-1000, 2000])
ax.set_ylim([-1200, 1200])
ax.set_zlim([0, 1200])

# 设置坐标轴标签
ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)

# 设置网格线
ax.grid(True, linestyle='--', color='grey', alpha=0.6)

# 调整视角
ax.view_init(elev=25, azim=-50)


# --- 3. 绘制场景中的元素 ---

# a. 绘制保护目标 (圆柱体)
z = np.linspace(0, target_height, 20)
theta = np.linspace(0, 2 * np.pi, 20)
theta_grid, z_grid = np.meshgrid(theta, z)
x_grid = target_radius * np.cos(theta_grid) + target_pos[0]
y_grid = target_radius * np.sin(theta_grid) + target_pos[1]
ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.8, color='#3498db', rstride=1, cstride=1, label='Protected Target')
# 添加文字标签
ax.text(target_pos[0], target_pos[1], target_height + 50, '保护目标', color='black', fontsize=12, ha='center')


# b. 绘制无人机
ax.scatter(uav_pos[0], uav_pos[1], uav_pos[2], s=200, c='#e74c3c', marker='^', depthshade=True, label='UAV')
ax.text(uav_pos[0], uav_pos[1], uav_pos[2] + 60, '无人机', color='black', fontsize=12, ha='center')

# c. 绘制烟幕球
def plot_sphere(center, radius, ax, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, linewidth=0, alpha=alpha)

plot_sphere(smoke_cloud_1_pos, smoke_radius, ax, color='grey', alpha=0.5)
plot_sphere(smoke_cloud_2_pos, smoke_radius, ax, color='grey', alpha=0.5)
ax.scatter([], [], [], color='grey', s=100, label='Smoke Cloud') # 用于图例显示

# d. 绘制来袭导弹 (箭头)
for i in range(len(missile_positions)):
    pos = missile_positions[i]
    direction = missile_directions[i]
    ax.quiver(pos[0], pos[1], pos[2],
              direction[0], direction[1], direction[2],
              length=150, normalize=True, color='#f39c12',
              arrow_length_ratio=0.3, label='Incoming Missile' if i == 0 else "") # 只为一个导弹添加图例

# --- 4. 添加图例和标题 ---
ax.legend(loc='upper left', fontsize=12)
plt.title('无人机投放烟幕干扰示意图 (3D)', fontsize=18, pad=20)


# --- 5. 保存图像 ---
plt.savefig("uav_3d_plot.png", dpi=300, bbox_inches='tight')

# 显示图像
plt.show()