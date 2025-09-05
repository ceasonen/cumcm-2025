# 安装所需库:
# pip install numpy matplotlib

import matplotlib.pyplot as plt
import numpy as np
# 必须导入 Axes3D 才能激活 3D 绘图功能
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 设置 Matplotlib 以正确显示中文和负号 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei (黑体)
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# --- 2. 定义示意图的坐标数据 (为了清晰，坐标经过调整) ---

# 导弹 M1 的轨迹 (红色，位置较低)
missile_start = np.array([18000, 0, 1200])
missile_end = np.array([17500, 0, 1800])

# 无人机 FY1 的轨迹 (蓝色，位置较高)
uav_start = np.array([18200, -500, 2600])
# 无人机轨迹的终点就是投放点
deployment_point = np.array([17800, -200, 2300])

# 关键事件点
# 起爆点：在投放点之后，位置略有偏移 (模拟平抛)
detonation_point = np.array([17750, -150, 2200])
# 云团初始中心：与起爆点非常接近
cloud_initial_center = np.array([17740, -140, 2180])

# --- 3. 开始绘图 ---

# 创建一个图形窗口和 3D 坐标轴
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 绘制导弹轨迹
ax.plot([missile_start[0], missile_end[0]], 
        [missile_start[1], missile_end[1]], 
        [missile_start[2], missile_end[2]], 
        color='red', linewidth=2.5, label='导弹M1轨迹')

# 绘制无人机轨迹
ax.plot([uav_start[0], deployment_point[0]], 
        [uav_start[1], deployment_point[1]], 
        [uav_start[2], deployment_point[2]], 
        color='blue', linewidth=2.5, label='无人机FY1轨迹')

# 绘制关键事件点 (zorder=5 确保点在轨迹线上方)
ax.scatter(deployment_point[0], deployment_point[1], deployment_point[2], 
           color='green', s=100, label='投放点', zorder=5, ec='black')
ax.scatter(detonation_point[0], detonation_point[1], detonation_point[2], 
           color='orange', s=100, label='起爆点', zorder=5, ec='black')
ax.scatter(cloud_initial_center[0], cloud_initial_center[1], cloud_initial_center[2], 
           color='purple', s=100, label='云团初始中心', zorder=5, ec='black')

# --- 4. 设置图表样式 ---

# 设置标题
ax.set_title('3D轨迹可视化', fontsize=18, pad=20)

# 设置坐标轴标签
ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)

# 设置坐标轴范围，让元素居中且有呼吸空间
ax.set_xlim(17000, 19000)
ax.set_ylim(-4000, 4000)
ax.set_zlim(0, 3000)

# 设置视角 (elev=仰角, azim=方位角)，这个角度与您的示例图非常接近
ax.view_init(elev=25, azim=-75)

# 移除背景填充色，使其更干净
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# 显示图例
ax.legend(fontsize=12, loc='upper right')

# 调整布局并显示图形
plt.tight_layout()
plt.savefig('p1_2.png', dpi=300)
plt.show()
