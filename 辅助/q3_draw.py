import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
def plot_final_strategy_3d_elegant():
    """
    根据给定的最优策略数据，生成一张纯粹的、优雅的、
    适合科研刊物登载的三维轨迹图。
    """
    # --- 1. 数据定义 (完全基于您提供的数据) ---
    
    # 关键实体位置 (来自题目)
    fake_target = np.array([0, 0, 0])
    real_target_center = np.array([0, 200, 0])
    missile_init_pos = np.array([20000, 0, 2000])
    uav_init_pos = np.array([17800, 0, 1800])

    # 最优策略参数 (来自您的终端输出)
    theta = 0.0933  # rad
    
    # 弹药投放与起爆点数据 (来自您的Excel/终端数据)
    deploy_points = np.array([
        [17800, 0, 1800],
        [17939.39, 13.04, 1800],
        [18078.78, 26.07, 1800]
    ])
    detonation_points = np.array([
        [17800, 0, 1800],
        [17939.39, 13.04, 1800],
        [18078.78, 26.07, 1800]
    ])
    
    # --- 2. 创建图表 ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- 3. 绘制元素 (采用经典科学配色) ---
    
    # 实体: 我方(蓝色)、敌方(红色)、目标(绿色)、参考点(黑色)
    ax.scatter(*uav_init_pos, s=100, c='#0077B6', marker='s', label='FY1无人机初始位置', depthshade=False)
    ax.scatter(*missile_init_pos, s=100, c='#D62728', marker='>', label='M1导弹初始位置', depthshade=False)
    ax.scatter(*real_target_center, s=150, c='#2CA02C', marker='o', label='真目标', depthshade=False)
    ax.scatter(*fake_target, s=150, c='black', marker='X', label='假目标（原点）', depthshade=False)

    # 轨迹
    missile_path = np.array([missile_init_pos, fake_target])
    ax.plot(missile_path[:,0], missile_path[:,1], missile_path[:,2], color='#D62728', linestyle='--', linewidth=2, label='导弹轨迹')
    
    uav_dir = np.array([np.cos(theta), np.sin(theta), 0])
    # 计算一个合理的轨迹结束点用于显示
    uav_end_pos = deploy_points[-1] + uav_dir * 500
    uav_path = np.array([uav_init_pos, uav_end_pos])
    ax.plot(uav_path[:,0], uav_path[:,1], uav_path[:,2], color='#0077B6', linestyle='-', linewidth=2, label='无人机轨迹')

    # 弹药事件点
    colors = ['#FF7F0E', '#1F77B4', '#A9A9A9'] # 橙色, 蓝色, 灰色
    labels = ['第一枚', '第二枚', '第三枚（无效）']
    for i in range(3):
        # 投放点用空心圆
        ax.scatter(*deploy_points[i], s=80, facecolors='none', edgecolors=colors[i], linewidth=2, label=f'{labels[i]}投放')
        # 起爆点用星形
        ax.scatter(*detonation_points[i], s=200, c=colors[i], marker='*', label=f'{labels[i]}起爆')
        # 连接投放与起爆的示意线
        ax.plot([deploy_points[i,0], detonation_points[i,0]],
                  [deploy_points[i,1], detonation_points[i,1]],
                  [deploy_points[i,2], detonation_points[i,2]], 
                  color=colors[i], linestyle=':', linewidth=2)

    # --- 4. 设置样式 (专业、优雅) ---
    
    # 标题和轴标签
    ax.set_title('三枚烟幕弹最优投放策略', fontsize=18, pad=20)
    ax.set_xlabel('X (米)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (米)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (米)', fontsize=12, labelpad=10)
    
    # 视角
    ax.view_init(elev=25, azim=-125)
    
    # 背景和网格
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    
    # 图例
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=11)
    
    # 调整布局以防止图例被截断
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # --- 5. 保存和显示 ---
    output_path = "strategy_3d_visualization_final.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"优雅的三维策略图已保存至: {output_path}")
    plt.show()

if __name__ == '__main__':
    plot_final_strategy_3d_elegant()