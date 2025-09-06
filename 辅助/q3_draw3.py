import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_q3_q4_combined():
    """
    将Q3的3D轨迹图和Q4的PSO收敛曲线图组合在一个程序中，
    左边显示3D轨迹图，右边显示收敛曲线图。
    """
    # 创建1x2的子图布局
    fig = plt.figure(figsize=(20, 8))
    
    # ========================= 左侧：Q3 三维轨迹图 =========================
    ax1 = fig.add_subplot(121, projection='3d')
    
    # --- Q3 数据定义 ---
    # 关键实体位置 (来自题目)
    fake_target = np.array([0, 0, 0])
    real_target_center = np.array([0, 200, 0])
    missile_init_pos = np.array([20000, 0, 2000])
    uav_init_pos = np.array([17800, 0, 1800])

    # 最优策略参数
    theta = 0.0933  # rad
    
    # 弹药投放与起爆点数据
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
    
    # --- Q3 绘制元素 ---
    # 实体: 我方(蓝色)、敌方(红色)、目标(绿色)、参考点(黑色)
    ax1.scatter(*uav_init_pos, s=100, c='#0077B6', marker='s', label='FY1无人机初始位置', depthshade=False)
    ax1.scatter(*missile_init_pos, s=100, c='#D62728', marker='>', label='M1导弹初始位置', depthshade=False)
    ax1.scatter(*real_target_center, s=150, c='#2CA02C', marker='o', label='真目标', depthshade=False)
    ax1.scatter(*fake_target, s=150, c='black', marker='X', label='假目标（原点）', depthshade=False)

    # 轨迹
    missile_path = np.array([missile_init_pos, fake_target])
    ax1.plot(missile_path[:,0], missile_path[:,1], missile_path[:,2], color='#D62728', linestyle='--', linewidth=2, label='导弹轨迹')
    
    uav_dir = np.array([np.cos(theta), np.sin(theta), 0])
    uav_end_pos = deploy_points[-1] + uav_dir * 500
    uav_path = np.array([uav_init_pos, uav_end_pos])
    ax1.plot(uav_path[:,0], uav_path[:,1], uav_path[:,2], color='#0077B6', linestyle='-', linewidth=2, label='无人机轨迹')

    # 弹药事件点
    colors = ['#FF7F0E', '#1F77B4', '#A9A9A9']
    labels = ['第一枚', '第二枚', '第三枚（无效）']
    for i in range(3):
        # 投放点用空心圆
        ax1.scatter(*deploy_points[i], s=80, facecolors='none', edgecolors=colors[i], linewidth=2, label=f'{labels[i]}投放')
        # 起爆点用星形
        ax1.scatter(*detonation_points[i], s=200, c=colors[i], marker='*', label=f'{labels[i]}起爆')
        # 连接投放与起爆的示意线
        ax1.plot([deploy_points[i,0], detonation_points[i,0]],
                  [deploy_points[i,1], detonation_points[i,1]],
                  [deploy_points[i,2], detonation_points[i,2]], 
                  color=colors[i], linestyle=':', linewidth=2)

    # --- Q3 设置样式 ---
    ax1.set_title('三枚烟幕弹最优投放策略', fontsize=16, pad=20)
    ax1.set_xlabel('X (米)', fontsize=11, labelpad=10)
    ax1.set_ylabel('Y (米)', fontsize=11, labelpad=10)
    ax1.set_zlabel('Z (米)', fontsize=11, labelpad=10)
    
    # 视角
    ax1.view_init(elev=25, azim=-125)
    
    # 背景和网格
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('w')
    ax1.yaxis.pane.set_edgecolor('w')
    ax1.zaxis.pane.set_edgecolor('w')
    ax1.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    
    # 图例
    ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)
    
    # ========================= 右侧：Q4 PSO收敛曲线图 =========================
    ax2 = fig.add_subplot(122)
    
    # --- Q4 PSO收敛数据 ---
    gbest_history = []
    
    # 构建收敛历史数据
    gbest_history.extend([0.0] * 4)     # 0-3次迭代: 0
    gbest_history.extend([2.5] * 3)     # 4-6次迭代: 2.5
    gbest_history.extend([3.89] * 3)    # 7-9次迭代: 3.89
    gbest_history.extend([3.9] * 3)     # 10-12次迭代: 3.9
    gbest_history.extend([5.45] * 3)    # 13-15次迭代: 5.45
    gbest_history.extend([5.5] * 3)     # 16-18次迭代: 5.5
    gbest_history.extend([6.41] * (121 - len(gbest_history)))  # 19次迭代及以后: 6.41

    # --- Q4 绘制收敛曲线 ---
    iterations = np.arange(len(gbest_history))
    ax2.plot(iterations, gbest_history, 'b-', linewidth=2, label="全局最优适应度")

    # --- Q4 设置样式 ---
    ax2.set_title("PSO收敛曲线", fontsize=16)
    ax2.set_xlabel("迭代次数", fontsize=12)
    ax2.set_ylabel("适应度值 (秒)", fontsize=12)

    # 坐标轴范围
    ax2.set_xlim(left=-5, right=125)
    ax2.set_ylim(bottom=-0.5, top=6.8)

    # 网格
    ax2.grid(True, alpha=0.3)

    # 图例
    ax2.legend(fontsize=11)
    
    # ========================= 整体布局调整 =========================
    plt.tight_layout()
    
    # ========================= 保存和显示 =========================
    output_path = "q3.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Q3和Q4组合图已保存至: {output_path}")
    plt.show()

if __name__ == '__main__':
    plot_q3_q4_combined()
