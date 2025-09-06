import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
def plot_exact_convergence_curve():
    """
    严格按照用户提供的图片和数据，复现PSO收敛曲线图。
    不包含任何额外的字体设置或复杂样式。
    """
    # --- 1. 严格根据您的日志，构建真实的历史数据 ---
    # 并根据您图片中的阶梯形状，手动插入中间值以复现曲线
    gbest_history = []
    
    # 0-3次迭代: 0
    gbest_history.extend([0.0] * 4)
    # 4-6次迭代: 2.5 (模拟第一次跃升)
    gbest_history.extend([2.5] * 3)
    # 7-9次迭代: 3.89 (模拟第二次跃升)
    gbest_history.extend([3.89] * 3)
    # 10-12次迭代: 3.9 (微小提升)
    gbest_history.extend([3.9] * 3)
    # 13-15次迭代: 5.45 (模拟第三次跃升)
    gbest_history.extend([5.45] * 3)
    # 16-18次迭代: 5.5 (微小提升)
    gbest_history.extend([5.5] * 3)
    # 19次迭代及以后: 6.41 (最终值)
    gbest_history.extend([6.41] * (121 - len(gbest_history)))

    # --- 2. 绘制图表 (严格复现风格) ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制曲线
    iterations = np.arange(len(gbest_history))
    ax.plot(iterations, gbest_history, label="全局最优适应度") # 使用中文标签

    # --- 3. 设置样式 (严格复现) ---
    # 标题和轴标签 (使用中文)
    ax.set_title("PSO收敛曲线")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("适应度值 (秒)")

    # 坐标轴范围
    ax.set_xlim(left=-5, right=125) # 左右留出一些空白
    ax.set_ylim(bottom=-0.5, top=6.8) # 上下留出一些空白

    # 网格
    ax.grid(True)

    # 图例
    ax.legend()

    # --- 4. 保存和显示 ---
    output_path = "pso_convergence_curve_reproduced.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"已严格复现的收敛曲线图已保存至: {output_path}")
    plt.show()

if __name__ == '__main__':
    plot_exact_convergence_curve()