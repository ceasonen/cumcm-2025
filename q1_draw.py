

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

# --- 计算模块 (与之前版本相同，保持不变) ---
class SimulationConfig:
    """仿真配置类，集中管理所有参数"""
    def __init__(self):
        self.gravity = 9.8
        self.time_step = 0.01
        self.numerical_epsilon = 1e-12

        self.deceptive_target = np.array([0.0, 0.0, 0.0])
        self.authentic_target = {
            "base_center": np.array([0.0, 200.0, 0.0]),
            "radius": 7.0,
            "height": 10.0
        }

        self.drone_config = {
            "starting_position": np.array([17800.0, 0.0, 1800.0]),
            "velocity": 120.0,
            "deployment_delay": 1.5,
            "detonation_delay": 3.6
        }

        self.smoke_config = {
            "effective_radius": 10.0,
            "descent_velocity": 3.0,
            "effective_duration": 20.0
        }

        self.missile_config = {
            "starting_position": np.array([20000.0, 0.0, 2000.0]),
            "velocity": 300.0
        }

class TrajectoryCalculator:
    """轨迹计算器，负责计算投放和起爆位置"""
    def __init__(self, config):
        self.config = config

    def compute_drone_velocity_vector(self):
        drone_pos = self.config.drone_config["starting_position"]
        speed = self.config.drone_config["velocity"]
        target = self.config.deceptive_target
        horizontal_vec = target[:2] - drone_pos[:2]
        norm = np.linalg.norm(horizontal_vec)
        direction_xy = horizontal_vec / norm if norm > self.config.numerical_epsilon else np.array([0.0, 0.0])
        return np.array([direction_xy[0] * speed, direction_xy[1] * speed, 0.0])

    def compute_deployment_location(self, drone_vel_vec):
        drone_pos = self.config.drone_config["starting_position"]
        delay = self.config.drone_config["deployment_delay"]
        return drone_pos + drone_vel_vec * delay

    def compute_detonation_location(self, deployment_pos, drone_vel_vec):
        delay = self.config.drone_config["detonation_delay"]
        horizontal_displacement = drone_vel_vec * delay
        vertical_drop = 0.5 * self.config.gravity * delay ** 2
        detonation_pos = deployment_pos + horizontal_displacement
        detonation_pos[2] -= vertical_drop
        return detonation_pos

class TargetSampler:
    """目标采样器，生成高密度采样点"""
    def __init__(self, config):
        self.config = config

    def create_dense_samples(self, angular_points=36, height_layers=10):
        samples = []
        center = self.config.authentic_target["base_center"]
        radius = self.config.authentic_target["radius"]
        height = self.config.authentic_target["height"]
        angles = np.linspace(0, 2 * np.pi, angular_points, endpoint=False)
        height_levels = np.linspace(center[2], center[2] + height, height_layers)
        for z in height_levels:
            for angle in angles:
                samples.append([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), z])
        samples.append(center)
        samples.append(center + np.array([0, 0, height]))
        return np.array(samples)

class ShieldEvaluator:
    """遮蔽评估器，判断目标是否被遮蔽"""
    def __init__(self, config):
        self.config = config

    def check_line_sphere_intersection(self, start_point, end_point, sphere_center, sphere_radius):
        line_vector = end_point - start_point
        center_to_start = start_point - sphere_center
        a = np.dot(line_vector, line_vector)
        b = 2 * np.dot(line_vector, center_to_start)
        c = np.dot(center_to_start, center_to_start) - sphere_radius**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0: return False
        sqrt_d = np.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2*a)
        t2 = (-b + sqrt_d) / (2*a)
        return max(0, t1) <= min(1, t2)

    def evaluate_target_coverage(self, missile_position, smoke_center, target_points):
        smoke_radius = self.config.smoke_config["effective_radius"]
        for point in target_points:
            if not self.check_line_sphere_intersection(missile_position, point, smoke_center, smoke_radius):
                return False
        return True

class SimulationRunner:
    """仿真运行器，主控制流程"""
    def __init__(self, config):
        self.config = config
        self.calculator = TrajectoryCalculator(config)
        self.sampler = TargetSampler(config)
        self.evaluator = ShieldEvaluator(config)

    def execute_simulation(self):
        print("开始执行仿真计算...")
        drone_vel_vec = self.calculator.compute_drone_velocity_vector()
        deployment_pos = self.calculator.compute_deployment_location(drone_vel_vec)
        detonation_pos = self.calculator.compute_detonation_location(deployment_pos, drone_vel_vec)
        sample_points = self.sampler.create_dense_samples()
        missile_vector = self.config.deceptive_target - self.config.missile_config["starting_position"]
        missile_direction = missile_vector / np.linalg.norm(missile_vector)
        detonation_time = self.config.drone_config["deployment_delay"] + self.config.drone_config["detonation_delay"]
        start_window = detonation_time
        end_window = detonation_time + self.config.smoke_config["effective_duration"]
        time_points = np.arange(start_window, end_window, self.config.time_step)
        total_shield_time = 0.0
        previous_shielded = False
        shield_periods = []
        for current_time in time_points:
            missile_current_pos = self.config.missile_config["starting_position"] + missile_direction * self.config.missile_config["velocity"] * current_time
            smoke_current_center = detonation_pos - np.array([0, 0, self.config.smoke_config["descent_velocity"] * (current_time - detonation_time)])
            currently_shielded = self.evaluator.evaluate_target_coverage(missile_current_pos, smoke_current_center, sample_points)
            if currently_shielded: total_shield_time += self.config.time_step
            if currently_shielded and not previous_shielded: shield_periods.append({"start": current_time})
            elif not currently_shielded and previous_shielded and shield_periods: shield_periods[-1]["end"] = current_time
            previous_shielded = currently_shielded
        if shield_periods and "end" not in shield_periods[-1]: shield_periods[-1]["end"] = end_window
        print(f"计算完成。总有效遮蔽时长: {total_shield_time:.4f} 秒")
        return {"total_shield_time": total_shield_time, "shield_periods": shield_periods, "deployment_pos": deployment_pos, "detonation_pos": detonation_pos, "drone_vel_vec": drone_vel_vec, "missile_direction": missile_direction, "detonation_time": detonation_time}

# --- 优化后的可视化模块 ---
class Visualizer:
    """可视化类，采用双视图展示宏观与微观场景"""
    def __init__(self, config, results):
        self.config = config
        self.results = results

    def _draw_sphere(self, ax, center, radius, color, label):
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        ax.plot_surface(x, y, z, color=color, alpha=0.6, label=label)

    def _draw_cylinder(self, ax, center, radius, height, color, label):
        z = np.linspace(center[2], center[2] + height, 50)
        theta = np.linspace(0, 2 * np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid) + center[0]
        y_grid = radius * np.sin(theta_grid) + center[1]
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.7, color=color, label=label)

    def _plot_global_view(self, ax, zoom_bounds):
        """绘制宏观全局视图"""
        ax.set_title("全局态势视图", fontsize=14)
        
        # 绘制轨迹
        drone_start = self.config.drone_config["starting_position"]
        deploy_pos = self.results["deployment_pos"]
        drone_path = np.array([drone_start, deploy_pos])
        ax.plot(drone_path[:, 0], drone_path[:, 1], drone_path[:, 2], 'b--', label="无人机轨迹")
        ax.scatter(*drone_start, c='blue', marker='^', s=80, label="无人机起点")

        missile_start = self.config.missile_config["starting_position"]
        end_time = self.results["detonation_time"] + self.config.smoke_config["effective_duration"]
        missile_end = missile_start + self.results["missile_direction"] * self.config.missile_config["velocity"] * end_time
        missile_path = np.array([missile_start, missile_end])
        ax.plot(missile_path[:, 0], missile_path[:, 1], missile_path[:, 2], 'r-', label="导弹轨迹")
        ax.scatter(*missile_start, c='red', marker='x', s=80, label="导弹起点")

        # 绘制真目标位置
        target_pos = self.config.authentic_target["base_center"]
        ax.scatter(*target_pos, c='purple', marker='s', s=80, label="真目标位置")
        
        # 绘制放大区域框
        min_b, max_b = zoom_bounds
        verts = [[(min_b[0], min_b[1], min_b[2]), (max_b[0], min_b[1], min_b[2]), (max_b[0], max_b[1], min_b[2]), (min_b[0], max_b[1], min_b[2])],
                 [(min_b[0], min_b[1], max_b[2]), (max_b[0], min_b[1], max_b[2]), (max_b[0], max_b[1], max_b[2]), (min_b[0], max_b[1], max_b[2])]]
        ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='b', alpha=.10))
        ax.text(max_b[0], max_b[1], max_b[2], " 交互区域", color='blue')

        ax.set_xlabel('X (米)'); ax.set_ylabel('Y (米)'); ax.set_zlabel('Z (米)')
        ax.legend(loc='best')

    def _plot_zoom_view(self, ax):
        """绘制交互区域的放大视图"""
        ax.set_title("核心交互区域放大视图", fontsize=14)
        
        # 确定展示时刻和位置
        snapshot_time = 0
        if self.results["shield_periods"]:
            first_period = self.results["shield_periods"][0]
            snapshot_time = first_period.get("start", self.results["detonation_time"])
        else: # 如果没有遮蔽，则展示起爆瞬间
            snapshot_time = self.results["detonation_time"]

        missile_start = self.config.missile_config["starting_position"]
        missile_snapshot_pos = missile_start + self.results["missile_direction"] * self.config.missile_config["velocity"] * snapshot_time
        descent_duration = snapshot_time - self.results["detonation_time"]
        smoke_center_snapshot = self.results["detonation_pos"] - np.array([0, 0, self.config.smoke_config["descent_velocity"] * descent_duration])
        
        # 绘制关键对象
        self._draw_sphere(ax, smoke_center_snapshot, self.config.smoke_config["effective_radius"], 'gray', '烟幕云团')
        target_cfg = self.config.authentic_target
        self._draw_cylinder(ax, target_cfg["base_center"], target_cfg["radius"], target_cfg["height"], 'purple', '真目标')
        
        ax.scatter(*missile_snapshot_pos, c='darkred', marker='s', s=100, label=f"导弹 @ {snapshot_time:.2f}s")
        
        # 绘制被遮挡的视线
        target_top = target_cfg["base_center"] + np.array([0, 0, target_cfg["height"]])
        los_path_center = np.array([missile_snapshot_pos, target_cfg["base_center"]])
        los_path_top = np.array([missile_snapshot_pos, target_top])
        ax.plot(los_path_center[:, 0], los_path_center[:, 1], los_path_center[:, 2], 'y--', label="被遮挡的视线 (LOS)")
        ax.plot(los_path_top[:, 0], los_path_top[:, 1], los_path_top[:, 2], 'y--')

        # 添加箭头标注
        ax.annotate("烟幕云团", xy=(smoke_center_snapshot[0], smoke_center_snapshot[1]), 
                    xytext=(smoke_center_snapshot[0] + 20, smoke_center_snapshot[1] + 20),
                    arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center')
        ax.annotate("真目标", xy=(target_cfg["base_center"][0], target_cfg["base_center"][1]), 
                    xytext=(target_cfg["base_center"][0] - 20, target_cfg["base_center"][1] - 20),
                    arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center')

        # 设置坐标轴范围和比例
        bounds_center = (smoke_center_snapshot + target_cfg["base_center"]) / 2
        max_dim = 50 # 设置一个固定的放大窗口大小
        min_b = bounds_center - max_dim
        max_b = bounds_center + max_dim
        ax.set_xlim(min_b[0], max_b[0]); ax.set_ylim(min_b[1], max_b[1]); ax.set_zlim(min_b[2], max_b[2])
        ax.set_xlabel('X (米)'); ax.set_ylabel('Y (米)'); ax.set_zlabel('Z (米)')
        ax.set_box_aspect([1,1,1]) # 保持长宽高比例一致
        ax.legend(loc='best')
        return min_b, max_b

    def plot_scene(self):
        """主绘图函数，创建并展示双视图"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(22, 10))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建左右两个3D子图
        ax_global = fig.add_subplot(1, 2, 1, projection='3d')
        ax_zoom = fig.add_subplot(1, 2, 2, projection='3d')

        # 绘制放大视图并获取其边界
        zoom_bounds = self._plot_zoom_view(ax_zoom)
        
        # 绘制全局视图，并传入放大视图的边界以绘制提示框
        self._plot_global_view(ax_global, zoom_bounds)

        # 设置总标题和文本信息
        total_time_text = f"总有效遮蔽时长: {self.results['total_shield_time']:.4f} 秒"
        fig.suptitle('问题一：烟幕干扰三维场景可视化 (宏观与微观双视图)', fontsize=20)
        fig.text(0.5, 0.92, total_time_text, ha='center', fontsize=14,
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0, 1, 0.9]) # 调整布局防止标题重叠
        plt.show()

# -------------------------- 主程序入口 --------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    sim_config = SimulationConfig()
    runner = SimulationRunner(sim_config)
    simulation_results = runner.execute_simulation()
    
    visualizer = Visualizer(sim_config, simulation_results)
    visualizer.plot_scene()
    
    end_time = time.time()
    print(f"\n程序总运行时间: {end_time - start_time:.4f} 秒")