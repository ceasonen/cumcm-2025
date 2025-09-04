
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

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
        """计算无人机速度向量"""
        drone_pos = self.config.drone_config["starting_position"]
        speed = self.config.drone_config["velocity"]
        target = self.config.deceptive_target
        
        drone_xy = drone_pos[:2]
        target_xy = target[:2]
        horizontal_vec = target_xy - drone_xy
        norm = np.linalg.norm(horizontal_vec)
        
        if norm < self.config.numerical_epsilon:
            direction_xy = np.array([0.0, 0.0])
        else:
            direction_xy = horizontal_vec / norm
            
        return np.array([direction_xy[0] * speed, direction_xy[1] * speed, 0.0])

    def compute_deployment_location(self, drone_vel_vec):
        """计算烟幕弹投放位置"""
        drone_pos = self.config.drone_config["starting_position"]
        delay = self.config.drone_config["deployment_delay"]
        return drone_pos + drone_vel_vec * delay

    def compute_detonation_location(self, deployment_pos, drone_vel_vec):
        """计算烟幕弹起爆位置"""
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
        """生成用于碰撞检测的采样点"""
        samples = []
        center = self.config.authentic_target["base_center"]
        radius = self.config.authentic_target["radius"]
        height = self.config.authentic_target["height"]
        
        angles = np.linspace(0, 2 * np.pi, angular_points, endpoint=False)
        height_levels = np.linspace(center[2], center[2] + height, height_layers)

        for z in height_levels:
            for angle in angles:
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                samples.append([x, y, z])
        
        # Add center points for robustness
        samples.append(center)
        samples.append(center + np.array([0, 0, height]))

        return np.array(samples)

class ShieldEvaluator:
    """遮蔽评估器，判断目标是否被遮蔽"""
    def __init__(self, config):
        self.config = config

    def check_line_sphere_intersection(self, start_point, end_point, sphere_center, sphere_radius):
        """检查线段是否与球体相交"""
        line_vector = end_point - start_point
        center_to_start = start_point - sphere_center
        
        a = np.dot(line_vector, line_vector)
        b = 2 * np.dot(line_vector, center_to_start)
        c = np.dot(center_to_start, center_to_start) - sphere_radius**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return False
        
        sqrt_d = np.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2*a)
        t2 = (-b + sqrt_d) / (2*a)
        
        # Check if the intersection interval [t1, t2] overlaps with the segment interval [0, 1]
        return max(0, t1) <= min(1, t2)

    def evaluate_target_coverage(self, missile_position, smoke_center, target_points):
        """评估目标是否完全被遮蔽"""
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
        """执行完整仿真并返回结果用于可视化"""
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
            flight_duration = current_time
            missile_current_pos = self.config.missile_config["starting_position"] + \
                                missile_direction * self.config.missile_config["velocity"] * flight_duration
            
            descent_duration = current_time - detonation_time
            smoke_current_center = detonation_pos - np.array([0, 0, self.config.smoke_config["descent_velocity"] * descent_duration])
            
            currently_shielded = self.evaluator.evaluate_target_coverage(
                missile_current_pos, smoke_current_center, sample_points
            )

            if currently_shielded:
                total_shield_time += self.config.time_step

            if currently_shielded and not previous_shielded:
                shield_periods.append({"start": current_time})
            elif not currently_shielded and previous_shielded:
                if shield_periods:
                    shield_periods[-1]["end"] = current_time
            
            previous_shielded = currently_shielded

        if shield_periods and "end" not in shield_periods[-1]:
            shield_periods[-1]["end"] = end_window
        
        print(f"计算完成。总有效遮蔽时长: {total_shield_time:.4f} 秒")

        return {
            "total_shield_time": total_shield_time,
            "shield_periods": shield_periods,
            "deployment_pos": deployment_pos,
            "detonation_pos": detonation_pos,
            "drone_vel_vec": drone_vel_vec,
            "missile_direction": missile_direction,
            "detonation_time": detonation_time
        }

class Visualizer:
    """可视化类，负责生成3D场景图"""
    def __init__(self, config, results):
        self.config = config
        self.results = results

    def _draw_sphere(self, ax, center, radius, color):
        """在3D坐标系中绘制一个球体"""
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        ax.plot_surface(x, y, z, color=color, alpha=0.5, label="烟幕云团")

    def _draw_cylinder(self, ax, center, radius, height, color):
        """在3D坐标系中绘制一个圆柱体"""
        z = np.linspace(center[2], center[2] + height, 2)
        theta = np.linspace(0, 2 * np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid) + center[0]
        y_grid = radius * np.sin(theta_grid) + center[1]
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.7, color=color)

    def plot_scene(self):
        """绘制整个3D对抗场景"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 1. 绘制无人机轨迹
        drone_start = self.config.drone_config["starting_position"]
        deploy_pos = self.results["deployment_pos"]
        drone_path = np.array([drone_start, deploy_pos])
        ax.plot(drone_path[:, 0], drone_path[:, 1], drone_path[:, 2], 'b--', label="无人机轨迹")
        ax.scatter(*drone_start, c='blue', marker='^', s=100, label="无人机起点")
        ax.scatter(*deploy_pos, c='cyan', marker='o', s=100, label="投放点")

        # 2. 绘制干扰弹轨迹 (平抛)
        detonation_time = self.config.drone_config["detonation_delay"]
        t = np.linspace(0, detonation_time, 50)
        grenade_path_x = deploy_pos[0] + self.results["drone_vel_vec"][0] * t
        grenade_path_y = deploy_pos[1] + self.results["drone_vel_vec"][1] * t
        grenade_path_z = deploy_pos[2] - 0.5 * self.config.gravity * t**2
        ax.plot(grenade_path_x, grenade_path_y, grenade_path_z, 'g-.', label="干扰弹轨迹")
        ax.scatter(*self.results["detonation_pos"], c='green', marker='*', s=150, label="起爆点")

        # 3. 绘制导弹轨迹
        missile_start = self.config.missile_config["starting_position"]
        end_time = self.results["detonation_time"] + self.config.smoke_config["effective_duration"]
        missile_end = missile_start + self.results["missile_direction"] * self.config.missile_config["velocity"] * end_time
        missile_path = np.array([missile_start, missile_end])
        ax.plot(missile_path[:, 0], missile_path[:, 1], missile_path[:, 2], 'r-', label="导弹轨迹")
        ax.scatter(*missile_start, c='red', marker='x', s=100, label="导弹起点")

        # 4. 绘制真目标 (圆柱体)
        target_cfg = self.config.authentic_target
        self._draw_cylinder(ax, target_cfg["base_center"], target_cfg["radius"], target_cfg["height"], 'purple')
        ax.text(target_cfg["base_center"][0], target_cfg["base_center"][1], target_cfg["base_center"][2] + 20, "真目标", color='purple')

        # 5. 绘制烟幕球体和导弹在特定时刻的位置
        if self.results["shield_periods"]:
            # 选择第一个遮蔽时间段的中点作为展示时刻
            first_period = self.results["shield_periods"][0]
            snapshot_time = first_period.get("start", self.results["detonation_time"]) + \
                            (first_period.get("end", first_period.get("start")) - first_period.get("start", self.results["detonation_time"])) / 2
            
            # 计算该时刻导弹位置
            missile_snapshot_pos = missile_start + self.results["missile_direction"] * self.config.missile_config["velocity"] * snapshot_time
            ax.scatter(*missile_snapshot_pos, c='darkred', marker='s', s=80, label=f"导弹位置 @ {snapshot_time:.2f}s")
            
            # 计算该时刻烟幕位置
            descent_duration = snapshot_time - self.results["detonation_time"]
            smoke_center_snapshot = self.results["detonation_pos"] - np.array([0, 0, self.config.smoke_config["descent_velocity"] * descent_duration])
            self._draw_sphere(ax, smoke_center_snapshot, self.config.smoke_config["effective_radius"], 'gray')
            ax.text(smoke_center_snapshot[0], smoke_center_snapshot[1], smoke_center_snapshot[2]+15, f"烟幕 @ {snapshot_time:.2f}s", color='black')

        # 6. 设置图表样式
        ax.set_xlabel('X 轴 (米)')
        ax.set_ylabel('Y 轴 (米)')
        ax.set_zlabel('Z 轴 (米)')
        ax.set_title('问题一：烟幕干扰三维场景可视化', fontsize=16)
        
        # 调整坐标轴比例以获得更好的视觉效果
        x_coords = np.concatenate([drone_path[:,0], missile_path[:,0], [target_cfg["base_center"][0]]])
        y_coords = np.concatenate([drone_path[:,1], missile_path[:,1], [target_cfg["base_center"][1]]])
        z_coords = np.concatenate([drone_path[:,2], missile_path[:,2], [target_cfg["base_center"][2]]])
        
        x_range = np.ptp(x_coords)
        y_range = np.ptp(y_coords)
        z_range = np.ptp(z_coords)
        max_range = max(x_range, y_range, z_range)
        
        mid_x = (np.max(x_coords) + np.min(x_coords)) / 2
        mid_y = (np.max(y_coords) + np.min(y_coords)) / 2
        mid_z = (np.max(z_coords) + np.min(z_coords)) / 2
        
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        ax.legend(loc='upper left', bbox_to_anchor=(0.75, 0.95))
        
        # 添加文本框显示总遮蔽时间
        total_time_text = f"总有效遮蔽时长: {self.results['total_shield_time']:.4f} 秒"
        ax.text2D(0.05, 0.95, total_time_text, transform=ax.transAxes,
                  fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        ax.view_init(elev=20., azim=-120) # 设置一个较好的观察视角
        plt.tight_layout()
        plt.show()


# -------------------------- 主程序入口 --------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. 初始化配置
    sim_config = SimulationConfig()

    # 2. 创建仿真运行器并执行计算
    runner = SimulationRunner(sim_config)
    simulation_results = runner.execute_simulation()
    
    # 3. 创建可视化器并绘图
    visualizer = Visualizer(sim_config, simulation_results)
    visualizer.plot_scene()
    
    end_time = time.time()
    print(f"\n程序总运行时间: {end_time - start_time:.4f} 秒")