import numpy as np
import time

class SimulationConfig:
    """仿真配置类，集中管理所有参数"""
    def __init__(self):
        self.gravity = 9.8  # 重力加速度 (m/s²)
        self.time_step = 0.01  # 时间步长
        self.numerical_epsilon = 1e-12  # 数值计算保护阈值

        # 目标配置
        self.deceptive_target = np.array([0.0, 0.0, 0.0])  # 假目标位置
        self.authentic_target = {
            "base_center": np.array([0.0, 200.0, 0.0]),  # 真目标底面圆心
            "radius": 7.0,  # 圆柱半径
            "height": 10.0   # 圆柱高度
        }

        # 无人机配置
        self.drone_config = {
            "starting_position": np.array([17800.0, 0.0, 1800.0]),  # 初始位置
            "velocity": 120.0,  # 飞行速度(m/s)
            "deployment_delay": 1.5,  # 任务到投放延迟(s)
            "detonation_delay": 3.6    # 投放到起爆延迟(s)
        }

        # 烟幕配置
        self.smoke_config = {
            "effective_radius": 10.0,  # 有效半径(m)
            "descent_velocity": 3.0,  # 下沉速度(m/s)
            "effective_duration": 20.0  # 有效遮蔽时间(s)
        }

        # 导弹配置
        self.missile_config = {
            "starting_position": np.array([20000.0, 0.0, 2000.0]),  # 初始位置
            "velocity": 300.0  # 飞行速度(m/s)
        }

class TrajectoryCalculator:
    """轨迹计算器，负责计算投放和起爆位置"""
    def __init__(self, config):
        self.config = config

    def compute_deployment_location(self):
        """计算烟幕弹投放位置"""
        drone_pos = self.config.drone_config["starting_position"]
        speed = self.config.drone_config["velocity"]
        delay = self.config.drone_config["deployment_delay"]
        target = self.config.deceptive_target

        # 计算水平方向向量
        drone_xy = drone_pos[:2]
        target_xy = target[:2]
        horizontal_distance = np.linalg.norm(target_xy - drone_xy)

        if horizontal_distance < self.config.numerical_epsilon:
            direction_xy = np.array([0.0, 0.0])
        else:
            direction_xy = (target_xy - drone_xy) / horizontal_distance

        # 计算投放位置
        flight_distance = speed * delay
        deployment_xy = drone_xy + direction_xy * flight_distance
        deployment_z = drone_pos[2]  # 保持高度不变

        return np.array([deployment_xy[0], deployment_xy[1], deployment_z])

    def compute_detonation_location(self, deployment_pos):
        """计算烟幕弹起爆位置"""
        speed = self.config.drone_config["velocity"]
        delay = self.config.drone_config["detonation_delay"]
        target = self.config.deceptive_target

        # 水平运动计算
        deployment_xy = deployment_pos[:2]
        target_xy = target[:2]
        horizontal_distance = np.linalg.norm(target_xy - deployment_xy)

        if horizontal_distance < self.config.numerical_epsilon:
            direction_xy = np.array([0.0, 0.0])
        else:
            direction_xy = (target_xy - deployment_xy) / horizontal_distance

        horizontal_displacement = speed * delay
        detonation_xy = deployment_xy + direction_xy * horizontal_displacement

        # 垂直自由落体运动
        vertical_drop = 0.5 * self.config.gravity * delay ** 2
        detonation_z = deployment_pos[2] - vertical_drop

        return np.array([detonation_xy[0], detonation_xy[1], detonation_z])

class TargetSampler:
    """目标采样器，生成高密度采样点"""
    def __init__(self, config):
        self.config = config

    def create_dense_samples(self, angular_points=60, height_layers=20):
        """生成超高密度采样点"""
        samples = []
        center = self.config.authentic_target["base_center"]
        radius = self.config.authentic_target["radius"]
        height = self.config.authentic_target["height"]
        center_xy = center[:2]
        min_height = center[2]
        max_height = center[2] + height

        # 生成角度序列
        angles = np.linspace(0, 2*np.pi, angular_points, endpoint=False)

        # 外表面采样
        # 底面圆周
        for angle in angles:
            x = center_xy[0] + radius * np.cos(angle)
            y = center_xy[1] + radius * np.sin(angle)
            samples.append([x, y, min_height])

        # 顶面圆周
        for angle in angles:
            x = center_xy[0] + radius * np.cos(angle)
            y = center_xy[1] + radius * np.sin(angle)
            samples.append([x, y, max_height])

        # 侧面采样
        height_levels = np.linspace(min_height, max_height, height_layers, endpoint=True)
        for z in height_levels:
            for angle in angles:
                x = center_xy[0] + radius * np.cos(angle)
                y = center_xy[1] + radius * np.sin(angle)
                samples.append([x, y, z])

        # 内部网格点采样
        radial_steps = np.linspace(0, radius, 5, endpoint=True)
        internal_heights = np.linspace(min_height, max_height, 10, endpoint=True)
        internal_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)

        for z in internal_heights:
            for r in radial_steps:
                for angle in internal_angles:
                    x = center_xy[0] + r * np.cos(angle)
                    y = center_xy[1] + r * np.sin(angle)
                    samples.append([x, y, z])

        # 轴线关键点
        axis_points = [
            [center_xy[0], center_xy[1], min_height],
            [center_xy[0], center_xy[1], min_height + height/4],
            [center_xy[0], center_xy[1], min_height + height/2],
            [center_xy[0], center_xy[1], min_height + 3*height/4],
            [center_xy[0], center_xy[1], max_height]
        ]
        samples.extend(axis_points)

        return np.unique(np.array(samples), axis=0)

class ShieldEvaluator:
    """遮蔽评估器，判断目标是否被遮蔽"""
    def __init__(self, config):
        self.config = config

    def check_line_sphere_intersection(self, start_point, end_point, sphere_center, sphere_radius):
        """检查线段是否与球体相交"""
        line_vector = end_point - start_point
        center_to_start = sphere_center - start_point

        a = np.dot(line_vector, line_vector)

        if a < self.config.numerical_epsilon:
            return np.linalg.norm(center_to_start) <= sphere_radius + self.config.numerical_epsilon

        b = -2 * np.dot(line_vector, center_to_start)
        c = np.dot(center_to_start, center_to_start) - sphere_radius ** 2

        discriminant = b ** 2 - 4 * a * c
        if discriminant < -self.config.numerical_epsilon:
            return False

        if discriminant < 0:
            discriminant = 0

        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)

        return (t1 <= 1.0 + self.config.numerical_epsilon) and (t2 >= -self.config.numerical_epsilon)

    def evaluate_target_coverage(self, missile_position, smoke_center, smoke_radius, target_points):
        """评估目标是否完全被遮蔽"""
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
        """执行完整仿真"""
        start_time = time.time()

        # 计算关键位置
        deployment_pos = self.calculator.compute_deployment_location()
        detonation_pos = self.calculator.compute_detonation_location(deployment_pos)

        print("=== 基础位置信息 ===")
        print(f"无人机初始位置：{self.config.drone_config['starting_position'].round(4)}")
        print(f"烟幕弹投放点：{deployment_pos.round(4)}")
        print(f"烟幕弹起爆点：{detonation_pos.round(4)}")
        print(f"假目标位置：{self.config.deceptive_target}")

        # 生成采样点
        sample_points = self.sampler.create_dense_samples()
        print(f"\n=== 采样点信息 ===")
        print(f"真目标采样点总数：{len(sample_points)}（含外表面和内部点）")

        # 计算导弹飞行方向
        missile_vector = self.config.deceptive_target - self.config.missile_config["starting_position"]
        missile_distance = np.linalg.norm(missile_vector)
        if missile_distance < self.config.numerical_epsilon:
            missile_direction = np.array([0.0, 0.0, 0.0])
        else:
            missile_direction = missile_vector / missile_distance
        print(f"\n=== 导弹信息 ===")
        print(f"导弹初始位置：{self.config.missile_config['starting_position'].round(4)}")
        print(f"导弹飞行方向向量：{missile_direction.round(6)}")

        # 定义时间范围
        detonation_time = self.config.drone_config["deployment_delay"] + self.config.drone_config["detonation_delay"]
        start_window = detonation_time
        end_window = detonation_time + self.config.smoke_config["effective_duration"]
        time_points = np.arange(start_window, end_window + self.config.time_step, self.config.time_step)
        print(f"\n=== 时间范围 ===")
        print(f"起爆时刻：{detonation_time:.2f}s")
        print(f"有效时间窗口：[{start_window:.2f}s, {end_window:.2f}s]，共{len(time_points)}个时间步")

        # 执行仿真计算
        total_shield_time = 0.0
        shield_records = []
        previous_shielded = False
        shield_periods = []

        for current_time in time_points:
            # 计算导弹当前位置
            flight_duration = current_time
            missile_current_pos = self.config.missile_config["starting_position"] + \
                                missile_direction * self.config.missile_config["velocity"] * flight_duration

            # 计算烟幕当前位置
            descent_duration = current_time - detonation_time
            smoke_current_center = np.array([
                detonation_pos[0],
                detonation_pos[1],
                detonation_pos[2] - self.config.smoke_config["descent_velocity"] * descent_duration
            ])

            # 判断遮蔽状态
            currently_shielded = self.evaluator.evaluate_target_coverage(
                missile_current_pos, smoke_current_center,
                self.config.smoke_config["effective_radius"], sample_points
            )

            # 累积有效时间
            if currently_shielded:
                total_shield_time += self.config.time_step
                shield_records.append({
                    "time": round(current_time, 3),
                    "missile_position": missile_current_pos.round(4),
                    "smoke_center": smoke_current_center.round(4)
                })

            # 记录遮蔽时间段
            if currently_shielded and not previous_shielded:
                shield_periods.append({"start": current_time})
            elif not currently_shielded and previous_shielded:
                if shield_periods:
                    shield_periods[-1]["end"] = current_time - self.config.time_step

            previous_shielded = currently_shielded

        # 处理最后一个未结束的遮蔽段
        if shield_periods and "end" not in shield_periods[-1]:
            shield_periods[-1]["end"] = end_window

        # 输出结果
        end_time = time.time()
        print("\n" + "="*80)
        print(f"【最终结果】真目标被有效遮蔽的总时长：{total_shield_time:.4f} 秒")
        print("="*80)
        print(f"仿真耗时：{end_time - start_time:.4f} 秒")

        # 输出遮蔽时间段详情
        print("\n=== 遮蔽时间段详情 ===")
        if not shield_periods:
            print("无有效遮蔽时间段")
        else:
            for idx, period in enumerate(shield_periods, 1):
                duration = period["end"] - period["start"]
                print(f"第{idx}段：{period['start']:.4f}s ~ {period['end']:.4f}s，时长：{duration:.4f}s")

        # 输出采样时刻状态
        print("\n=== 采样时刻状态示例 ===")
        if shield_records:
            print("前3个有效时刻：")
            for record in shield_records[:3]:
                print(f"t={record['time']}s | 导弹位置：{record['missile_position']} | 烟幕中心：{record['smoke_center']}")

            print("\n最后3个有效时刻：")
            for record in shield_records[-3:]:
                print(f"t={record['time']}s | 导弹位置：{record['missile_position']} | 烟幕中心：{record['smoke_center']}")
        else:
            print("无有效遮蔽时刻")

# -------------------------- 主程序入口 --------------------------
if __name__ == "__main__":
    # 初始化配置
    sim_config = SimulationConfig()

    # 创建仿真运行器并执行
    runner = SimulationRunner(sim_config)
    runner.execute_simulation()
