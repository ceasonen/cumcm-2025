import math

def radians_to_degrees(radians):
    """将弧度转换为角度"""
    return radians * (180 / math.pi)

def degrees_to_radians(degrees):
    """将角度转换为弧度"""
    return degrees * (math.pi / 180)

# 主程序
print("弧度制与角度制转换工具")
print("1. 弧度转角度")
print("2. 角度转弧度")

choice = input("请选择转换类型 (1 或 2): ")

if choice == "1":
    radians = float(input("请输入弧度值: "))
    degrees = radians_to_degrees(radians)
    print(f"{radians} 弧度 = {degrees:.2f} 度")
elif choice == "2":
    degrees = float(input("请输入角度值: "))
    radians = degrees_to_radians(degrees)
    print(f"{degrees} 度 = {radians:.4f} 弧度")
else:
    print("无效选择，请输入 1 或 2。")