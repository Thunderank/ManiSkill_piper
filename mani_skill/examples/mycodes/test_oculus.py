import sys
import os
import time
import numpy as np

'''
前:z轴正轴
右:x轴正轴
上:y轴正轴
'''


# 设置oculus_reader模块路径
sys.path[0] = os.path.abspath("/home/robot/Desktop/3DGS/oculus_reader")
from oculus_reader.reader import OculusReader

def rotation_matrix_to_quaternion(R):
    """
    将3x3旋转矩阵转为四元数 [w, x, y, z]
    """
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])

def main():
    # 初始化OculusReader
    oculus_reader = OculusReader(print_positions=False)
    print("OculusReader initialized.")

    try:
        while True:
            # 获取VR控制器的变换和按键
            transforms, buttons = oculus_reader.get_transformations_and_buttons()
            print(f"Transforms: {transforms}")
            print(f"Buttons: {buttons}")

            # 拆分右手柄的x, y, z和旋转矩阵
            if 'r' in transforms:
                mat = transforms['r']
                pos = mat[:3, 3]
                rot = mat[:3, :3]
                quat = rotation_matrix_to_quaternion(rot)
                print(f"Right controller position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
                print(f"Right controller quaternion: [w, x, y, z]={quat}")
            else:
                print("Right controller not detected.")

            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        oculus_reader.stop()

if __name__ == "__main__":
    main()
