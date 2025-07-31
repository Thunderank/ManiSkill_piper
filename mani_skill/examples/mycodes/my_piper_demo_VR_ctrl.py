"""
使用mplib添加ik逆解,运动规划,碰撞避免
"""

import gymnasium as gym
import numpy as np
import sapien
import pygame
import time
import math

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode

# Add import for OculusReader
import sys
import os
sys.path.append(os.path.abspath("/home/robot/Desktop/3DGS/oculus_reader"))   #引入oculus仓库目录
from oculus_reader.reader import OculusReader

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

"""
piper机械臂关节操作效果:
full_joints[0]  左右旋转 yaw    -2.618 ~ 2.618 逆时针/顺时针 
full_joints[1]  上下        0 ~ 3.14     平/下
full_joints[2]  小臂打开     -2.967 ~ 0        后伸展/平
full_joints[3]              -1.745 ~ 1.745
full_joints[4]  手腕旋转     -1.22 ~ 1.22
full_joints[5]  小臂上下摆  -2.094 ~ -2.094
 
full_joints[6]  左夹爪      0 ~ 0.035
full_joints[7]  右夹爪      -0.035 ~ 0

"""


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PushCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

# 为Piper机器人适配的关节映射函数
def get_mapped_joints(robot):
    """获取Piper机器人的关节位置映射"""
    if robot is None:
        return np.zeros(8)  # Piper有8个可控关节
    
    # 获取完整关节位置
    full_joints = robot.get_qpos()
    
    # 转换为numpy数组
    if hasattr(full_joints, 'numpy'):
        full_joints = full_joints.numpy()
    
    # 处理多维数组
    if full_joints.ndim > 1:
        full_joints = full_joints.squeeze()
    
    # Piper只需要前8个关节
    return full_joints[:8]

# 使用mplib的ik逆解
from mplib import Planner
from mplib.pymp import Pose as MPPose  # 导入mplib的Pose类

# 初始化Planner实例 (需要在main函数中调用)
planner = None

# 修改逆运动学函数
def inverse_kinematics(planner, current_qpos, x, y, pitch, last_valid_joints):
    """
    使用mplib的IK函数计算逆运动学
    :param planner: Planner实例
    :param current_qpos: 当前关节位置
    :param x: 末端执行器X坐标
    :param y: 末端执行器Y坐标
    :param pitch: 俯仰角
    :param last_valid_joints: 上一次有效的关节位置
    :return: 关节0,1,2,3的角度（基座旋转、肩关节、肘关节、手腕俯仰）
    """
    try:
        # 构建目标位姿 (相对于机器人基座)
        # 注意：Piper的工作空间主要在XZ平面，Y固定为0
        # 俯仰角(pitch)应用于绕Y轴的旋转
        
        # 计算旋转四元数 (绕Y轴旋转pitch角度)
        cy = math.cos(pitch * 0.5)
        sy = math.sin(pitch * 0.5)
        quat = [cy, 0.0, sy, 0.0]  # [w, x, y, z]
        
        # 修正坐标系：Piper基座坐标系X向前，Z向上
        # 目标位置：X距离，Y高度（转换为Z）
        goal_pose = MPPose([x, 0, y], quat)
        
        # 设置mask - 只优化前4个关节 (基座旋转和手臂关节及手腕俯仰)
        mask = [False] * len(current_qpos)  # 所有关节默认不固定
        mask[4:] = [True] * (len(current_qpos) - 4)  # 固定手腕旋转和夹爪关节
        
        # 调用IK求解器
        status, q_goals = planner.IK(
            goal_pose,
            current_qpos,
            mask,
            n_init_qpos=100,  # 随机采样点数量
            threshold=0.1,   # 进一步增大容差阈值
            return_closest=True,    #设置返回最近解
            verbose=True      # 开启详细输出
        )
        
        # 如果IK成功
        if q_goals is not None:
            print(f"使用IK最近解: {q_goals}")
            return q_goals[0], q_goals[1], q_goals[2], q_goals[3]
        else:
            # IK失败,需尝试增大n_init_qpos,或检查位姿是否可达
            # 没有最近解，使用上一次有效解
            print(f"IK失败: {status}, 使用上一次有效关节位置")
            return last_valid_joints[0], last_valid_joints[1], last_valid_joints[2], last_valid_joints[3]
    
    except Exception as e:
        print(f"逆运动学计算异常: {e}, 使用上一次有效关节位置")
        return last_valid_joints[0], last_valid_joints[1], last_valid_joints[2], last_valid_joints[3]


def rotation_matrix_to_euler_angles(R):
    """
    将旋转矩阵转换为欧拉角 (roll, pitch, yaw)
    """
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def main(args: Args):
    # 定义机械臂工作空间范围    可修改
    MAX_X = 0.5  # 最大水平距离
    MAX_Y = 0.3  # 最大高度

    # 关节限制
    JOINT_LIMITS = [
        (-2.618, 2.618),   # joint1
        (0, 3.14),         # joint2
        (-2.967, 0),       # joint3
        (-1.745, 1.745),   # joint4
        (-1.22, 1.22),     # joint5
        (-2.0944, 2.0944)  # joint6
    ]


    pygame.init()
    
    screen_width, screen_height = 600, 600  # 控制窗口大小
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("VR Control for Piper Robot")
    font = pygame.font.SysFont(None, 24)
    
    # 初始化OculusReader
    oculus_reader = OculusReader(print_positions=False)
    
    # 定义VR控制器的初始位置和缩放因子
    initial_x = 0.0
    initial_y = -0.1
    initial_z = -0.15

    vr_scale_y = 1
    vr_scale_x = 0.8
    rotation_scale = 0.8

    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
        if len(env_kwargs["robot_uids"]) == 1:
            env_kwargs["robot_uids"] = env_kwargs["robot_uids"][0]
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    
    # Piper机器人动作空间为8维
    action = np.zeros(8) if env.action_space is None else np.zeros_like(env.action_space.sample())
    
    # 初始化target_joints变量
    target_joints = np.zeros(8)
    
    # 初始化末端执行器位置
    initial_ee_pos = np.array([0.247, -0.023])
    ee_pos = initial_ee_pos.copy()
    
    # 初始化俯仰角调整
    initial_pitch = 0.0
    pitch = initial_pitch
    
    # 定义末端执行器尖端长度
    tip_length = 0.108
    
    # 定义比例增益
    p_gain = np.ones(8)  # 8个关节
    p_gain[0:6] = 1.0   # 手臂关节
    p_gain[6:8] = 0.05  # 夹爪关节
    
    # 获取机器人实例
    robot = None
    if hasattr(env.unwrapped, "agent"):
        robot = env.unwrapped.agent.robot
    elif hasattr(env.unwrapped, "agents") and len(env.unwrapped.agents) > 0:
        robot = env.unwrapped.agents[0]
    
    print("robot", robot)
    print("")

    # 初始化上一次有效关节位置
    last_valid_joints = np.zeros(8)  # 初始化为零
    if robot:
        last_valid_joints = get_mapped_joints(robot).copy()

# 初始化MPlib的Planner 
    global planner  # 使用全局planner实例
    if robot:
        # piper的URDF和SRDF文件路径
        planner = Planner(
            urdf="/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/piper/piper.urdf",
            srdf="/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/piper/piper.srdf",
            move_group="gripper_base",  # URDF中的末端执行器名称
            #下面两个列表用于定义连杆和关节顺序,在sapein仿真器中需要用到
            user_link_names = [
                "dummy_link",
                "base_link",
                "link1",
                "link2",
                "link3",
                "link4",
                "link5",
                "link6",
                "gripper_base",
                "link7",
                "link8",
            ],
            user_joint_names = [
                "joint1",  # 基座旋转关节
                "joint2",  # 肩关节
                "joint3",  # 肘关节
                "joint4",  # 手腕俯仰关节
                "joint5",  # 手腕旋转关节
                "joint6",  # 手腕旋转关节2
                "joint7",  # 左夹爪关节
                "joint8"   # 右夹爪关节
            ],
            # use_convex=True,  # 使用凸碰撞网格，提高性能  会报错,需要用base_link.STL.convex.stl文件
            # verbose=True  # 开启详细输出
        )

        # 设置planner的机器人基座姿态为仿真中的实际位置
        base_pose_tensor = robot.pose.raw_pose
        # 提取位置和四元数
        position = base_pose_tensor[0, :3].tolist()  # 前3个元素是位置
        quaternion = base_pose_tensor[0, 3:].tolist()  # 后4个元素是四元数
        planner.set_base_pose(MPPose(position, quaternion))
    
    # 打印planner基本信息
    if planner:
        # 打印工作空间信息
        print("Planner初始化信息:")
        print(f"机器人基座位置: {planner.robot.get_base_pose()}")
        print(f"末端执行器链接: {planner.move_group}")
        print(f"可规划关节数: {len(planner.move_group_joint_indices)}")
        print(f"关节限制: {planner.joint_limits}")
        
        # 设置初始状态
        planner.robot.set_qpos(last_valid_joints, True)
        
        # 检查初始状态是否有效
        collisions = planner.planning_world.check_collision()   #TODO:初始姿态发生了碰撞,导致后续IK始终失败
        if collisions:
            print(f"初始状态碰撞检测: {collisions}")

        # 打印初始末端执行器位置
        ee_pose = planner.pinocchio_model.get_link_pose(planner.move_group_link_id)
        print(f"初始末端执行器ee_pose: {ee_pose}")


    # 获取初始关节位置
    current_joints = get_mapped_joints(robot)
    
    # 设置初始目标关节位置
    target_joints = np.zeros_like(current_joints)
    # 调整为安全姿态：肩关节抬高，肘关节弯曲，手腕保持水平
    target_joints[0] = 0.0      # 基座旋转 - 中立位置
    target_joints[1] = 1.2      # 肩关节 - 抬高
    target_joints[2] = -1.2     # 肘关节 - 弯曲
    target_joints[3] = 0.0      # 手腕俯仰 - 水平
    target_joints[4] = 0.0      # 手腕旋转 - 中立
    target_joints[5] = 0.0      # 手腕旋转2 - 中立
    target_joints[6] = 0.035    # 左夹爪 - 打开
    target_joints[7] = -0.035   # 右夹爪 - 打开

    # 夹爪控制变量
    gripper_force = 0.0

    # 初始化控制器旋转角
    controller_euler = np.zeros(3)


    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                oculus_reader.stop()
                return
            elif event.type == pygame.KEYDOWN:
                # 按R键重置位置
                if event.key == pygame.K_r:
                    ee_pos = initial_ee_pos.copy()
                    pitch = initial_pitch
                    target_joints = np.zeros(8)
                    # 调整为安全姿态：肩关节抬高，肘关节弯曲，手腕保持水平
                    target_joints[0] = 0.0      # 基座旋转 - 中立位置
                    target_joints[1] = 1.2      # 肩关节 - 抬高
                    target_joints[2] = -1.2     # 肘关节 - 弯曲
                    target_joints[3] = 0.0      # 手腕俯仰 - 水平
                    target_joints[4] = 0.0      # 手腕旋转 - 中立
                    target_joints[5] = 0.0      # 手腕旋转2 - 中立
                    target_joints[6] = 0.035    # 左夹爪 - 打开
                    target_joints[7] = -0.035   # 右夹爪 - 打开
                    try:
                        compensated_y = ee_pos[1] - tip_length * math.sin(pitch)
                        # 调用逆运动学函数
                        target_joints[1], target_joints[2] = inverse_kinematics(
                            planner, 
                            current_joints, 
                            ee_pos[0], 
                            compensated_y,
                            pitch
                        )
                    except Exception as e:
                        print(f"重置时计算逆运动学出错: {e}")
                        # 出错时保持当前关节位置
                        target_joints[1] = current_joints[1]
                        target_joints[2] = current_joints[2]
                    print("位置已重置为初始值")
        
        # 获取VR控制器数据
        transforms, buttons = oculus_reader.get_transformations_and_buttons()
        
        print(f"oculus读取结果 transforms:{transforms}")
        print(f"buttons:{buttons}")
        

        # 更新目标关节位置 
        # 获取控制器位置
        controller_pos = None
        
        # 使用右手控制器
        if 'r' in transforms:
            controller_pos = transforms['r'][:3, 3]
            # 获取控制器旋转
            controller_rot = transforms['r'][:3, :3]
            controller_euler = rotation_matrix_to_euler_angles(controller_rot)
            # 计算调整后的位置
            x_new = controller_pos[0] - initial_x
            y_new = controller_pos[1] - initial_y
            z_new = -controller_pos[2] + initial_z
            
            # 计算水平距离并限制终端执行器范围
            r = math.sqrt(x_new**2 + z_new**2) * vr_scale_x
            ee_pos[0] = min(max(r, 0), MAX_X)  # 限制X范围                
            ee_pos[1] = min(max(y_new * vr_scale_y, -MAX_Y), MAX_Y)# 限制Y范围

            # 使用控制器旋转控制末端执行器姿态
            # pitch角控制夹爪俯仰
            pitch = controller_euler[1] * rotation_scale
            
            # 计算旋转角度
            if abs(x_new) > 0.05 or abs(z_new) > 0.05:
                rotation_angle = math.atan2(x_new, z_new)
                target_joints[0] = max(min(rotation_angle * 1.2, JOINT_LIMITS[0][1]), JOINT_LIMITS[0][0])  # 基座旋转     #控制基座
        
        # 处理夹爪控制
        if transforms and buttons:
            # 使用Trig或Grip键控制夹爪压力,同时按下则闭合,任意一个键松开则打开
            #rightTrig RTr  右手柄前键  rightGrip RG 右手柄侧键
            gripper_force = min(buttons['rightTrig'][0], buttons['rightGrip'][0])   #根据按键最轻的力度设置夹爪力度.
            # 映射到夹爪目标位置 ([6] <= 0 [7] >= 0 = 闭合, [6] >= 0.1 [7] <= -0.1 = 打开)
            # gripper_force=0时完全打开, gripper_force=1时完全闭合
            target_joints[6] = 1 * (0.9 - gripper_force)    # [6] = 0.135(开) ~ -0.015(闭)
            target_joints[7] = -1 * (0.9 - gripper_force)   # [7] = -0.135(开) ~ 0.015(闭)
            print(f"gripper_force: {gripper_force}, target_joints[6]: {target_joints[6]}, target_joints[7]: {target_joints[7]}")

            # 额外的夹爪控制：A/B按钮用于精细控制
            if 'A' in buttons and buttons['A']:
                # 按下A按钮时轻微闭合
                target_joints[6] -= 0.1
                target_joints[7] += 0.1
            if 'B' in buttons and buttons['B']:
                # 按下B按钮时轻微打开
                target_joints[6] += 0.1
                target_joints[7] -= 0.1
    
        # 计算逆运动学
        try:
            compensated_y = ee_pos[1] - tip_length * math.sin(pitch) # 补偿末端执行器长度
            current_joints = get_mapped_joints(robot)   # 获取当前关节位置

            print(f"IK输入: x={ee_pos[0]:.3f}, y={compensated_y:.3f}, pitch={pitch:.3f}")
            print(f"当前关节: {np.round(current_joints, 3)}")
            
            # 调用逆运动学函数
            target_joints[0], target_joints[1], target_joints[2], target_joints[3] = inverse_kinematics(    #控制手臂
                planner, 
                current_joints, 
                ee_pos[0], 
                compensated_y,
                pitch,
                last_valid_joints  # 传入上一次有效关节位置
            )

            # 更新上一次有效关节位置
            last_valid_joints[0] = target_joints[0]
            last_valid_joints[1] = target_joints[1]
            last_valid_joints[2] = target_joints[2]
            last_valid_joints[3] = target_joints[3]

            print(f"目标关节: {np.round(target_joints, 3)}")
        except Exception as e:
            print(f"计算逆运动学出错: {e}")
            # 出错时保持当前关节位置
            target_joints[0] = current_joints[0]
            target_joints[1] = current_joints[1]
            target_joints[2] = current_joints[2]
            target_joints[3] = current_joints[3]

        print("="*50)
        

        # 设置小臂俯仰（来自VR控制器）
        target_joints[4] = - pitch * 10                   #控制pitch

        # 添加关节范围限制
        for i in range(6):  # 只检查前6个关节
            target_joints[i] = max(min(target_joints[i], JOINT_LIMITS[i][1]), JOINT_LIMITS[i][0])
        
        # 比例控制器
        # 应用比例控制
        for i in range(len(action)):
            action[i] = p_gain[i] * (target_joints[i] - current_joints[i])      #将target_joints换算比例,转化为action控制仿真机械臂
        
        # 绘制控制界面
        screen.fill((0, 0, 0))
        
        text = font.render("Piper VR Control:", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        

        
        # 控制说明
        control_texts = [
            "Keyboard Control:",
            "W/S: forward/backward",
            "A/D: turn left/right",
            "R: Reset all positions",
            "Oculus Handle Control:",
            "rightTrig+rightGrip: right claw grisp/release",
            "A/B: right claw close/open",
            "",
        ]
        
        for i, txt in enumerate(control_texts):
            ctrl_text = font.render(txt, True, (255, 255, 255))
            screen.blit(ctrl_text, (10, 40 + i * 25))
        
        # 显示完整关节位置
        y_pos = 40 + len(control_texts) * 25 + 10
        
        full_joints = robot.get_qpos() if robot is not None else np.zeros(8)
        if hasattr(full_joints, 'numpy'):
            full_joints = full_joints.numpy()
        if full_joints.ndim > 1:
            full_joints = full_joints.squeeze()
            
        full_joints_text = font.render(
            f"Joints: {np.round(full_joints, 2)}", 
            True, (255, 150, 0)
        )
        screen.blit(full_joints_text, (10, y_pos))
        y_pos += 25
        
        # 显示当前关节位置
        joints_text = font.render(
            f"Current Joints: {np.round(current_joints, 2)}", 
            True, (255, 255, 0)
        )
        screen.blit(joints_text, (10, y_pos))
        y_pos += 25
        
        # 显示目标关节位置
        target_text = font.render(
            f"Target Joints: {np.round(target_joints, 2)}", 
            True, (0, 255, 0)
        )
        screen.blit(target_text, (10, y_pos))
        y_pos += 35
        
        # 显示末端执行器位置
        ee_text = font.render(
            f"End Effector: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f})", 
            True, (255, 100, 100)
        )
        screen.blit(ee_text, (10, y_pos))
        y_pos += 25
        
        # 显示俯仰角
        pitch_text = font.render(
            f"Pitch: {pitch:.3f}", 
            True, (255, 100, 255)
        )
        screen.blit(pitch_text, (10, y_pos))
        y_pos += 25
        
        # 显示动作值
        action_text = font.render(
            f"Action: {np.round(action, 2)}", 
            True, (255, 255, 255)
        )
        screen.blit(action_text, (10, y_pos))
        
        pygame.display.flip()
        
        # 执行环境步骤
        obs, reward, terminated, truncated, info = env.step(action)
        
        if args.render_mode is not None:
            env.render()
        
        
        
        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
    
    pygame.quit()
    env.close()
    oculus_reader.stop()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)


