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


# Piper机械臂的空间三连杆逆解（基于URDF参数）
def piper_inverse_kinematics(x, y, z, tip_length=0.108):
    """
    输入: 末端目标(x, y, z)（单位: m, 以base_link为原点）
    输出: joint1, joint2, joint3, joint4
    """
    # 连杆长度（根据piper.urdf）
    L1 = 0.123   # base_link->link1
    L2 = 0.285   # link2
    L3 = 0.25075 # link3

    # 末端补偿（z方向减去末端长度）
    z = z - tip_length

    # joint1: 底座yaw
    theta1 = math.atan2(y, x)
    # 计算在base_link坐标系下，投影到x-z平面
    r = math.sqrt(x**2 + y**2) - L1
    s = z

    # 计算joint2, joint3（平面两连杆逆解）
    D = (r**2 + s**2 - L2**2 - L3**2) / (2 * L2 * L3)
    D = np.clip(D, -1.0, 1.0)
    theta3 = math.atan2(-math.sqrt(1 - D**2), D)  # 肘下解
    theta2 = math.atan2(s, r) - math.atan2(L3 * math.sin(theta3), L2 + L3 * math.cos(theta3))
    # joint4: 保持0或用于末端姿态补偿
    theta4 = 0.0
    return theta1, theta2, theta3, theta4


def main(args: Args):
    pygame.init()
    
    screen_width, screen_height = 600, 600  # 减小窗口尺寸以适应单臂
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("VR Control for Piper Robot")
    font = pygame.font.SysFont(None, 24)
    
    # 初始化OculusReader
    oculus_reader = OculusReader(print_positions=False)
    
    # 定义VR控制器的初始位置和缩放因子
    initial_x = 0.0
    initial_y = -0.1
    initial_z = -0.15
    vr_scale_y = 2.5
    vr_scale_x = 0.8

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

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))  #重置环境,初始化环境与机器人,gymnasium中在env.render()前所必须
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    
    # Piper机器人动作空间为8维
    action = np.zeros(8) if env.action_space is None else np.zeros_like(env.action_space.sample())
    
    # 初始化目标关节位置
    target_joints = np.zeros(8)
    
    
    # 初始化俯仰角调整
    initial_pitch = 0.0
    pitch = initial_pitch
    pitch_step = 0.02
    
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
    
    # 获取初始关节角度.get_qpos()返回二维数组,由于只有一个机械臂,所以维度为[1,1],取首位
    current_joints = robot.get_qpos()[0]

    print(f"current_joints:{current_joints}")
    
    # 设置初始目标关节角
    target_joints = np.zeros_like(current_joints)
    

    while True:
        # 获取VR控制器数据
        transforms, buttons = oculus_reader.get_transformations_and_buttons()
        
        # 更新目标关节位置
        controller_pos = None # 获取控制器位置
        
        # 使用右手控制器
        if 'r' in transforms:
            controller_pos = transforms['r'][:3, 3]
            # 这里假设VR空间和机械臂基座空间对齐，需根据实际标定调整
            x_new = controller_pos[0] - initial_x
            y_new = controller_pos[1] - initial_y
            z_new = -controller_pos[2] + initial_z
            # 缩放xyz并进行ik逆解,根据求得的结果设置关节角
            try:
                # 根据传入的VR坐标,设置 joint1~joint4
                target_joints[:4] = piper_inverse_kinematics(x_new * vr_scale_x + 0.3, y_new * vr_scale_y, z_new * 1.0 + 0.2)
            except Exception as e:
                print(f"计算逆运动学出错: {e}")
            
            # 计算旋转角度
            if abs(x_new) > 0.05 or abs(z_new) > 0.05:
                rotation_angle = math.atan2(x_new, z_new)
                target_joints[3] = rotation_angle  # 旋转关节
        
            # 处理夹爪控制
            if transforms and buttons:
                # 使用Trig或Grip键控制夹爪压力,同时按下则闭合,任意一个键松开则打开
                #rightTrig RTr  右手柄前键  rightGrip RG 右手柄侧键
                gripper_force = min(buttons['rightTrig'][0], buttons['rightGrip'][0])   #根据按键最轻的力度设置夹爪力度.
                # 映射到夹爪目标位置 ([6] <= 0 [7] >= 0 = 闭合, [6] >= 0.1 [7] <= -0.1 = 打开)
                # gripper_force=0时完全打开, gripper_force=1时完全闭合
                target_joints[6] = 1 * (0.9 - gripper_force)    # [6] = 0.135(开) ~ -0.015(闭)
                target_joints[7] = -1 * (0.9 - gripper_force)   # [7] = -0.135(开) ~ 0.015(闭)
                # print(f"gripper_force: {gripper_force}, target_joints[6]: {target_joints[6]}, target_joints[7]: {target_joints[7]}")

                # 额外的夹爪控制：A/B按钮用于精细控制
                if 'A' in buttons and buttons['A']:
                    # 按下A按钮时轻微闭合
                    target_joints[6] -= 0.1
                    target_joints[7] += 0.1
                if 'B' in buttons and buttons['B']:
                    # 按下B按钮时轻微打开
                    target_joints[6] += 0.1
                    target_joints[7] -= 0.1
        
            print(f"当前target_joints:1:{target_joints[0]},2:{target_joints[1]},3:{target_joints[2]},4:{target_joints[3]},5:{target_joints[4]},6:{target_joints[5]},夹爪力度:{gripper_force}")

        # 比例控制器
        # 应用比例控制
        for i in range(len(action)):
            action[i] = p_gain[i] * (target_joints[i] - current_joints[i])
        
        # 绘制控制界面
        screen.fill((0, 0, 0))
        
        text = font.render("Piper VR Control:", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        

        # 控制说明
        control_texts = [
            "Right Controller: Move end effector",
            "Pinch Gesture: Toggle gripper"
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