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
sys.path.append(os.path.abspath("/home/vec/lerobot/oculus_reader"))
from oculus_reader.reader import OculusReader

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

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

# 为Piper机器人适配的2连杆简化逆运动学
def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """为Piper机器人计算逆运动学"""
    # 计算目标点到原点的距离
    r = math.sqrt(x**2 + y**2)
    r = max(abs(l1 - l2), min(l1 + l2, r))  # 限制在工作空间内
    
    # 使用余弦定理计算theta2
    cos_theta2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    
    theta2 = math.acos(cos_theta2)
    
    # 计算theta1
    theta1 = math.atan2(y, x) - math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    
    return theta1, theta2

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
    
    # 初始化目标关节位置
    target_joints = np.zeros(8)
    
    # 初始化末端执行器位置
    initial_ee_pos = np.array([0.247, -0.023])
    ee_pos = initial_ee_pos.copy()
    
    # 初始化俯仰角调整
    initial_pitch = 0.0
    pitch = initial_pitch
    pitch_step = 0.02
    
    # 定义末端执行器尖端长度
    tip_length = 0.108
    
    # 定义步长
    joint_step = 0.01
    ee_step = 0.005
    
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
    
    # 获取初始关节位置
    current_joints = get_mapped_joints(robot)
    
    # 设置初始目标关节位置
    target_joints = np.zeros_like(current_joints)
    
    # 设置热身阶段计数器
    step_counter = 0
    warmup_steps = 50
    
    # 跟踪之前的捏合状态
    prev_pinch = False
    
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
                    try:
                        compensated_y = ee_pos[1] - tip_length * math.sin(pitch)
                        target_joints[0], target_joints[1] = inverse_kinematics(ee_pos[0], compensated_y)
                        target_joints[2] = pitch
                    except Exception as e:
                        print(f"重置时计算逆运动学出错: {e}")
                    print("位置已重置为初始值")
        
        # 获取VR控制器数据
        transforms, buttons = oculus_reader.get_transformations_and_buttons()
        
        # 更新目标关节位置 - 仅在热身阶段后
        if step_counter >= warmup_steps:
            # 获取控制器位置
            controller_pos = None
            
            # 使用右手控制器
            if 'r' in transforms:
                controller_pos = transforms['r'][:3, 3]
                # 计算调整后的位置
                x_new = controller_pos[0] - initial_x
                y_new = controller_pos[1] - initial_y
                z_new = -controller_pos[2] + initial_z
                
                # 计算水平距离
                r = math.sqrt(x_new**2 + z_new**2) * vr_scale_x
                
                # 更新末端执行器位置
                ee_pos[0] = r
                ee_pos[1] = y_new * vr_scale_y
                
                # 计算旋转角度
                if abs(x_new) > 0.05 or abs(z_new) > 0.05:
                    rotation_angle = math.atan2(x_new, z_new)
                    target_joints[3] = rotation_angle  # 旋转关节
            
            # 处理捏合检测
            if buttons and 'r' in buttons and 'pinch' in buttons['r']:
                current_pinch = buttons['r']['pinch']
                if current_pinch and not prev_pinch:
                    # 切换夹爪
                    if target_joints[6] < 0.4 and target_joints[7] < 0.4:
                        target_joints[6] = 0.5  # 打开
                        target_joints[7] = 0.5
                    else:
                        target_joints[6] = 0.0  # 关闭
                        target_joints[7] = 0.0
                prev_pinch = current_pinch
        
        # 计算逆运动学
        try:
            compensated_y = ee_pos[1] - tip_length * math.sin(pitch)
            target_joints[0], target_joints[1] = inverse_kinematics(ee_pos[0], compensated_y)
            target_joints[2] = pitch  # 俯仰角直接控制
        except Exception as e:
            print(f"计算逆运动学出错: {e}")
        
        # 获取当前关节位置
        current_joints = get_mapped_joints(robot)
        
        # 比例控制器
        if step_counter < warmup_steps:
            action = np.zeros(8)
        else:
            # 应用比例控制
            for i in range(len(action)):
                action[i] = p_gain[i] * (target_joints[i] - current_joints[i])
        
        # 绘制控制界面
        screen.fill((0, 0, 0))
        
        text = font.render("Piper VR Control:", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        
        # 热身状态显示
        if step_counter < warmup_steps:
            warmup_text = font.render(f"WARMUP: {step_counter}/{warmup_steps} steps", True, (255, 0, 0))
            screen.blit(warmup_text, (300, 10))
        
        # 控制说明
        control_texts = [
            "R: Reset all positions",
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
        step_counter += 1
        
        if args.render_mode is not None:
            env.render()
        
        time.sleep(0.01)
        
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