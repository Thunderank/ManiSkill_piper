import gymnasium as gym
import numpy as np
import sapien
import pygame
import time
import math

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode


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

#将机器人完整关节位置映射到目标控制关节
def get_mapped_joints(robot):
    # print(f"控制代码中 get_mapped_joints")
    # print(f"robot:{robot}")
    # print(f"robot.get_qpos():{robot.get_qpos()}")

    """Piper 关节映射"""
    if robot is None:
        return np.zeros(8)  # 单臂8个可控制的活动关节
    
    full_joints = robot.get_qpos()
    if hasattr(full_joints, 'numpy'):
        full_joints = full_joints.numpy()
    # print(f"get_mapped_joints中全部关节full_joints:{full_joints}")
    
    if full_joints.ndim > 1:
        full_joints = full_joints.squeeze()

    return full_joints[:8]  # 直接返回所有8个关节，跳过基座关节（joint1-joint8）

#计算2连杆机械臂(分为l1上机械臂、l2下机械臂)的逆运动学，返回关节角度
def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """为Piper机器人适配的2连杆简化逆运动学"""
    # Calculate distance from origin to target point 基础逆运动学
    r = math.sqrt(x**2 + y**2)
    r = max(abs(l1 - l2), min(l1 + l2, r))  # 限制在工作空间内
    
    # 使用余弦定理计算theta2
    cos_theta2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    
    theta2 = math.acos(cos_theta2)
    
    # 计算theta1
    theta1 = math.atan2(y, x) - math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    
    return theta1, theta2   # 返回关节角度（对应joint2和joint3）



"""my keyboard控制方式"""
#引入手动控制相关依赖
import signal
from mani_skill.utils import common, visualization
from matplotlib import pyplot as plt
import time
from pynput import keyboard  # 替换 keyboard 库，需 pip install pynput
# 处理Ctrl+C中断
signal.signal(signal.SIGINT, signal.SIG_DFL)
# 设置全局参数
ACTION_STEP = 0.05   #end effector执行步长，[0 ~ 1.0]

# 手动控制主交互循环
pressed_keys = set()

def on_press(key):
    # print(f"按下键: {key}")
    try:
        k = key.char
    except AttributeError:
        k = key.name
    pressed_keys.add(k)
    # print(f"当前pressed_keys为{pressed_keys}")

def on_release(key):
    # print(f"释放键: {key}")
    try:
        k = key.char
    except AttributeError:
        k = key.name
    pressed_keys.discard(k)
    
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def main(args: Args):
    # 设置NumPy打印格式，处理随机种子
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])

    # 若选择渲染方式为human则在单个场景中进行并行渲染
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False

    # 构建环境配置参数字典
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
    # 处理机器人UID参数
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
        if len(env_kwargs["robot_uids"]) == 1:
            env_kwargs["robot_uids"] = env_kwargs["robot_uids"][0]
    # 创建强化学习环境
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )
    # 如果指定了记录目录，添加视频录制功能
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))

    # 打印环境信息
    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    # 重置环境并设置随机种子
    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    # 初始化渲染，打开渲染窗口viewer
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()

    """                    键盘交互控制代码               """

    print("\n===== 关节空间手动控制说明 =====")
    print("关节控制 (6个关节):")
    print("  1/ctrl+1: 关节1 增加/减少")
    print("  2/ctrl+2: 关节2 增加/减少")
    print("  3/ctrl+3: 关节3 增加/减少")
    print("  4/ctrl+4: 关节4 增加/减少")
    print("  5/ctrl+5: 关节5 增加/减少")
    print("  6/ctrl+6: 关节6 增加/减少")
    print("夹爪控制:")
    print("  7/ctrl+7: 夹爪打开/关闭")
    print("其他功能:")
    print("  r: 重置环境")
    print("  x: 退出")
    print("===========================\n")
    
    # 手动控制主交互循环
    pressed_keys = set()

    def on_press(key):
        # print(f"按下键: {key}")
        try:
            k = key.char
        except AttributeError:
            k = key.name
        pressed_keys.add(k)
        # print(f"当前pressed_keys为{pressed_keys}")

    def on_release(key):
        # print(f"释放键: {key}")
        try:
            k = key.char
        except AttributeError:
            k = key.name
        pressed_keys.discard(k)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        joint_action = np.zeros(8)
        while True:
            claw_control = False
            # 关节1
            if '1' in pressed_keys and 'ctrl' not in pressed_keys:
                joint_action[0] += ACTION_STEP
            if '1' in pressed_keys and 'ctrl' in pressed_keys:
                joint_action[0] -= ACTION_STEP
            # 关节2
            if '2' in pressed_keys and 'ctrl' not in pressed_keys:
                joint_action[1] += ACTION_STEP
            if '2' in pressed_keys and 'ctrl' in pressed_keys:
                joint_action[1] -= ACTION_STEP
            # 关节3
            if '3' in pressed_keys and 'ctrl' not in pressed_keys:
                joint_action[2] += ACTION_STEP
            if '3' in pressed_keys and 'ctrl' in pressed_keys:
                joint_action[2] -= ACTION_STEP
            # 关节4
            if '4' in pressed_keys and 'ctrl' not in pressed_keys:
                joint_action[3] += ACTION_STEP
            if '4' in pressed_keys and 'ctrl' in pressed_keys:
                joint_action[3] -= ACTION_STEP
            # 关节5
            if '5' in pressed_keys and 'ctrl' not in pressed_keys:
                joint_action[4] += ACTION_STEP
            if '5' in pressed_keys and 'ctrl' in pressed_keys:
                joint_action[4] -= ACTION_STEP
            # 关节6
            if '6' in pressed_keys and 'ctrl' not in pressed_keys:
                joint_action[5] += ACTION_STEP
            if '6' in pressed_keys and 'ctrl' in pressed_keys:
                joint_action[5] -= ACTION_STEP
            # 夹爪
            if '7' in pressed_keys and 'ctrl' not in pressed_keys:
                joint_action[6] += ACTION_STEP
                joint_action[7] -= ACTION_STEP
                claw_control = True
            if '7' in pressed_keys and 'ctrl' in pressed_keys:
                joint_action[6] -= ACTION_STEP
                joint_action[7] += ACTION_STEP
                claw_control = True
                

            # 重置环境
            if 'r' in pressed_keys:
                obs, _ = env.reset()
                time.sleep(0.2)
                if args.render_mode is not None:
                    env.render()  # 确保采集一帧再退出，避免报错
                continue
            
            if 'x' in pressed_keys:
                break

            # 若有动作则执行
            if np.any(joint_action[:6] != 0) or claw_control:
                obs, reward, terminated, truncated, info = env.step(joint_action)
                if verbose:
                    print(f"动作: {joint_action.round(2)}")
                    print(f"奖励: {reward}")
                    print(f"终止: {terminated}, 截断: {truncated}")
                    print(f"信息: {info}\n")

            if args.render_mode is not None:
                env.render()
            # time.sleep(0.1)   #加上会导致控制与仿真环境不同步
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        listener.stop()
        env.close()
        if record_dir:
            print(f"视频已保存至: {record_dir}")

if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
