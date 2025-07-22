"""
测试maniskill中action的具体含义:xyz轴的正向,关节角的具体作用
python -m mani_skill.examples.test_maniskill --render-mode="human" --shader="rt-fast" -c "pd_joint_delta_pos" -e "PushCube-v1" -r "piper"
"""

import gymnasium as gym
import numpy as np
import sapien
import pygame
import time
import math

from mani_skill.envs.sapien_env import BaseEnv

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



if __name__ == "__main__":
    args = tyro.cli(Args)
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
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
        if len(env_kwargs["robot_uids"]) == 1:
            env_kwargs["robot_uids"] = env_kwargs["robot_uids"][0]
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    env.render()


    target_joints = np.zeros(8)
    current_joints = np.zeros(8)
    action = np.zeros(8)


    import mplib
    robot = env.unwrapped.agent.robot

    # #测试maniskill中xyz轴的正向,以及四元数的确切分配
    # x = 0   # x轴正向是机械臂前方 1
    # y = 0   # y轴正向是机械臂左侧 1
    # z = 0   # z轴正向是机械臂上方 1
    # w = 1  # 设为1时机械臂朝前
    # ix = 0  # 设为1时机械臂倒下 倒悬朝前 即绕y轴逆时针选择180度
    # iy = 0  # 设为1时机械臂倒下 倒悬朝后 即绕y轴顺时针旋转180度
    # iz = 0   # 设为1时机械臂朝后
    # pos = [x, y, z]
    # quat = [w, ix, iy, iz]  # wxyz
    # pose = sapien.Pose(pos, quat)
    # robot.set_root_pose(pose)


    # # 测试maniskill中四元数顺序是xyzw还是wxyz
    # 运行测试程序：python -m mani_skill.examples.my_piper_demo_ctrl_action_ee --render-mode="human" --shader="rt-fast" -c "pd_joint_delta_pos" -e "PushCube-v1" -r "piper"
    # link1 初始四元数 [1,0,0,0]，按住1绕z轴逆时针旋转180度后变为 [0,0,0,1]
    # 证明maniskill/sapien中四元数的顺序是wxyz
     



    #在此处设置8维变量action,代表每一个关节的角度,执行一次step然后render,即可实现maniskill环境控制
    while True:
        
        target_joints[1] = 1
        target_joints[2] = -1
        target_joints[4] = 0.4
        for i in range(len(action)):
            action[i] = target_joints[i]
        env.step(action)

        env.render()