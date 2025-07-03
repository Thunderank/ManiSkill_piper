import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode


import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

#引入手动控制相关依赖
import signal
from mani_skill.utils import common, visualization
from matplotlib import pyplot as plt
import time
# 处理Ctrl+C中断
signal.signal(signal.SIGINT, signal.SIG_DFL)
# 设置全局参数
ACTION_STEP = 1.0   #end effector执行步长，[0 ~ 1.0]

# 使用dataclass定义命令行参数
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

    # # 打印当前动作空间信息
    # # 若当前是离散动作空间
    # if hasattr(env.action_space, 'n'):  
    #     print(f"离散动作空间包含 {env.action_space.n} 个动作")
    #     # 遍历离散动作空间
    #     try:
    #         for action in env.action_space:
    #             print("当前动作空间包含动作：", action)
    #     except TypeError:
    #         print("离散动作空间过大，无法逐一列举")
    
    # # 当前是连续动作空间
    # elif hasattr(env.action_space, 'shape'):  
    #     print(f"连续动作空间维度：{env.action_space.shape}")
    #     print(f"动作取值范围：{env.action_space.low} 到 {env.action_space.high}")

    # #随机采样动作以观察动作空间

    # while True:
    #     action = env.action_space.sample()
    #     print("随机采样的动作：", action)
    #     time.sleep(0.5)

    # 编写键盘交互控制代码
    # # 覆盖matplotlib键盘交互
    # plt.rcParams["keymap.fullscreen"].remove("f")
    # plt.rcParams["keymap.home"].remove("h")
    # plt.rcParams["keymap.home"].remove("r")
    # plt.rcParams["keymap.back"].remove("c")
    # plt.rcParams["keymap.forward"].remove("v")
    # plt.rcParams["keymap.pan"].remove("p")
    # plt.rcParams["keymap.zoom"].remove("o")
    # plt.rcParams["keymap.save"].remove("s")
    # plt.rcParams["keymap.grid"].remove("g")
    # plt.rcParams["keymap.yscale"].remove("l")
    # plt.rcParams["keymap.xscale"].remove("k")
    

    print("\n===== 关节空间手动控制说明 =====")
    print("关节控制 (7个关节):")
    print("  1/11: 关节1 增加/减少")
    print("  2/22: 关节2 增加/减少")
    print("  3/33: 关节3 增加/减少")
    print("  4/44: 关节4 增加/减少")
    print("  5/55: 关节5 增加/减少")
    print("  6/66: 关节6 增加/减少")
    print("  7/77: 关节7 增加/减少")
    print("夹爪控制:")
    print("  8/88: 夹爪打开/关闭 (f打开, g关闭)")
    print("其他功能:")
    print("  reset: 重置环境")
    print("  esc: 退出")
    print("===========================\n")
    
    # 手动控制主交互循环
    joint_action = np.zeros(8)
    try:
        while True:
            # 检测键盘输入
            control_input = input("请输入控制指令:")

            # 处理键盘输入 - 关节控制
            #直接控制：每次控制直接调整该维度的关节位置
            if control_input == '1':  # 关节1 增加
                joint_action[0] = + ACTION_STEP
            elif control_input == '11':  # 关节1 减少
                joint_action[0] = - ACTION_STEP
            elif control_input == '2':  # 关节2 增加
                joint_action[1] = + ACTION_STEP
            elif control_input == '22':  # 关节2 减少
                joint_action[1] = - ACTION_STEP
            elif control_input == '3':  # 关节3 增加
                joint_action[2] = + ACTION_STEP
            elif control_input == '33':  # 关节3 减少
                joint_action[2] = - ACTION_STEP
            elif control_input == '4':  # 关节4 增加
                joint_action[3] = + ACTION_STEP
            elif control_input == '44':  # 关节4 减少
                joint_action[3] = - ACTION_STEP
            elif control_input == '5':  # 关节5 增加
                joint_action[4] = + ACTION_STEP
            elif control_input == '55':  # 关节5 减少
                joint_action[4] = - ACTION_STEP
            elif control_input == '6':  # 关节6 增加
                joint_action[5] = + ACTION_STEP
            elif control_input == '66':  # 关节6 减少
                joint_action[5] = - ACTION_STEP
            elif control_input == '7':  # 关节7 增加
                joint_action[6] = + ACTION_STEP
            elif control_input == '77':  # 关节7 减少
                joint_action[6] = - ACTION_STEP
            elif control_input == '8':  # 打开夹爪
                joint_action[7] = + ACTION_STEP
            elif control_input == '88':  # 关闭夹爪
                joint_action[7] = - ACTION_STEP
            
            # 重置环境（单独处理r键，避免与关节4的减少键冲突）
            if control_input == 'reset':
                obs, _ = env.reset()
                joint_action = np.zeros(8)
                continue
            
            # ESC键退出
            if control_input == 'esc':
                break
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(joint_action)
            if verbose:
                print(f"动作: {joint_action.round(2)}")
                print(f"奖励: {reward}")
                print(f"终止: {terminated}, 截断: {truncated}")
                print(f"信息: {info}\n")

            #进行渲染，更新action后的环境画面
            if args.render_mode is not None:
                env.render()
            
            joint_action = np.zeros(8)  #重置控制向量
            # # 控制频率（降低循环速度，便于控制）
            # time.sleep(0.1)  # 每次动作间隔0.1秒
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        env.close()
        if record_dir:
            print(f"视频已保存至: {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
