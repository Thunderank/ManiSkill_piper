'''
python -m mani_skill.examples.mycodes.test_cobot_pick_and_place_demo --render-mode="human" --shader="rt-fast" --reward_mode="dense" -c "pd_joint_delta_pos_dual_arm" -e "CobotPickAndPlaceCube-v1" -r "dual_piper"

该函数使用插值法实现，实现了与mplib路径规划的解耦，从而规避了mplib可行域小的缺陷，便于快速验证路径点设置正确性



直接同时设置双臂的qpos,从而无需异步即可实现同步控制

获取目标关节：robot.find_link_by_name(link_name)
获取目标关节位姿：robot.find_link_by_name(link_name).pose.raw_pose[0].tolist()

获取左臂所有关节：
left_arm_active_joints = [j for j in robot.get_active_joints() if j.get_name().startswith("left_arm")]

获取编号为index的关节角：
robot.get_qpos()[0].numpy()[index]

获取左右臂关节索引:
    # 创建关节索引映射
    joint_index_map = {}
    for idx, joint in enumerate(robot.get_active_joints()):
        joint_index_map[joint.get_name()] = idx

    # 获取左右臂关节索引
    arm_joint_indices = {
        "left": [idx for name, idx in joint_index_map.items() if "left_joint" in name], #[8, 10, 12, 14, 16, 18, 20, 21]

        "right": [idx for name, idx in joint_index_map.items() if "right_joint" in name] #[9, 11, 13, 15, 17, 19, 22, 23]
    }


'''
import gymnasium as gym
import numpy as np
import sapien
import mplib 
import time
from scipy.spatial.transform import Rotation

# 注册参数
import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "GraspCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "pointcloud"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = "dual_piper"
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_delta_pos_dual_arm"
    """Control mode"""

    render_mode: str = "human"
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

    base: str = "tracer"

def _load_env():
    env = gym.make(
        args.env_id,    # "CobotPickAndPlaceCube-v1"
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        reward_mode=args.reward_mode,
        robot_uids=args.robot_uids,
        sensor_configs=dict(shader_pack="rt-fast"),
        human_render_camera_configs=dict(shader_pack="rt-fast"),
        viewer_camera_configs=dict(shader_pack="rt-fast"),
        enable_shadow=True,
    )
    obs, _ = env.reset(options=dict(reconfigure=True, robot_uids="dual_piper"))
    return env
        
def _load_scene(env):
    scene = env.unwrapped.scene
    scene.set_timestep(1 / 120)
    # 设置热身步数，以完成场景加载
    step_counter = 0
    warmup_steps = 10
    return scene

def _load_robot(env):
    robot = env.unwrapped.agent.robot  

    # robot.set_pose(sapien.Pose(p=[-1.2, 0, -0.9]))    #设置机器人初始位置。对宜家raskog底座
    robot.set_pose(sapien.Pose(p=[-1.2, 0, -0.75]))    #设置机器人初始位置。对mobile_aloha tracer底座
        
    # 设置初始关节角度
    init_qpos = np.zeros(robot.dof)
    robot.set_qpos(init_qpos)
    
    return robot

def _setup_planners(arm_names = ["piper_left_arm", "piper_right_arm"], gripper_names = ["left_arm_gripper_base", "right_arm_gripper_base"]):
    """为每个机械臂设置规划器"""
    planners = []
    assert len(arm_names) == len(gripper_names), "传入的arm_names与gripper_names个数不匹配"
    for i in range(len(arm_names)):
        # print(f"urdf路径：/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/{args.robot_uids}/{arm_names[i]}.urdf.xacro") #/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/dual_piper/piper_left_arm.urdf.xacro
        # print(f"srdf路径：/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/{args.robot_uids}/{args.robot_uids}_{args.base}.srdf")    #home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/dual_piper/dual_piper.srdf
        planner = mplib.Planner(
            urdf=f"/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/{args.robot_uids}/{arm_names[i]}.urdf.xacro",
            srdf=f"/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/{args.robot_uids}/{args.robot_uids}_{args.base}.srdf",
            move_group=gripper_names[i],
        )
        planners.append(planner)
    
    return planners

def robot_to_world(local_pose: list, root_pose: list):
    """
    机器人坐标系 -> 世界坐标系
    local_pose: [x, y, z, w, qx, qy, qz] (wxyz)
    root_pose:  [x, y, z, w, qx, qy, qz] (wxyz)
    返回: [x, y, z, w, qx, qy, qz] (wxyz)
    """
    # 1. 位置
    world_pos = (np.array(local_pose[:3]) + np.array(root_pose[:3])).tolist()

    # 2. 姿态
    # wxyz → xyzw
    q_local = [local_pose[4], local_pose[5], local_pose[6], local_pose[3]]
    q_root  = [root_pose[4],  root_pose[5],  root_pose[6],  root_pose[3]]

    r_local = Rotation.from_quat(q_local)
    r_root  = Rotation.from_quat(q_root)
    r_world = r_root * r_local          # 世界姿态 = 根姿态 ⊕ 局部姿态
    q_world_xyzw = r_world.as_quat()
    q_world_wxyz = [q_world_xyzw[3], q_world_xyzw[0], q_world_xyzw[1], q_world_xyzw[2]] #返回wxyz

    return world_pos + q_world_wxyz

def world_to_robot(world_pose: list, root_pose: list):
    """
    世界坐标系 -> 机器人坐标系
    world_pose: [x, y, z, w, qx, qy, qz] (wxyz)
    root_pose:  [x, y, z, w, qx, qy, qz] (wxyz)
    返回: [x, y, z, w, qx, qy, qz] (wxyz)
    """  
    # 位置部分 减去基坐标  
    local_pos = (np.array(world_pose[:3]) - np.array(root_pose[:3])).tolist() 

    # 姿态转换（使用四元数乘法）
    from scipy.spatial.transform import Rotation
    r_world = Rotation.from_quat([world_pose[4], world_pose[5], world_pose[6], world_pose[3]])  # xyzw
    r_root = Rotation.from_quat([root_pose[4], root_pose[5], root_pose[6], root_pose[3]])  # xyzw
    r_local = r_root.inv() * r_world
    local_quat_xyzw = r_local.as_quat()
    local_quat_wxyz = [local_quat_xyzw[3], local_quat_xyzw[0], local_quat_xyzw[1], local_quat_xyzw[2]]

    return local_pos + local_quat_wxyz  #拼接这两个list对象

def _solve_ik(planner, pose, current_qpos):
    """求解逆运动学"""
    
    # 调用IK求解器
    status, q_goals = planner.IK(
        pose,
        current_qpos,
        n_init_qpos=20,
        threshold=1e-3,
        verbose=True
    )
    
    if status == "Success" and q_goals is not None:
        return q_goals[0]  # 返回第一个解
    else:
        print(f"IK求解失败! {status}    将返回当前关节角")
        return current_qpos

def _move_to_target_qpos(target_qpos, steps=10):
    """平滑设置关节位置"""
    current_qpos = robot.get_qpos()[0].numpy().squeeze()
    target_qpos = np.array(target_qpos)  # 将目标位置转换为 NumPy 数组
    
    # 插值轨迹
    for i in range(steps):
        alpha = i / (steps - 1)
        qpos = current_qpos * (1 - alpha) + target_qpos * alpha
        
        # 设置关节位置
        robot.set_qpos(qpos)
        scene.step()
        env.render()

def _move_to_pose(left_arm_pose_world=None, right_arm_pose_world=None):
    """移动到指定世界坐标位置。如果没传入某个机械臂的位姿就以其当前位姿为默认位姿"""
    if not left_arm_pose_world:
        left_arm_pose_world = robot.find_link_by_name("left_arm_gripper_base").pose.raw_pose[0].tolist()

    if not right_arm_pose_world:
        right_arm_pose_world = robot.find_link_by_name("right_arm_gripper_base").pose.raw_pose[0].tolist()

    target_qpos = [0] * robot.get_dof() #设置初始qpos向量

    #建立关节索引表，用于控制qpos
    joint_index_map = {}
    for idx, joint in enumerate(robot.get_active_joints()):
        joint_index_map[joint.get_name()] = idx

    # 获取左右臂关节索引
    arm_joint_indices = {
        "left": [idx for name, idx in joint_index_map.items() if "left_arm_joint" in name], #[8, 10, 12, 14, 16, 18, 20, 21]
        "right": [idx for name, idx in joint_index_map.items() if "right_arm_joint" in name] #[9, 11, 13, 15, 17, 19, 22, 23]
    }


    #================左臂================
    # 转换为机器人坐标系
    left_arm_root_pose = robot.find_link_by_name("left_arm_base_link").pose.raw_pose[0].tolist()
    left_arm_pose_robot = world_to_robot(left_arm_pose_world, left_arm_root_pose)
    pos = left_arm_pose_robot[:3]
    quat = left_arm_pose_robot[3:]
    left_arm_mplib_pose_robot = mplib.Pose(pos, quat)

    # 求解IK，获取目标关节角
    left_arm_qpos = []
    for index in arm_joint_indices["left"]:
        left_arm_qpos.append(robot.get_qpos()[0].numpy()[index])    #获取左臂所有关节当前角度

    left_arm_qpos = np.array(left_arm_qpos) # 转换为NumPy数组

    left_arm_target_qpos = _solve_ik(planners[0], left_arm_mplib_pose_robot, left_arm_qpos)

    #设置对应关节角
    for i in range(len(left_arm_target_qpos)):
        target_qpos[arm_joint_indices["left"][i]] = left_arm_target_qpos[i]

    #================右臂================
    # 转换为机器人坐标系
    right_arm_root_pose = robot.find_link_by_name("right_arm_base_link").pose.raw_pose[0].tolist()
    right_arm_pose_robot = world_to_robot(right_arm_pose_world, right_arm_root_pose)
    pos = right_arm_pose_robot[:3]
    quat = right_arm_pose_robot[3:]
    right_arm_mplib_pose_robot = mplib.Pose(pos, quat)

    # 求解IK，获取目标关节角
    right_arm_qpos = []
    for index in arm_joint_indices["right"]:
        right_arm_qpos.append(robot.get_qpos()[0].numpy()[index])

    right_arm_qpos = np.array(right_arm_qpos)# 转换为NumPy数组

    right_arm_target_qpos = _solve_ik(planners[1], right_arm_mplib_pose_robot, right_arm_qpos)

    #设置对应关节角
    for i in range(len(right_arm_target_qpos)):
        target_qpos[arm_joint_indices["right"][i]] = right_arm_target_qpos[i]

    
    # 根据关节角移动机器人
    _move_to_target_qpos(target_qpos)
    return True

def _set_gripper(arm_name, state):
    prefix = "left" if arm_name == "left" else "right"
    target_pos = 0.2 if state == "open" else 0.0

    arm_active_joints = [j for j in robot.get_active_joints() if j.get_name().startswith(f"{prefix}_arm_joint")]

    arm_active_joints[-2].set_drive_target(target_pos)
    arm_active_joints[-1].set_drive_target(-target_pos)

    #添加与_move_to_target_qpos逻辑类似的插值渲染
    for i in range(5):
        scene.step()
        env.render()

def generate_trajectory(cube_pose, bin_pose, gripper_init_pose):
    """生成抓取和放置轨迹，调整姿态和高度参数"""
    # 计算抓取姿态 - 竖直向下抓取
    quaternion = [0, 0, 1, 0] #world_pose quan xyzw

    # #倾斜60度姿态抓取 绕 x 轴旋转的旋转四元数为：q=[sin(θ/2),0,0,cos(θ/2)]
    # q=[sin(15),0,0,cos(15)]≈[0.2588,0,0,0.9659]
    # quaternion = [0.2588,0,0,0.9659]

    Ry = Rotation.from_rotvec([0, np.deg2rad(30), 0])   # 世界 y 轴 +30° 沿y轴顺时针旋转30°
    quaternion = (Ry * Rotation.from_quat(quaternion)).as_quat().tolist()

    cube_pos = np.array(cube_pose.p)
    bin_pos = np.array(bin_pose.p)
    
    print(f"Planning trajectory:")
    print(f"- From gripper initial pos: {[f'{x:.3f}' for x in gripper_init_pose[:3]]}")
    print(f"- To cube at: {[f'{x:.3f}' for x in cube_pos]}")
    print(f"- To bin at: {[f'{x:.3f}' for x in bin_pos]}\n")

    # 高度参数调整
    approach_height = 0.16  # 接近高度
    grasp_height = 0.08    # 抓取高度
    place_height = 0.1    # 放置高度

    # print("cube_pos为：", cube_pos)
    # print("cube_pos + np.array([0, 0, approach_height])为：", cube_pos + np.array([0, 0, approach_height]))
    # time.sleep(100)
    
    traj = [
        # 1. 初始位置
        # {"xyz": gripper_init_pose[:3], "quat": gripper_init_pose[3:], "gripper": 1},
        {"xyz": gripper_init_pose[:3], "quat": gripper_init_pose[3:], "gripper": 1},
        
        # 2. 方块上方预备位置
        {"xyz": list(cube_pos + np.array([0, 0, approach_height])), 
         "quat": quaternion, "gripper": 1},
         
        # 3. 下降到抓取位置
        {"xyz": list(cube_pos + np.array([0, 0, grasp_height])),
         "quat": quaternion, "gripper": 1},
         
        # 4. 闭合夹爪
        {"xyz": list(cube_pos + np.array([0, 0, grasp_height])),
         "quat": quaternion, "gripper": 0},
         
        # 5. 抬起
        {"xyz": list(cube_pos + np.array([0, 0, approach_height])),
         "quat": quaternion, "gripper": 0},
         
        # 6. 移动到盒子上方
        {"xyz": list(bin_pos + np.array([-0.1, 0, approach_height])),
         "quat": quaternion, "gripper": 0},
         
        # 7. 下降到放置位置
        {"xyz": list(bin_pos + np.array([-0.1, 0, place_height])),
         "quat": quaternion, "gripper": 0},
         
        # 8. 释放物体
        {"xyz": list(bin_pos + np.array([-0.1, 0, place_height])),
         "quat": quaternion, "gripper": 1},
         
        # 9. 抬起
        {"xyz": list(bin_pos + np.array([-0.1, 0, approach_height])),
         "quat": quaternion, "gripper": 1},
         
        # 10. 返回初始位置
        {"xyz": gripper_init_pose[:3], "quat": gripper_init_pose[3:], "gripper": 1}
    ]

    # print("生成轨迹为：", traj)
    
    return traj

def grasp_cube_demo(env):
    """
    自动识别并抓取方块放到对应颜色的bin
    """
    env.reset()
    scene = env.unwrapped.scene
    
    # 获取场景中的方块和bin
    cubes = {}
    bins = {}
    for actor in scene.get_all_actors():
        if "cube" in actor.get_name():
            if "blue" in actor.get_name():
                cubes["blue"] = actor
            elif "red" in actor.get_name():
                cubes["red"] = actor
        elif "bin" in actor.get_name():
            if "blue" in actor.get_name():
                bins["blue"] = actor
            elif "red" in actor.get_name():
                bins["red"] = actor

    if "red" not in cubes or "blue" not in cubes or "red" not in bins or "blue" not in bins:
        print("错误：场景中没有足够的方块或bin")
        return

    print("="*50)
    print("开始双机械臂协同抓取任务")
    print("="*50)

    # ================ 生成轨迹 ================
    # 左臂抓取蓝色方块放到蓝色bin
    print("\n>>> 生成左臂抓取蓝色方块并放置到蓝色bin轨迹 <<<")
    cur_left_base_pose = robot.find_link_by_name('left_arm_gripper_base').pose.raw_pose[0].tolist()
    left_traj = generate_trajectory(
        cubes["blue"].get_pose(),    # 改为blue
        bins["blue"].get_pose(),     # 改为blue
        cur_left_base_pose
    )

    # 右臂抓取红色方块放到红色bin
    print("\n>>> 生成右臂抓取红色方块并放置到红色bin轨迹 <<<")
    cur_right_base_pose = robot.find_link_by_name('right_arm_gripper_base').pose.raw_pose[0].tolist()
    right_traj = generate_trajectory(
        cubes["red"].get_pose(),     # 改为red
        bins["red"].get_pose(),      # 改为red
        cur_right_base_pose
    )

    # ================ 遵循轨迹 ================
    # 确保左右轨迹长度相同
    assert len(left_traj) == len(right_traj), "左右轨迹长度不一致！"

    for i in range(len(left_traj)):
        print(f"\n步骤 {i+1}/{len(left_traj)}:")
        
        # 获取当前轨迹点
        left_point = left_traj[i]
        right_point = right_traj[i]

        print(f"left_point:{left_point}")
        
        # 提取位姿和夹爪状态
        left_pose = left_point['xyz'] + left_point['quat']
        right_pose = right_point['xyz'] + right_point['quat']
        left_gripper = left_point['gripper']
        right_gripper = right_point['gripper']
        
        # 打印当前目标
        print(f"左臂目标: pos={left_point['xyz']}, gripper={'打开' if left_gripper == 1 else '关闭'}")
        print(f"右臂目标: pos={right_point['xyz']}, gripper={'打开' if right_gripper == 1 else '关闭'}")
        
        # 移动到目标位姿
        _move_to_pose(left_arm_pose_world=left_pose, right_arm_pose_world=right_pose)
        
        # 控制夹爪
        if left_gripper == 1:
            _set_gripper("left", "open")
        else:
            _set_gripper("left", "close")
            
        if right_gripper == 1:
            _set_gripper("right", "open")
        else:
            _set_gripper("right", "close")
        
        # 短暂暂停
        time.sleep(0.1)

    print("\n" + "="*50)
    print("任务完成！")
    print("="*50)


def demo():
    global env, scene, robot, planners
    # 初始化环境和机器人
    env = _load_env()
    scene = _load_scene(env)
    robot = _load_robot(env)
    planners = _setup_planners()

    for i in range(10):  # 热身
        scene.step()
        env.render()

    _move_to_pose()

    # 运行抓取演示
    grasp_cube_demo(env)


if __name__ == "__main__":
    args = tyro.cli(Args)
    demo()