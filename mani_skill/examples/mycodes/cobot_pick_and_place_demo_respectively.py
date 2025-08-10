'''
python -m mani_skill.examples.mycodes.cobot_pick_and_place_demo_respectively --render-mode="human" --shader="rt-fast" --reward_mode="dense" -c "pd_joint_delta_pos_dual_arm" -e "CobotPickAndPlaceCube-v1" -r "dual_piper"
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
    warmup_steps = 50
    return scene

def _load_robot(env):
    robot = env.unwrapped.agent.robot  

    # robot.set_pose(sapien.Pose(p=[-1.2, 0, -0.9]))    #设置机器人初始位置。对宜家raskog底座
    robot.set_pose(sapien.Pose(p=[-1.15, 0, -0.75]))    #设置机器人初始位置。对mobile_aloha tracer底座
        
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

def generate_trajectory(cube_pose, bin_pose, gripper_init_pose):
    """生成抓取和放置轨迹，调整姿态和高度参数"""
    # 计算抓取姿态 - 竖直向下抓取
    quaternion = [0, 0, 1, 0] #world_pose quan xyzw 竖直向下

    # #倾斜60度姿态抓取
    # #倾斜60度姿态抓取 绕 x 轴旋转的旋转四元数为：q=[sin(θ/2),0,0,cos(θ/2)]
    # q=[sin(15),0,0,cos(15)]≈[0.2588,0,0,0.9659]
    # quaternion = [0.2588,0,0,0.9659]
    Ry = Rotation.from_rotvec([0, np.deg2rad(30), 0])   # 世界 y 轴 +30° 沿y轴顺时针旋转30°
    quaternion = (Ry * Rotation.from_quat(quaternion)).as_quat().tolist()   #斜向下60°

    claw_length = 0.12  # 夹爪长度
    gripper_tip_offset = np.array([-claw_length * np.sin(np.deg2rad(30)), 0, claw_length * np.cos(np.deg2rad(30))])    #设置尖端位姿到夹爪基座的偏移量

    cube_pos = np.array(cube_pose.p)
    bin_pos = np.array(bin_pose.p)
    
    print(f"Planning trajectory:")
    print(f"- From gripper initial pos: {[f'{x:.3f}' for x in gripper_init_pose[:3]]}")
    print(f"- To cube at: {[f'{x:.3f}' for x in cube_pos]}")
    print(f"- To bin at: {[f'{x:.3f}' for x in bin_pos]}\n")

    # 高度参数调整
    approach_height = 0.12  # 接近高度
    grasp_height = 0.0    # 抓取高度
    place_height = 0.08    # 放置高度

    # print("cube_pos为：", cube_pos)
    # print("cube_pos + np.array([0, 0, approach_height])为：", cube_pos + np.array([0, 0, approach_height]))
    # time.sleep(100)
    
    traj = [
        # 1. 初始位置
        # {"xyz": gripper_init_pose[:3], "quat": gripper_init_pose[3:], "gripper": 1},
        {"xyz": gripper_init_pose[:3], "quat": gripper_init_pose[3:], "gripper": 1},
        
        # 2. 方块上方预备位置
        {"xyz": cube_pos + np.array([0, 0, approach_height] + gripper_tip_offset), 
         "quat": quaternion, "gripper": 1},
         
        # 3. 下降到抓取位置
        {"xyz": cube_pos + np.array([0, 0, grasp_height] + gripper_tip_offset),
         "quat": quaternion, "gripper": 1},
         
        # 4. 闭合夹爪
        {"xyz": cube_pos + np.array([0, 0, grasp_height] + gripper_tip_offset),
         "quat": quaternion, "gripper": 0},
         
        # 5. 抬起
        {"xyz": cube_pos + np.array([0, 0, approach_height] + gripper_tip_offset),
         "quat": quaternion, "gripper": 0},
         
        # 6. 移动到盒子上方
        {"xyz": bin_pos + np.array([0, 0, approach_height] + gripper_tip_offset),
         "quat": quaternion, "gripper": 0},
         
        # 7. 下降到放置位置
        {"xyz": bin_pos + np.array([0, 0, place_height] + gripper_tip_offset),
         "quat": quaternion, "gripper": 0},
         
        # 8. 释放物体
        {"xyz": bin_pos + np.array([0, 0, place_height] + gripper_tip_offset),
         "quat": quaternion, "gripper": 1},
         
        # 9. 抬起
        {"xyz": bin_pos + np.array([0, 0, approach_height] + gripper_tip_offset),
         "quat": quaternion, "gripper": 1},
         
        # 10. 返回初始位置
        {"xyz": gripper_init_pose[:3], "quat": gripper_init_pose[3:], "gripper": 1}
    ]

    # print("生成轨迹为：", traj)
    
    return traj

def cobot_pick_and_place_demo(env):
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
            if "red" in actor.get_name():
                cubes["red"] = actor
            elif "blue" in actor.get_name():
                cubes["blue"] = actor
        elif "bin" in actor.get_name():
            if "red" in actor.get_name():
                bins["red"] = actor
            elif "blue" in actor.get_name():
                bins["blue"] = actor

    if "blue" not in cubes or "red" not in cubes or "blue" not in bins or "red" not in bins:
        print("错误：场景中没有足够的方块或bin")
        return

    print("="*50)
    print("开始双机械臂协同抓取任务")
    print("="*50)

    # 左臂抓取红色方块放到红色bin
    print("\n>>> 左臂抓取红色方块并放置到红色bin <<<")
    cur_left_base_pose = robot.find_link_by_name('left_arm_gripper_base').pose.raw_pose[0].tolist()
    left_traj = generate_trajectory(
        cubes["red"].get_pose(),    
        bins["red"].get_pose(),     
        cur_left_base_pose
    )
    execute_trajectory(left_traj, "left")

    # 右臂抓取蓝色方块放到蓝色bin
    print("\n>>> 右臂抓取蓝色方块并放置到蓝色bin <<<")
    cur_right_base_pose = robot.find_link_by_name('right_arm_gripper_base').pose.raw_pose[0].tolist()
    right_traj = generate_trajectory(
        cubes["blue"].get_pose(),     
        bins["blue"].get_pose(),      
        cur_right_base_pose
    )
    execute_trajectory(right_traj, "right")

    print("\n" + "="*50)
    print("任务完成！")
    print("="*50)


    # self.robot.set_pose(
    #     sapien.Pose(p=[-0.35, 0, 0])
    # )
    # root_pose = self.robot.get_root_pose().raw_pose[0].tolist() #基座位姿
    # # print(f"基座位姿：{root_pose}")

    # # print(f"末端执行器初始位姿(wxyz)：{self.robot.find_link_by_name('gripper_base').pose.raw_pose[0].tolist()}") #[0.737256, 2.5551e-06, 0.675613, -5.0007e-06]

    # cube_pose = self.env.get_wrapper_attr('obj').pose.raw_pose[0].tolist()  #或 self.env.unwrapped.obj.scene.actors['cube'].pose.raw_pose[0].tolist()
    # print(f"物品坐标位于：{cube_pose[:3]}\n")

    # cube_pos_top = cube_pose[:3].copy()
    # cube_pos_top[2] += 0.16
    # pose1_world = cube_pos_top + [0, 0, 1, 0]  
    # pose1_robot = self.world_to_robot(pose1_world, root_pose) #转换到机械臂坐标系

def execute_trajectory(traj, arm_name):
    for step in traj:
        pose_world = np.concatenate([step["xyz"], step["quat"]]).tolist()
        print("执行目标世界坐标系pose_world：",pose_world)
        base_link_name = f"{arm_name}_arm_base_link"
        base_link = robot.find_link_by_name(base_link_name)
        if base_link is None:
            print(f"错误：未找到基座链接 {base_link_name}")
            continue
            
        if _move_arm_to_pose(pose_world, arm_name) < 0:
            # print(f"轨迹点执行失败，暂停3秒后继续执行后续点\n")
            # time.sleep(3)
            print(f"===========================轨迹点执行失败===========================\n")
            continue
        _set_gripper(arm_name, "open" if step["gripper"] else "close")

        print("轨迹点执行成功,当前末端执行器位姿：", robot.find_link_by_name(f"{arm_name}_arm_gripper_base").pose.raw_pose[0].tolist(),"\n")

def _move_arm_to_pose(pose_world, arm_name):
    """
    移动指定机械臂到目标位姿
    current_arm_joints: 当前手臂的关节状态数组
    """
    arm_idx = 0 if arm_name == "left" else 1
    planner = planners[arm_idx]
    
    try:
        # 获取当前完整关节状态
        current_qpos = robot.get_qpos()[0].numpy().squeeze()
        # 提取当前手臂关节状态
        joint_indices = arm_joint_indices[arm_name]
        current_arm_qpos = current_qpos[joint_indices]

        # print(f"规划机械臂 {arm_name} arm")
        # print(f"目标世界坐标系位姿: {[f'{x:.3f}' for x in pose_world]}")
        # print(f"当前关节角: {[f'{x:.3f}' for x in current_arm_qpos]}")
        # print(f"关节数量: {len(current_arm_joints)}")

        # 确保关节数量匹配规划器预期
        if len(current_arm_qpos) != planner.joint_limits.shape[0]:
            print(f"关节数量不匹配! 规划器期望 {planner.joint_limits.shape[0]} 个关节, 实际有 {len(current_arm_qpos)}")
            return -1
            
        pose_robot = world_to_robot(pose_world, robot.find_link_by_name(f'{arm_name}_arm_base_link').pose.raw_pose[0].tolist()) #获取当前机械臂root位姿，从而将目标点坐标转换到机械臂坐标系
        # print(f"传给mplib的机器人坐标系坐标：{pose_robot}")

        result = planner.plan_screw(
            mplib.Pose(pose_robot[:3], pose_robot[3:]),         #必须使用机器人坐标系的值！必须转化为pose对象！
            current_arm_qpos,  # 使用当前手臂关节状态。必须是最新的结果
            time_step=1/140
        )
        
        if result["status"] != "Success":
            print(f"机械臂{arm_name} screw规划失败，尝试RRTConnect")
            result = planner.plan_pose(
                mplib.Pose(pose_robot[:3], pose_robot[3:]), 
                current_arm_qpos,
                time_step=1/140
            )
            if result["status"] != "Success":
                print(f"机械臂{arm_name} RRTConnect规划失败: {result['status']}\n")
                return -1
            else:
                print(f"===========================机械臂{arm_name} RRTConnect规划成功!===========================")
        else:
            print(f"===========================机械臂{arm_name} screw规划成功!===========================")

        # 执行规划的轨迹
        _follow_path(result, planner, arm_idx)
        return 0
        
    except Exception as e:
        import traceback
        print(f"规划出错: {e}")
        traceback.print_exc()   #把完整堆栈跟踪打印到终端，告诉你异常发生在哪一行、哪个函数，方便调试。
        return -1
   
def _follow_path(result, planner, arm_idx, headless=False):
    n_step = result["position"].shape[0]
    print(f"follow_path中得到的plan给出的总时间步数:{n_step} 总用时：{result['duration']}")

    # 实时取出当前 arm 对应的关节
    prefix = "left" if arm_idx == 0 else "right"
    
    for i in range(n_step):
        qf = robot.compute_passive_force(
            gravity=True, coriolis_and_centrifugal=True
        )
        robot.set_qf(qf)

        # 设置关节位置/速度
        arm_joints_index = arm_joint_indices[prefix]    #对应机械臂的joint所在index
        for j in range(len(planner.move_group_joint_indices)):
            joint = robot.get_active_joints()[arm_joints_index[j]]
            joint.set_drive_target(result["position"][i][j])
            joint.set_drive_velocity_target(result["velocity"][i][j])

        scene.step()
        if i % 4 == 0 and not headless:
            env.render()
    


def _set_gripper(arm_name, state):
    prefix = "left" if arm_name == "left" else "right"
    arm_active_joints = [j for j in robot.get_active_joints() if j.get_name().startswith(f"{prefix}_arm_joint")]       #是这里不对，设置关节角时根据left检索，会把left_wheel检索进去，从而导致控制时错位（已修复）
                         
    target_pos = 0.2 if state == "open" else 0.0
    arm_active_joints[-2].set_drive_target(target_pos)
    arm_active_joints[-1].set_drive_target(-target_pos)

    for i in range(100):
        qf = robot.compute_passive_force(
            gravity=True, coriolis_and_centrifugal=True
        )
        robot.set_qf(qf)
        scene.step()
        if i % 4 == 0:
            scene.update_render()
            env.render()

# 定义统计时间的装饰器
def timeit(label):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{label} 耗时: {end - start:.6f} 秒")
            return result
        return wrapper
    return decorator

# 包装需要统计的函数
@timeit("env.step")
def timed_step(env, action):
    return env.step(action)

@timeit("scene.update_render")
def timed_update_render(scene):
    return scene.update_render()


@timeit("env.render")
def timed_render(env):
    return env.render()



def demo():
    global env, scene, robot, planners, arm_joint_indices
    # 初始化环境和机器人
    env = _load_env()
    scene = _load_scene(env)
    robot = _load_robot(env)

    # 创建关节索引映射
    joint_index_map = {}
    for idx, joint in enumerate(robot.get_active_joints()):
        joint_index_map[joint.get_name()] = idx

    # 获取左右臂关节索引
    arm_joint_indices = {
        "left": [idx for name, idx in joint_index_map.items() if "left_arm_joint" in name], #[8, 10, 12, 14, 16, 18, 20, 21]

        "right": [idx for name, idx in joint_index_map.items() if "right_arm_joint" in name] #[9, 11, 13, 15, 17, 19, 22, 23]
    }

    # print(f"左臂关节索引: {arm_joint_indices['left']}")
    # print(f"右臂关节索引: {arm_joint_indices['right']}")

    # # 调试左右机械臂qpos对应关系
    # for i in range(10000):
    #     # 打印左右机械臂的关节位置（qpos），小数形式输出2位
    #     if i % 10 == 0:
    #         left_qpos = robot.get_qpos()[0].numpy().squeeze()[arm_joint_indices["left"]]    #这里获取的即为左臂的全部qpos值,robot.get_qpos()[0].numpy().squeeze()的[8, 10, 12, 14, 16, 18, 20, 21]
    #         right_qpos = robot.get_qpos()[0].numpy().squeeze()[arm_joint_indices["right"]]
    #         left_qpos_str = "[" + ", ".join([f"{x:.2f}" for x in left_qpos]) + "]"
    #         right_qpos_str = "[" + ", ".join([f"{x:.2f}" for x in right_qpos]) + "]"
    #         print(f"左臂 qpos: {left_qpos_str}")
    #         print(f"右臂 qpos: {right_qpos_str}\n")
    #         scene.step()
    #         env.render()


    planners = _setup_planners()

    for i in range(10):  # 热身
        scene.step()
        env.render()

    # 运行抓取演示
    cobot_pick_and_place_demo(env)


if __name__ == "__main__":
    args = tyro.cli(Args)
    demo()