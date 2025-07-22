'''
python -m mani_skill.examples.test_IK
'''


import time
import numpy as np

import mplib 
from mplib import Pose

#mplib中使用四元数的方式是wxyz
 
def inverse_kinematics(planner, goal_pose: Pose, start_qpos):    #此处会将传入的带4元数pose自动转化为mplib的Pose格式

    print(f"目标位姿为:{goal_pose}")
    # print(f"传入的当前关节角度为:{start_qpos}")

    #Pose([0.671142, -5.6263e-08, 0.213193], [2.32604e-06, 0.675613, -4.99487e-06, 0.737256]) 失败

    # 调用IK求解器
    status, q_goals = planner.IK(
        goal_pose,
        start_qpos,
        n_init_qpos=20,  # 随机采样点数量
        threshold=1e-3,      # 判断IK成功的距离阈值
        return_closest=False,    #设置返回最近解
        verbose=True      # 开启碰撞有关的详细输出
    )
    if q_goals is not None:
        # print(f"逆解状态为:{status},目标关节角为:{q_goals}")
        return status, q_goals
    else:
        print(f"IK逆解失败! 原因: {status}")
        return status, q_goals

def wxyz_to_xyzw(q):
    # 四元数从 [w, x, y, z] 转为 [x, y, z, w]
    return [q[1], q[2], q[3], q[0]]

def xyzw_to_wxyz(q):
    # 四元数从 [x, y, z, w] 转为 [w, x, y, z]
    return [q[3], q[0], q[1], q[2]]

from scipy.spatial.transform import Rotation
def robot_to_world(local_pose, root_pose):
    """
    机器人坐标系 -> 世界坐标系
    local_pose: [x, y, z, w, qx, qy, qz] (wxyz)
    root_pose:  [x, y, z, w, qx, qy, qz] (wxyz)
    返回: [x, y, z, w, qx, qy, qz] (wxyz)
    """
    # 位置部分
    world_pos = np.array(local_pose[:3]) + np.array(root_pose[:3])

    # 姿态部分 默认机器人坐标系与世界坐标系朝向一致，因此无需改变四元数
    # r_root = Rotation.from_quat(wxyz_to_xyzw(root_pose[3:]))
    # r_local = Rotation.from_quat(wxyz_to_xyzw(local_pose[3:]))
    # world_quat_xyzw = (r_root * r_local).as_quat()
    # world_quat_wxyz = xyzw_to_wxyz(world_quat_xyzw)

    world_quat_wxyz = np.array(root_pose[3:])  
    return np.concatenate([world_pos, world_quat_wxyz])

def world_to_robot(world_pose, root_pose):
    """
    世界坐标系 -> 机器人坐标系
    world_pose: [x, y, z, w, qx, qy, qz] (wxyz)
    root_pose:  [x, y, z, w, qx, qy, qz] (wxyz)
    返回: [x, y, z, w, qx, qy, qz] (wxyz)
    """  
    # world_pose: Pose([0.0561425, -5.6263e-08, 0.213193], [2.32604e-06, 0.675613, -4.99487e-06, 0.737256]), root_pose: Pose([-0.615, 0, 0], [1, 0, 0, 0])
    
    # 位置部分
    local_pos = world_pose.p - root_pose.p  

    # 姿态部分 默认机器人坐标系与世界坐标系朝向一致，因此无需改变四元数
    # r_root = Rotation.from_quat(wxyz_to_xyzw(root_pose[3:]))
    # r_world = Rotation.from_quat(wxyz_to_xyzw(world_pose[3:]))
    # local_quat_xyzw = (r_root.inv() * r_world).as_quat()
    # local_quat_wxyz = xyzw_to_wxyz(local_quat_xyzw)

    local_quat_wxyz = world_pose.q  

    return mplib.Pose(local_pos, local_quat_wxyz)


if __name__ == "__main__":
    #对piper构建planner
    piper_planner = mplib.Planner(
        # urdf="/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/piper/piper.urdf",
        urdf="/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/piper/piper_new.urdf",
        srdf="/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/piper/piper.srdf",
        move_group="gripper_base",
        # move_group="link6",
        joint_vel_limits=np.ones(6),
        joint_acc_limits=np.ones(6)
        )
    
    #对panda的planner
    panda_planner = mplib.Planner(
        urdf="/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf",
        srdf="/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/panda/panda_v2.srdf",
        move_group="panda_hand",
        # move_group="panda_link8",
        # move_group="panda_leftfinger",
        joint_vel_limits=np.ones(7),  #panda共9个关节,这里设为7,忽略最后两个夹爪关节.这是因为设置move_group为panda_hand表示只包含前7个关节
        joint_acc_limits=np.ones(7)
        )

    piper_start_qpos = np.zeros(8)
    panda_start_qpos = np.zeros(9)

    # root_pose = piper_planner.robot.get_root_pose()    #获取piper的根位姿 Pose(raw_pose=tensor([[-0.6150,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000]])) 
    piper_root_pose = mplib.Pose([-0.6150,  0.0000,  0.0000], [1.0000,  0.0000,  0.0000,  0.0000])  #机器人基座位姿 即机器人坐标系原点
    # print("piper root pose:", piper_root_pose)

    
    #使用maniskill中的初始位姿读数，转换为机器人坐标系，再调用IK 成功！
    # 从maniskill拿到的世界坐标系目标位姿 robot.find_link_by_name('gripper_base').pose  
    # Pose(raw_pose=tensor([[-5.5886e-01, -8.7399e-08,  2.1319e-01,  7.3726e-01,  2.5551e-06, 6.7561e-01, -5.0007e-06]])) 
    goal_pose_world = mplib.Pose([-5.5886e-01, -8.7399e-08,  2.1319e-01], [7.3726e-01,  2.5551e-06, 6.7561e-01, -5.0007e-06])
    goal_pose_robot = world_to_robot(goal_pose_world, piper_root_pose)  #将世界坐标系下的目标位姿转换为机器人坐标系下的目标位姿

    piper_status, piper_q_goals = inverse_kinematics(piper_planner, goal_pose_robot, piper_start_qpos)
    print("piper_IK status:", piper_status)
    print("piper_IK solution:", piper_q_goals)
    

    # #测试四元数为1000 失败
    # goal_pose_world = mplib.Pose([-5.5886e-01, -8.7399e-08,  2.1319e-01], [1,  0, 0, 0])
    # goal_pose_robot = world_to_robot(goal_pose_world, piper_root_pose)  #将世界坐标系下的目标位姿转换为机器人坐标系下的目标位姿

    # piper_status, piper_q_goals = inverse_kinematics(piper_planner, goal_pose_robot, piper_start_qpos)
    # print("piper_IK status:", piper_status)
    # print("piper_IK solution:", piper_q_goals)

    # # 从maniskill中拿到的读数 四元数为wxyz    # 转换为Pose对象Pose(position, quaternion)
    # goal_pose = Pose([-0.615, 1.35042e-08, 0.123], [-0.380352, 0.380358, 0.596096, 0.596096]) #期望对应关节角:[0, 1, -1, 0, 0.4, 0, 任意, 任意]

    # #从rviz中拿到的0位姿数据 xyzw
    # goal_pose = mplib.Pose([0.056142471730709076, -5.626303689609813e-08, 0.21319308876991272], [2.326040430489229e-06, 0.6756133437156677, -4.994869414076675e-06, 0.7372561097145081])

    # piper_status, piper_q_goals = inverse_kinematics(piper_planner, goal_pose, piper_start_qpos)
    # print("piper_IK status:", piper_status)
    # print("piper_IK solution:", piper_q_goals)
        
    # goal_pose = mplib.Pose([0.056142471730709076, -5.626303689609813e-08, 0.21319308876991272], [ 0.7372561097145081, 2.326040430489229e-06, 0.6756133437156677, -4.994869414076675e-06])
    
    # piper_status, piper_q_goals = inverse_kinematics(piper_planner, goal_pose, piper_start_qpos)
    # print("piper_IK status:", piper_status)
    # print("piper_IK solution:", piper_q_goals)

    # # time.sleep(100)


    # panda_status, panda_q_goals = inverse_kinematics(panda_planner, goal_pose, panda_start_qpos)
    # print("panda_IK status:", panda_status)
    # print("panda_IK solution:", panda_q_goals)


    # piper_status, piper_q_goals = inverse_kinematics(piper_planner, goal_pose, piper_start_qpos)
    # print("piper_IK status:", piper_status)
    # print("piper_IK solution:", piper_q_goals)

    # status, q_goals = solve_ik(planner, goal_pose, start_qpos, wrt_world=True, n_init_qpos=100, threshold=1e-3, verbose=True)
    # print("IK status:", status)
    # print("IK solution:", q_goals)
    
    

