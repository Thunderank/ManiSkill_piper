'''
python -m mani_skill.examples.test_planning_coordinate --render-mode="human" --shader="rt-fast" -c "pd_joint_delta_pos" -e "PushCube-v1" -r "piper"
使用mplib的IK,但不使用plan_pose,而是自定义进行motion planning
'''

import time
import numpy as np
import sapien
import gymnasium as gym
import mplib
from scipy.spatial.transform import Rotation

class ManiSkillPiperDemo:
    def __init__(self):
        # 1. 启动ManiSkill环境
        self.env = gym.make(
            "PushCube-v1",
            obs_mode="none",
            control_mode="pd_joint_delta_pos",
            render_mode="human",
            robot_uids="piper",
            sensor_configs=dict(shader_pack="rt-fast"),
            human_render_camera_configs=dict(shader_pack="rt-fast"),
            viewer_camera_configs=dict(shader_pack="rt-fast"),
            enable_shadow=True,
        )
        self.viewer = self.env
        obs, _ = self.env.reset(options=dict(reconfigure=True))
        self.scene = self.env.unwrapped.scene
        
        # 2. 加载机器人
        self.robot = self.env.unwrapped.agent.robot
        self.active_joints = self.robot.get_active_joints()
        
        # 3. 初始化Planner
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.active_joints]
        self.planner = mplib.Planner(
            urdf="/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/piper/piper.urdf",
            srdf="/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/piper/piper.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="gripper_base",
            joint_vel_limits=np.ones(6),
            joint_acc_limits=np.ones(6)
        )
        
        # 4. 设置初始关节位置
        init_qpos = np.zeros(self.robot.dof)
        self.robot.set_qpos(init_qpos)

    # def set_joint_positions(self, target_qpos, smoothly=False, steps=50):
    #     """平滑设置关节位置"""
    #     #该方法无法正确计算力度，会把物块抓飞
    #     current_qpos = self.robot.get_qpos()[0].numpy().squeeze()
        
    #     if smoothly:
    #         # 插值轨迹
    #         for i in range(steps):
    #             alpha = i / (steps - 1)
    #             qpos = current_qpos * (1 - alpha) + target_qpos * alpha
                
    #             # 设置关节位置
    #             self.robot.set_qpos(qpos)
    #             self.scene.step()
                
    #             # 更新渲染
    #             self.scene.update_render()
    #             self.viewer.render()
    #     else:
    #         for i in range(steps):
    #             self.robot.set_qpos(target_qpos)
    #             self.scene.step()
    #             self.scene.update_render()
    #             self.viewer.render()
        
    def set_joint_positions(self, target_qpos, steps=30):
        """平滑设置关节位置"""
        #该方法借鉴follow_path的逻辑，不直接设置关节角，而是设置关节驱动力
        current_qpos = self.robot.get_qpos()[0].numpy().squeeze()
        
        # 插值轨迹
        for i in range(steps):
            alpha = i / (steps - 1)
            qpos = current_qpos * (1 - alpha) + target_qpos * alpha
            
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            # 设置活动关节的驱动速度和驱动位置
            for j in range(len(self.planner.move_group_joint_indices)):
                self.active_joints[j].set_drive_target(qpos[j])
                self.active_joints[j].set_drive_velocity_target(
                    0.01
                )
                # 设置关节位置
                self.robot.set_qpos(qpos)
                self.scene.step()
                
                # 更新渲染
                self.scene.update_render()
                self.viewer.render()


    
    def open_gripper(self):
        self.active_joints[-2].set_drive_target(0.4)    #joint7
        self.active_joints[-1].set_drive_target(-0.4)    #joint8
        for i in range(100): 
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def close_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0)
        for i in range(100):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()    

    def solve_ik(self, pose):
        """求解逆运动学"""
        current_qpos = self.robot.get_qpos()[0].numpy().squeeze()
        
        # 调用IK求解器
        status, q_goals = self.planner.IK(
            pose,
            current_qpos,
            n_init_qpos=20,
            threshold=1e-3,
            verbose=True
        )
        
        if status == "Success" and q_goals is not None:
            return q_goals[0]  # 返回第一个解
        else:
            print(f"IK求解失败! 原因: {status}")
            return None

    def robot_to_world(self, local_pose: list, root_pose: list):
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

    def world_to_robot(self, world_pose: list, root_pose: list):
        """
        世界坐标系 -> 机器人坐标系
        world_pose: [x, y, z, w, qx, qy, qz] (wxyz)
        root_pose:  [x, y, z, w, qx, qy, qz] (wxyz)
        返回: [x, y, z, w, qx, qy, qz] (wxyz)
        """  
        # 位置部分 减去基坐标
        # print("world_pose:", world_pose)  
        # print("root_pose:", root_pose)    
        local_pos = (np.array(world_pose[:3]) - np.array(root_pose[:3])).tolist() 
        # print("local_pos:", local_pos)  # [0.671142472, -5.62630369e-08, 0.213193089]

        # 姿态转换（使用四元数乘法）
        from scipy.spatial.transform import Rotation
        r_world = Rotation.from_quat([world_pose[4], world_pose[5], world_pose[6], world_pose[3]])  # xyzw
        r_root = Rotation.from_quat([root_pose[4], root_pose[5], root_pose[6], root_pose[3]])  # xyzw
        r_local = r_root.inv() * r_world
        local_quat_xyzw = r_local.as_quat()
        local_quat_wxyz = [local_quat_xyzw[3], local_quat_xyzw[0], local_quat_xyzw[1], local_quat_xyzw[2]]

        return local_pos + local_quat_wxyz  #拼接这两个list对象

    def move_to_pose(self, pose_robot):
        # 给出的位姿应基于机器人坐标系
        pos = pose_robot[:3]
        quat = pose_robot[3:]
        mplib_pose = mplib.Pose(pos, quat)
        
        # 求解IK
        target_qpos = self.solve_ik(mplib_pose)
        if target_qpos is None:
            return False
        
        # 设置关节位置
        self.set_joint_positions(target_qpos)
        return True


    def demo(self):
        """直接从test_mplib.py迁移的demo函数"""
        # 获取基座位姿
        root_pose = self.robot.get_root_pose().raw_pose[0].tolist()
        print(f"基座位姿：{root_pose}")
        
        # 打印末端执行器初始位姿
        gripper_pose = self.robot.find_link_by_name('gripper_base').pose.raw_pose[0].tolist()
        print(f"末端执行器初始位姿(wxyz)：{gripper_pose}")
        
        # 获取立方体位置
        cube_pose = self.env.get_wrapper_attr('obj').pose.raw_pose[0].tolist()
        print(f"物品坐标位于：{cube_pose[:3]}")
        
        # 1. 移动到物块上方
        cube_pos_top = cube_pose[:3].copy()
        cube_pos_top[2] += 0.15
        pose1_world = cube_pos_top + [0, 0, 1, 0]  # 姿态：垂直向下
        
        # 转换为机器人坐标系
        pose1_robot = self.world_to_robot(pose1_world, root_pose)
        print("\n1. 移动到物块上方")
        self.move_to_pose(pose1_robot)
        
        # 2. 打开夹爪
        print("\n2. 打开夹爪")
        self.open_gripper()
        
        # 3. 向下移动到物块位置
        pose2_robot = pose1_robot.copy()
        pose2_robot[2] -= 0.1  # 向下移动10cm
        print("\n3. 向下移动到物块位置")
        self.move_to_pose(pose2_robot)
        
        # 4. 关闭夹爪
        print("\n4. 关闭夹爪")
        self.close_gripper()
        
        # 5. 提起物块
        pose3_robot = pose2_robot.copy()
        pose3_robot[2] += 0.12  # 向上移动12cm
        print("\n5. 提起物块")
        self.move_to_pose(pose3_robot)
        
        # 6. 移动到放置位置
        pose4_robot = pose3_robot.copy()
        pose4_robot[0] -= 0.2  # 向后移动20cm
        print("\n6. 移动到放置位置")
        self.move_to_pose(pose4_robot)
        
        # 7. 放下物块
        pose5_robot = pose4_robot.copy()
        pose5_robot[2] -= 0.08  # 向下移动8cm
        print("\n7. 放下物块")
        self.move_to_pose(pose5_robot)
        
        # 8. 打开夹爪
        print("\n8. 打开夹爪")
        self.open_gripper()
        
        # 9. 提起夹爪
        pose6_robot = pose5_robot.copy()
        pose6_robot[2] += 0.08  # 向上移动8cm
        print("\n9. 提起夹爪")
        self.move_to_pose(pose6_robot)
        
        print("\n抓取放置演示完成")

# 备用：可进行键盘控制
    def test_move_to_pose(self):
        """
        实时从键盘读取 x y z，驱动机械臂到达（垂直向下朝向）
        输入格式：x y z   （空格分隔，回车结束）
        输入 q 回车退出
        """

        # 固定朝向：垂直向下（wxyz）
        FIXED_QUAT = [0, 0, 1, 0]

        print("实时键盘控制模式：")
        print("  输入 x y z 回车 → 机械臂末端移动到该坐标")
        print("  输入 q 回车   → 退出")
        print("示例：0.05 0.10 0.25\n")

        while True:
            cmd = input(">>> ").strip()
            if cmd.lower() == "q":
                print("退出键盘控制")
                break

            try:
                x, y, z = map(float, cmd.split())
            except ValueError:
                print("格式错误，请输入 3 个数字，空格分隔")
                continue

            # 构造世界坐标系目标位姿
            world_pose = [x, y, z] + FIXED_QUAT
            robot_pose = self.world_to_robot(world_pose)  # 转机器人坐标系
            pos, quat = robot_pose[:3], robot_pose[3:]
            goal = mplib.Pose(pos, quat)

            # 调用 IK + 规划 + 执行
            target_qpos = self.solve_ik(goal)
            if target_qpos is None:
                print("IK 无解，请换个点")
                continue

            print(f"正在前往 [{x:.3f}, {y:.3f}, {z:.3f}] ...")
            self.set_joint_positions(target_qpos)
            print("到达！")

if __name__ == "__main__":
    demo = ManiSkillPiperDemo()
    demo.demo()