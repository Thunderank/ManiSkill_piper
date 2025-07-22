'''
python -m mani_skill.examples.test_planning_continuous --render-mode="human" --shader="rt-fast" -c "pd_joint_delta_pos" -e "PushCube-v1" -r "piper"

或python -m mani_skill.examples.my_piper_demo_ctrl_action_ee --render-mode="human" --shader="rt-fast" -c "pd_joint_delta_pos" -e "PushCube-v1" -r "piper
'''

import time
import numpy as np
import sapien
import gymnasium as gym
import mplib
from scipy.spatial.transform import Rotation
import threading
import queue
import sys

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
        
        # 5. 初始化控制变量
        self.target_pose = None
        self.target_qpos = None
        self.command_queue = queue.Queue()
        self.running = True
        self.gripper_open = True
        self.last_log_time = time.time()

    def set_joint_positions(self, target_qpos, steps=10):
        """平滑设置关节位置"""
        current_qpos = self.robot.get_qpos()[0].numpy().squeeze()
        
        # 插值轨迹
        for i in range(steps):
            alpha = i / (steps - 1)
            qpos = current_qpos * (1 - alpha) + target_qpos * alpha
            
            # 设置关节位置
            self.robot.set_qpos(qpos)
            self.scene.step()
            
            # 更新渲染
            self.scene.update_render()
            self.viewer.render()
    
    def open_gripper(self):
        """打开夹爪"""
        self.gripper_open = True
        qpos = self.robot.get_qpos()[0].numpy().squeeze()
        qpos[-2:] = [0.035, -0.035]  # piper夹爪张开位置
        self.set_joint_positions(qpos, steps=50)
    
    def close_gripper(self):
        """关闭夹爪"""
        self.gripper_open = False
        qpos = self.robot.get_qpos()[0].numpy().squeeze()
        qpos[-2:] = [0.0, 0.0]  # piper夹爪闭合位置
        self.set_joint_positions(qpos, steps=50)

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

    def world_to_robot(self, world_pose):
        """世界坐标系 -> 机器人坐标系"""
        root_pose = self.robot.get_root_pose().raw_pose[0].tolist()
        
        # 位置转换
        local_pos = np.array(world_pose[:3]) - np.array(root_pose[:3])
        
        # 姿态转换
        r_world = Rotation.from_quat([
            world_pose[4], world_pose[5], world_pose[6], world_pose[3]  # xyzw
        ])
        r_root = Rotation.from_quat([
            root_pose[4], root_pose[5], root_pose[6], root_pose[3]  # xyzw
        ])
        r_local = r_root.inv() * r_world
        local_quat_xyzw = r_local.as_quat()
        local_quat_wxyz = [local_quat_xyzw[3],  # w
                           local_quat_xyzw[0],  # x
                           local_quat_xyzw[1],  # y
                           local_quat_xyzw[2]]  # z
        
        return list(local_pos) + local_quat_wxyz

    def get_end_effector_pose(self):
        """获取末端执行器位姿（世界坐标系）"""
        # 获取夹爪基座的位姿
        gripper_base = self.robot.find_link_by_name('gripper_base')
        pose = gripper_base.pose.raw_pose[0].tolist()
        return pose
    
    def get_joint_positions(self):
        """获取当前关节角度"""
        return self.robot.get_qpos()[0].numpy().squeeze().tolist()
    
    def log_status(self):
        """打印当前状态"""
        current_time = time.time()
        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            
            # 获取末端位姿和关节角
            ee_pose = self.get_end_effector_pose()
            joint_pos = self.get_joint_positions()
            
            # 格式化输出
            print("\n" + "="*80)
            print(f"时间: {time.strftime('%H:%M:%S')}")
            print(f"末端位姿 (世界坐标系):")
            print(f"  位置: [{ee_pose[0]:.4f}, {ee_pose[1]:.4f}, {ee_pose[2]:.4f}]")
            print(f"  朝向: [{ee_pose[3]:.4f}, {ee_pose[4]:.4f}, {ee_pose[5]:.4f}, {ee_pose[6]:.4f}]")
            print(f"关节角度 (弧度):")
            for i, pos in enumerate(joint_pos):
                print(f"  关节 {i}: {pos:.4f}")
            print(f"夹爪状态: {'打开' if self.gripper_open else '关闭'}")
            print("="*80 + "\n")

    def input_handler(self):
        """处理键盘输入的线程函数"""
        print("\n键盘控制指令:")
        print("  x y z - 移动机械臂到指定位置 (世界坐标系)")
        print("  o - 打开夹爪")
        print("  c - 关闭夹爪")
        print("  q - 退出程序")
        print("示例: 0.0 0.0 0.3\n")
        
        while self.running:
            try:
                # 使用非阻塞方式读取输入
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    cmd = sys.stdin.readline().strip()
                    
                    if cmd.lower() == 'q':
                        self.running = False
                        break
                    elif cmd.lower() == 'o':
                        self.command_queue.put(('gripper', 'open'))
                    elif cmd.lower() == 'c':
                        self.command_queue.put(('gripper', 'close'))
                    else:
                        try:
                            parts = cmd.split()
                            if len(parts) == 3:
                                x, y, z = map(float, parts)
                                # 固定朝向：垂直向下（wxyz）
                                fixed_quat = [0.737, 0, 0.675, 0]
                                self.command_queue.put(('move', [x, y, z] + fixed_quat))
                            else:
                                print("错误: 请输入三个数字 (x y z)")
                        except ValueError:
                            print("错误: 无效的输入格式")
            except:
                self.running = False
                break

    def continuous_control(self):
        """持续控制主循环"""
        # 启动输入处理线程
        input_thread = threading.Thread(target=self.input_handler, daemon=True)
        input_thread.start()
        
        # 初始位置
        initial_pose = [-0.4, 0.0, 0.4, 0.737, 0, 0.675, 0]
        self.move_to_pose(initial_pose)
        
        # 主控制循环
        while self.running:
            # 处理命令队列
            while not self.command_queue.empty():
                cmd_type, data = self.command_queue.get()
                
                if cmd_type == 'move':
                    success = self.move_to_pose(data)
                    if success:
                        print(f"已到达位置: [{data[0]:.3f}, {data[1]:.3f}, {data[2]:.3f}]")
                    else:
                        print(f"无法到达位置: [{data[0]:.3f}, {data[1]:.3f}, {data[2]:.3f}]")
                elif cmd_type == 'gripper':
                    if data == 'open':
                        self.open_gripper()
                        print("夹爪已打开")
                    elif data == 'close':
                        self.close_gripper()
                        print("夹爪已关闭")
            
            # 步进模拟
            self.scene.step()
            self.scene.update_render()
            self.viewer.render()
            
            # 打印状态日志
            self.log_status()
        
        print("程序退出")

    def move_to_pose(self, pose_world):
        """移动到指定世界坐标位置"""
        # 转换为机器人坐标系
        pose_robot = self.world_to_robot(pose_world)
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

if __name__ == "__main__":
    demo = ManiSkillPiperDemo()
    demo.continuous_control()