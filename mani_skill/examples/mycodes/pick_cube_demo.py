'''
python -m mani_skill.examples.mycodes.test_mplib --render-mode="human" --shader="rt-fast" -c "pd_joint_delta_pos" -e "PushCube-v1" -r "piper"
'''

import time
import random
import gymnasium as gym
import numpy as np
import sapien

import mplib 

from scipy.spatial.transform import Rotation

# 注册参数
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

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_delta_pos"
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

# demo类
class ManiSkillPiperDemo:
    def __init__(self, args):
        # 1. 启动ManiSkill环境，获取scene和robot
        self.env = gym.make(
            args.env_id,    # PushCube-v1
            obs_mode="none",
            control_mode=args.control_mode,
            render_mode="human",
            robot_uids=args.robot_uids,
            sensor_configs=dict(shader_pack="rt-fast"),
            human_render_camera_configs=dict(shader_pack="rt-fast"),
            viewer_camera_configs=dict(shader_pack="rt-fast"),
            enable_shadow=True,
        )
        self.viewer = self.env
        obs, _ = self.env.reset(options=dict(reconfigure=True, robot_uids="piper"))
        self.scene = self.env.unwrapped.scene
        self.scene.set_timestep(1 / 240)
        self._load_robot()  # 加载机器人

        # 2. 初始化mplib Planner
        self._setup_planner()

        # 3. 设定初始关节
        init_qpos = np.zeros(self.robot.dof)
        self.robot.set_qpos(init_qpos)

    def _load_robot(self, **kwargs):
        """
        从Sapien场景加载机器人
        """
        self.robot = self.env.unwrapped.agent.robot
        
        # 设置机械臂在桌面的位置
        self.robot.set_pose(sapien.Pose(p=[-0.3, 0, 0]))

        self.active_joints = self.robot.get_active_joints()

    def _setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        self.planner = mplib.Planner(
            urdf="/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/piper/piper.urdf",
            srdf="/home/robot/miniconda3/envs/piper/lib/python3.10/site-packages/mani_skill/assets/robots/piper/piper.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="gripper_base",
            joint_vel_limits=np.ones(6),
            joint_acc_limits=np.ones(6)
        )

    def follow_path(self, result, headless=False):
        """跟随规划器生成的路径"""
        n_step = result["position"].shape[0]
        print(f"follow_path中得到的plan给出的总时间步数:{n_step} 总用时：{result['duration']}")

        for i in range(n_step):
            qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            for j in range(len(self.planner.move_group_joint_indices)):
                self.active_joints[j].set_drive_target(result["position"][i][j])
                self.active_joints[j].set_drive_velocity_target(result["velocity"][i][j])
            self.scene.step()
            if i % 4 == 0 and not headless:
                self.scene.update_render()
                self.viewer.render()
        
        cur_pose = self.robot.find_link_by_name('gripper_base').pose.raw_pose[0].tolist()
        print("运动结束，已到达点: [" + ", ".join(f"{x:.2f}" for x in cur_pose) + "]\n")

    def set_gripper(self, pos):
        """控制夹爪位置"""
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(pos)
        for i in range(100):
            qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def move_to_pose(self, pose, with_screw=True, headless=False):
        world_pose = self.robot_to_world(pose, self.robot.get_root_pose().raw_pose[0].tolist())
        print("\n开始规划，目标点: [" + ", ".join(f"{x:.2f}" for x in world_pose) + "]")
        if with_screw:
            return self._move_to_pose_with_screw(pose, headless)
        else:
            return self._move_to_pose_with_RRTConnect(pose, headless)

    def _move_to_pose_with_retry(self, pose, with_screw=True, max_retries=5, headless=False):
        """尝试移动到目标位姿，失败时在邻域内随机扰动重试"""
        result = self.move_to_pose(pose, with_screw)
        if result == 0:
            return 0
        
        original_pose = np.array(pose)
        pos = original_pose[:3]
        quat = original_pose[3:]

        for attempt in range(max_retries):
            print(f"\n第 {attempt+1} 次重试，原始目标: {[f'{x:.3f}' for x in original_pose]}")
            new_pos = pos + np.random.uniform(-0.01, 0.01, size=3)
            orig_rot = Rotation.from_quat(quat)
            random_axis = np.random.randn(3)
            random_axis /= np.linalg.norm(random_axis)
            random_angle = np.random.uniform(-0.5, 0.5)
            perturb_rot = Rotation.from_rotvec(random_angle * np.pi/180 * random_axis)
            new_rot = perturb_rot * orig_rot
            new_quat = new_rot.as_quat()
            new_pose = np.concatenate([new_pos, new_quat]).tolist()
            result = self.move_to_pose(new_pose, with_screw)
            if result == 0:
                return 0
        
        print(f"\n⚠️ 警告: 经过 {max_retries} 次重试仍失败")
        return result

    def _move_to_pose_with_RRTConnect(self, pose, headless=False):
        print("使用RRTConnect motion规划")
        result = self.planner.plan_pose(
            mplib.Pose(pose[:3], pose[3:]), 
            self.robot.get_qpos()[0].numpy().squeeze(), 
            time_step=1 / 140
        )
        print("RRTConnect规划结果为：", result["status"])
        if result["status"] != "Success":
            return -1
        self.follow_path(result, headless)
        return 0
    
    def _move_to_pose_with_screw(self, pose, headless=False):
        result = self.planner.plan_screw(
            mplib.Pose(pose[:3], pose[3:]), 
            self.robot.get_qpos()[0].numpy().squeeze(), 
            time_step=1 / 140,
        )
        if result["status"] == "Success":
            print("使用screw motion规划")
            self.follow_path(result, headless)
            return 0
        else:
            print("screw motion规划失败，回退到RRTConnect规划")
            return self._move_to_pose_with_RRTConnect(pose, headless)

    def open_gripper(self):
        self.set_gripper(0.4)
        print("已打开夹爪\n")

    def close_gripper(self):
        self.set_gripper(0)
        print("已关闭夹爪\n")

    def robot_to_world(self, local_pose: list, root_pose: list):
        """机器人坐标系 -> 世界坐标系"""
        world_pos = (np.array(local_pose[:3]) + np.array(root_pose[:3])).tolist()
        q_local = [local_pose[4], local_pose[5], local_pose[6], local_pose[3]]
        q_root = [root_pose[4], root_pose[5], root_pose[6], root_pose[3]]
        r_local = Rotation.from_quat(q_local)
        r_root = Rotation.from_quat(q_root)
        r_world = r_root * r_local
        q_world_xyzw = r_world.as_quat()
        q_world_wxyz = [q_world_xyzw[3], q_world_xyzw[0], q_world_xyzw[1], q_world_xyzw[2]]
        return world_pos + q_world_wxyz

    def world_to_robot(self, world_pose: list, root_pose: list):
        """世界坐标系 -> 机器人坐标系"""
        local_pos = (np.array(world_pose[:3]) - np.array(root_pose[:3])).tolist()
        r_world = Rotation.from_quat([world_pose[4], world_pose[5], world_pose[6], world_pose[3]])
        r_root = Rotation.from_quat([root_pose[4], root_pose[5], root_pose[6], root_pose[3]])
        r_local = r_root.inv() * r_world
        local_quat_xyzw = r_local.as_quat()
        local_quat_wxyz = [local_quat_xyzw[3], local_quat_xyzw[0], local_quat_xyzw[1], local_quat_xyzw[2]]
        return local_pos + local_quat_wxyz

    def pick_cube_demo(self, with_screw=True):
        self.robot.set_pose(sapien.Pose(p=[-0.35, 0, 0]))
        root_pose = self.robot.get_root_pose().raw_pose[0].tolist()
        cube_pose = self.env.get_wrapper_attr('obj').pose.raw_pose[0].tolist()
        print(f"物品坐标位于：{cube_pose[:3]}\n")

        cube_pos_top = cube_pose[:3].copy()
        cube_pos_top[2] += 0.16
        pose1_world = cube_pos_top + [0, 0, 1, 0]
        pose1_robot = self.world_to_robot(pose1_world, root_pose)
        pos = pose1_robot[:3]
        qual = pose1_robot[3:]
        pose_1 = pos + qual
        print("尝试到达物块上方")
        self.move_to_pose(pose_1, with_screw)
        self.open_gripper()

        pos[2] -= 0.08
        pose_2 = pos + qual
        print("尝试下放夹爪到物块位置")
        self.move_to_pose(pose_2, with_screw)
        self.close_gripper()

        pos[2] += 0.12
        pose_3 = pos + qual
        print("尝试提起物块")
        self.move_to_pose(pose_3, with_screw)

        pos[0] -= 0.2
        pose_4 = pos + qual
        print("尝试收回夹爪")
        self.move_to_pose(pose_4, with_screw)

        pos[2] -= 0.08
        pose_5 = pos + qual
        print("尝试放下物块")
        self.move_to_pose(pose_5, with_screw)
        self.open_gripper()

        pos[2] += 0.08
        pose_6 = pos + qual
        print("尝试抬起夹爪")
        self.move_to_pose(pose_6, with_screw)

    def test_MP_SR(self, num_tests=10000):
        """测试运动规划方法成功率"""
        test_cases = [
            {"method": "Screw", "with_screw": True},
            {"method": "RRTConnect", "with_screw": False}
        ]
        results = {}
        
        for case in test_cases:
            method_name = case["method"]
            success_count = 0
            print(f"\n开始测试 {method_name} 方法 ({num_tests}次)...")
            
            for i in range(num_tests):
                self.env.reset()
                root_pose = self.robot.get_root_pose().raw_pose[0].tolist()
                cube_pose = self.env.get_wrapper_attr('obj').pose.raw_pose[0].tolist()
                cube_pos_top = cube_pose[:3].copy()
                cube_pos_top[2] += 0.16
                pose1_world = cube_pos_top + [0, 0, 1, 0]
                pose1_robot = self.world_to_robot(pose1_world, root_pose)
                result = self.move_to_pose(pose1_robot, with_screw=case["with_screw"], headless=True)
                if result == 0:
                    success_count += 1
                if (i + 1) % max(1, num_tests//100) == 0:
                    print(f"进度: {i+1}/{num_tests} | 成功率: {success_count/(i+1):.2%}")
            
            results[method_name] = {
                "success": success_count,
                "failure": num_tests - success_count,
                "success_rate": success_count / num_tests
            }
        
        print("\n" + "="*50)
        print("运动规划方法成功率统计:")
        print("="*50)
        for method, data in results.items():
            print(f"{method}方法:")
            print(f"  成功次数: {data['success']}/{num_tests}")
            print(f"  失败次数: {data['failure']}/{num_tests}")
            print(f"  成功率: {data['success_rate']:.4f} ({data['success_rate']:.2%})")
            print("-"*50)
        return results

    def plot_reachable_space(self, num_samples=1000, max_retries=3, with_screw=True):
        """
        通过蒙特卡洛采样绘制固定姿态下的可达空间三维图
        Args:
            num_samples: 采样点数量
            max_retries: 每个点的最大重试次数
            with_screw: 是否使用screw motion规划
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D      
        
        # 获取机器人基座位姿
        root_pose = self.robot.get_root_pose().raw_pose[0].tolist()
        
        # 定义采样范围（TODO：根据机器人工作空间调整，在仿真器中调整姿态，找到大致边界）
        x_range = [-0.258, 0.09]    # x方向范围
        y_range = [-0.285, 0.285]   # y方向范围
        z_range = [0.136, 0.283]     # z方向范围
        
        # 存储采样结果
        reachable_points = []
        unreachable_points = []
        
        # 固定姿态（四元数 wxyz 格式）
        fixed_quat = [0, 0, 1, 0]  # 对应绕Y轴旋转180度
        
        print(f"开始可达空间采样，共{num_samples}个点...")
        start_time = time.time()
        
        for i in range(num_samples):
            # 随机采样一个位置
            pos = [
                random.uniform(x_range[0], x_range[1]),
                random.uniform(y_range[0], y_range[1]),
                random.uniform(z_range[0], z_range[1])
            ]
            
            # 构建目标位姿（位置+固定姿态）
            target_pose = pos + fixed_quat
            target_pose_robot = self.world_to_robot(target_pose, root_pose)
            
            # 尝试规划到这个位姿
            result = self.move_to_pose(
                target_pose_robot, 
                with_screw=with_screw, 
                headless=True
            )
            # print(f"result为：{result}")
            
            # 记录结果
            if result == 0:
                reachable_points.append(pos)
            else:
                unreachable_points.append(pos)
            
            # 重置机器人到初始位置
            self.robot.set_qpos(np.zeros(self.robot.dof))
            # 物理模拟几步确保稳定
            for _ in range(10):
                self.scene.step()
            
            # 打印进度
            if (i + 1) % max(1, num_samples // 10) == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"进度: {i+1}/{num_samples} | "
                      f"可达点: {len(reachable_points)} | "
                      f"不可达点: {len(unreachable_points)} | "
                      f"速度: {rate:.1f} 点/秒")
        
        # ---------- 1. 可达区域3D图 ----------
        plt.rcParams['font.family'] = ['WenQuanYi Zen Hei']   # Linux 设定中文字体 sudo apt-get install ttf-wqy-zenhei
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制可达点（绿色）
        if reachable_points:
            rp = np.array(reachable_points)
            ax.scatter(rp[:, 0], rp[:, 1], rp[:, 2], 
                      c='g', marker='o', alpha=0.6, label='可达')
        
        # 绘制不可达点（红色）
        if unreachable_points:
            up = np.array(unreachable_points)
            ax.scatter(up[:, 0], up[:, 1], up[:, 2], 
                      c='r', marker='x', alpha=0.3, label='不可达')
        
        # 设置坐标轴标签
        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')
        ax.set_title(f'固定姿态下的可达空间（姿态: [0,0,1,0]）\n采样点: {num_samples} | 可达率: {len(reachable_points)/num_samples:.2%}')
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        
        # 添加图例和网格
        ax.legend()
        ax.grid(True)
        
        # 添加机器人基座标记
        base_pos = root_pose[:3]
        ax.scatter(base_pos[0], base_pos[1], base_pos[2], 
                  c='b', marker='*', s=100, label='机器人基座')

        print(f"采样完成！可达点: {len(reachable_points)} | 不可达点: {len(unreachable_points)}") 
        if reachable_points:
            rp = np.array(reachable_points)
            print("可行 xyz 范围：")
            print(f"  x ∈ [{rp[:,0].min():.3f}, {rp[:,0].max():.3f}]")
            print(f"  y ∈ [{rp[:,1].min():.3f}, {rp[:,1].max():.3f}]")
            print(f"  z ∈ [{rp[:,2].min():.3f}, {rp[:,2].max():.3f}]")

        #显示可达区域采样结果3D图
        plt.tight_layout()
        plt.show()

        # ---------- 2. 二维投影直方图 ----------
        fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
        proj = [(0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ')]
        for ax, (i, j, name) in zip(axes, proj):
            if reachable_points:
                ax.hist2d(rp[:, i], rp[:, j], bins=50, cmap='Greens')
                ax.set_xlabel(f'{name[0]} (m)')
                ax.set_ylabel(f'{name[1]} (m)')
                ax.set_title(f'{name} 平面投影')
            ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show(block=False)

        # ---------- 3. 3D 凸包切片（可选）----------
        # 需要 convex-hull，pip install scipy
        from scipy.spatial import ConvexHull
        if reachable_points and len(reachable_points) >= 4:
            hull = ConvexHull(rp)
            fig3 = plt.figure(figsize=(8, 8))
            ax3 = fig3.add_subplot(111, projection='3d')
            # 绘制凸包面片
            for simplex in hull.simplices:
                ax3.plot_trisurf(rp[simplex, 0], rp[simplex, 1], rp[simplex, 2],
                                alpha=0.25, color='g')
            ax3.scatter(rp[:, 0], rp[:, 1], rp[:, 2], s=1, c='g')
            ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
            ax3.set_title('可行域凸包')
            plt.show()
                

            return reachable_points, unreachable_points

if __name__ == "__main__":
    args = tyro.cli(Args)
    demo = ManiSkillPiperDemo(args)
    demo.pick_cube_demo()
    # demo.test_MP_SR()
    # demo.plot_reachable_space(num_samples=1000)