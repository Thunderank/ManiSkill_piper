from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import dual_Piper
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

@register_env("CobotPickAndPlaceCube-v1", max_episode_steps=50)
class CobotPickAndPlaceCubeEnv(BaseEnv):
    """
    **Task Description:**
    双手抓取任务，需要同时抓取蓝色和红色方块，并将它们放置到对应颜色的盒子上。
    成功条件包括同时抓取两个方块、抬起方块、将方块放置到对应盒子上，以及确保盒子放在桌子上。
    
    **Randomizations:**
    - 蓝色方块、红色方块、蓝色盒子和红色盒子的初始位置在桌面上随机化
    - 盒子位置固定在桌面上，方块位置在盒子附近随机化
    
    **Success Conditions:**
    - 蓝色方块放置在蓝色盒子上
    - 红色方块放置在红色盒子上
    - 盒子保持放置在桌面上
    - 两个方块都被抬起离开桌面
    """

    SUPPORTED_ROBOTS = ["dual_piper"]
    agent: Union[dual_Piper]

    # 物体尺寸定义 借鉴place_sphere.py
    cube_half_size = 0.02   #方块边长/2
    inner_side_half_len = 0.05  #bin内部半边长  增大了该值以扩大bin的大小
    short_side_half_size = 0.0025  # bin四周壁的半长(组成bin的各小方块的最短边)
    block_half_size = [ #bin底部方块长宽高的半尺寸
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]  # The bottom block of the bin, which is larger: The list represents the half length of the block along the [x, y, z] axis respectively.
    edge_block_half_size = [    #bin四周壁方块长宽高的半尺寸
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + 0.04 ,    #可增大z值以增大bin边壁的高度
    ]  # The edge block of the bin, which is smaller. The representations are similar to the above one

    def __init__(self, *args, robot_uids="dual_piper", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    #改动： 1.修改全局相机视角
    #      2.绑定与URDF中对应的相机名称
    #       3.该部分设定会在agent中被覆盖
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0, 0, 0.2], target=[-0.1, 0, 0])
        return [
            CameraConfig(
                "left_wrist_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),  # 相对于mount的pose
                width=640,
                height=480,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.agent.robot.links_map["left_arm_camera"] #该语句会在maniskill中挂载相机视角
            ),
            CameraConfig(
                "right_wrist_camera",
                pose=sapien.Pose(),  # 将使用机器人坐标系
                width=640,
                height=480,
                fov=np.pi / 2,  #相机视场(Field of View)张角，用来描述拍摄的范围
                near=0.01,
                far=100,
                mount=self.agent.robot.links_map["right_arm_camera"] #该语句会在maniskill中挂载相机视角
            ),
            CameraConfig(
                "global_camera",
                pose=pose,
                width=640,
                height=480,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    #human渲染模式GUI中初始观察视角
    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        pose = sapien_utils.look_at([0, 0, 1], [-0.5, 0.0, 0])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    #改动：1.添加注册color 2.添加name字段
    def _build_bin(self, color):
        builder = self.scene.create_actor_builder()

        # init the locations of the basic blocks
        dx = self.block_half_size[1] - self.block_half_size[0]
        dy = self.block_half_size[1] - self.block_half_size[0]
        dz = self.edge_block_half_size[2] + self.block_half_size[0]

        # build the bin bottom and edge blocks
        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
            self.edge_block_half_size,
            self.edge_block_half_size,
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
        ]
        if color == "red":
            color_value = np.array([255, 0, 0, 255]) / 255  # 红色
            obj_name = "red_bin"
        elif color == "blue":
            color_value = np.array([0, 0, 255, 255]) / 255  # 蓝色 RGBA值
            obj_name = "blue_bin"

        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            # builder.add_box_visual(pose, half_size)
            builder.add_box_visual(pose, half_size,material=sapien.render.RenderMaterial(base_color=color_value)) #添加颜色，需指定RGBA值如[1, 0, 0, 1]代表红色

        # build the kinematic bin
        return builder.build_kinematic(name=obj_name)

    def _load_agent(self, options: dict):
        # super()._load_agent(options, sapien.Pose(p=[-0.3, 0, 0]))# 设置机器人的初始位置
        super()._load_agent(options, sapien.Pose(p=[-0.2, 0, 0.2]))# 升高一点以扩大抓取范围


    def _load_scene(self, options: dict):
        # 创建桌子和机械臂。实际的初始位姿会在_initialize_episode中设置。cube在一定范围内随机，bin固定
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # 创建蓝色方块
        self.blue_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([0, 0, 255, 255]) / 255,  # 蓝色 RGBA值
            name="blue_cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[-0.3, -0.5, 1]),
        )

        # 创建红色方块
        self.red_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([255, 0, 0, 255]) / 255,  # 红色
            name="red_cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[-0.3, 0.5, 1]),
        )

        # 尝试用actors.build_box创建盒子  -- 无效，maniskill中的box是纸盒、宝箱、披萨盒这些

        # 用自定义函数u创建盒子
        self.blue_bin = self._build_bin(color = "blue")
        self.red_bin = self._build_bin(color = "red")

    # 设定每个episode的cube和bin的初始位姿
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Use torch.device context for tensor creation
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            #对cube
            #生成env个数xyz随机数
            #重新生成红色方块的位姿 左红右蓝
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = torch.rand((b, 1)) * 0.05 - 0.7       #x
            xyz[..., 1] = torch.rand((b, 1)) * 0.1 - 0.25       #y 
            xyz[..., 2] = self.cube_half_size                   #设定中心点z轴高度
            q = [1, 0, 0, 0]
            red_cube_pose = Pose.create_from_pq(p=xyz, q=q)

            #重新生成蓝色方块的位姿
            xyz[..., 0] = torch.rand((b, 1)) * 0.05 - 0.7  
            xyz[..., 1] = torch.rand((b, 1)) * 0.1 + 0.25
            blue_cube_pose = Pose.create_from_pq(p=xyz, q=q)
            self.blue_cube.set_pose(red_cube_pose)
            self.red_cube.set_pose(blue_cube_pose)

            #对红色bin
            xyz[..., 0] = -0.55   #x
            xyz[..., 1] = 0.25   #y
            xyz[..., 2] = self.block_half_size[0]   #z

            red_bin_pose = Pose.create_from_pq(p=xyz, q=q)
            xyz[..., 1] *= -1   #对蓝色bin的y轴值取反，为负值
            blue_bin_pose = Pose.create_from_pq(p=xyz, q=q)
            self.red_bin.set_pose(red_bin_pose)
            self.blue_bin.set_pose(blue_bin_pose)


    # 设置任务判定条件，后续会以info变量的形式传递信息
    def evaluate(self):
        # 获取方块和bin的位姿
        blue_cube_pos = self.blue_cube.pose.p  # (b, 3)
        red_cube_pos = self.red_cube.pose.p
        blue_bin_pos = self.blue_bin.pose.p
        red_bin_pos = self.red_bin.pose.p

        # 判定参数
        bin_xy_radius = 0.03  # bin xy判定半径
        bin_z_thresh = 0.01   # z高度阈值

        # 蓝方块是否在蓝bin内
        blue_in_bin_xy = torch.linalg.norm(blue_cube_pos[..., :2] - blue_bin_pos[..., :2], dim=-1) < bin_xy_radius
        blue_in_bin_z = (blue_cube_pos[..., 2] > blue_bin_pos[..., 2] + self.cube_half_size - bin_z_thresh) & \
                        (blue_cube_pos[..., 2] < blue_bin_pos[..., 2] + self.cube_half_size + bin_z_thresh)
        blue_in_bin = blue_in_bin_xy & blue_in_bin_z

        # 红方块是否在红bin内
        red_in_bin_xy = torch.linalg.norm(red_cube_pos[..., :2] - red_bin_pos[..., :2], dim=-1) < bin_xy_radius
        red_in_bin_z = (red_cube_pos[..., 2] > red_bin_pos[..., 2] + self.cube_half_size - bin_z_thresh) & \
                       (red_cube_pos[..., 2] < red_bin_pos[..., 2] + self.cube_half_size + bin_z_thresh)
        red_in_bin = red_in_bin_xy & red_in_bin_z

        # 是否抬起（可选）
        blue_lifted = blue_cube_pos[..., 2] > self.cube_half_size + 0.01
        red_lifted = red_cube_pos[..., 2] > self.cube_half_size + 0.01

        is_success = blue_in_bin & red_in_bin

        return {
            "success": is_success,
            "blue_in_bin": blue_in_bin,
            "red_in_bin": red_in_bin,
            "blue_lifted": blue_lifted,
            "red_lifted": red_lifted,
        }

    #TODO:大改
    def _get_obs_extra(self, info: Dict):
        # print(f"task中get_obs_extra获取的info：{info}") 
        obs = dict(
            left_tcp_pose=self.agent.left_tcp.pose.raw_pose,
            right_tcp_pose=self.agent.right_tcp.pose.raw_pose,
            blue_cube_pose=self.blue_cube.pose.raw_pose,
            red_cube_pose=self.red_cube.pose.raw_pose,
            blue_bin_pose=self.blue_bin.pose.raw_pose,
            # red_bin_pose=self.red_bin.pose.raw_pose,
            # # 新增：夹爪开合状态
            # left_gripper_open=self.agent.left_tcp.gripper_open,
            # right_gripper_open=self.agent.right_tcp.gripper_open,
            # # 新增：是否接触桌面/盒子（可选，需在info中有这些key）
            # blue_cube_touch_table=info.get("blue_cube_touch_table", None),
            # red_cube_touch_table=info.get("red_cube_touch_table", None),
            # blue_cube_touch_bin=info.get("blue_cube_touch_bin", None),
            # red_cube_touch_bin=info.get("red_cube_touch_bin", None),
        )
        # 可选：夹爪开合、bin内是否有方块等
        return obs


    # 改写sapien_env.py中获取reward的子方法，使用详细、连续值定义reward
    # 启动指令中添加参数 -o "pointcloud" --reward_mode "dense"
    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # print(f"task中compute_dense_reward获取的obs：{obs},action：{action},info：{info}")
        # 末端到方块距离
        left_tcp_to_blue = torch.linalg.norm(self.agent.left_tcp.pose.p - self.blue_cube.pose.p, dim=-1)
        right_tcp_to_red = torch.linalg.norm(self.agent.right_tcp.pose.p - self.red_cube.pose.p, dim=-1)
        reach_reward = 2.0 - torch.tanh(5 * left_tcp_to_blue) - torch.tanh(5 * right_tcp_to_red)

        # 抓取奖励（可根据夹爪状态或距离判定）
        grasp_reward = ((left_tcp_to_blue < 0.04) & (right_tcp_to_red < 0.04)).float()

        # 抬起奖励
        lift_reward = (info["blue_lifted"] & info["red_lifted"]).float()

        # 放置奖励
        place_reward = (info["blue_in_bin"] & info["red_in_bin"]).float()

        # 成功奖励
        success_reward = info["success"].float() * 2.0

        reward = reach_reward + grasp_reward + lift_reward + place_reward + success_reward
        return reward


