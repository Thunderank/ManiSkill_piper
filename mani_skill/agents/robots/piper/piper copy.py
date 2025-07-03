from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array

PIPER_WHEELS_COLLISION_BIT = 30
"""Collision bit of the piper robot wheel links"""
PIPER_BASE_COLLISION_BIT = 31
"""Collision bit of the piper base"""


@register_agent()
class Piper(BaseAgent):
    uid = "piper"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/piper/piper.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            link7=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            link8=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            #设置初始关节位置 基座[0] + 左臂(link-joint 1-6)[1-12] + 夹爪(连接关节、夹爪基座、夹爪1joint,link、夹爪2joint,link)[13,-18]
            qpos=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),   #19维
            # #设置初始关节位置 基座[0,1,2] + 左臂[3-8] + 右臂[9-14] + 夹爪[15,16] + 头部[17]
            # qpos=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        )
    )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="piper_head",
                pose=Pose.create_from_pq([0, 0, 0.1], [1, 0, 0, 0]),
                width=128,
                height=128,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="link6", #设置相机位于夹爪末端
            )
        ]

    def __init__(self, *args, **kwargs):
        # 定义Piper机器人的关节名称
        self.base_joint_names = [
            "base_to_dummy"
        ]
        # self.base_joint_names = [
        #     "root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"
        # ]
        
        # 左臂关节
        self.left_arm_joint_names = [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        ]
        
        # # 右臂关节
        # self.right_arm_joint_names = [
        #     "joint1_2", "joint2_2", "joint3_2", "joint4_2", "joint5_2", "joint6_2"
        # ]
        
        # 夹爪关节
        self.gripper_left_joint_names = ["joint7", "joint8"]
        # self.gripper_right_joint_names = ["joint7_2", "joint8_2"]
        
        # # 头部关节
        # self.head_joint_names = ["head_pan_joint", "head_tilt_joint"]
        
        # 控制器参数
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100
        
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100
        
        # self.head_stiffness = 1e3
        # self.head_damping = 1e2
        # self.head_force_limit = 50
        
        # 末端执行器链接名称
        self.ee_left_link_name = "link7"  # 左臂夹爪1
        self.ee_right_link_name = "link8"  # 左臂夹爪2

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # --------------------------------------------------------------------------
        # 双臂控制器config
        # --------------------------------------------------------------------------
        # 左臂位置控制器
        left_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        
        # # 右臂位置控制器
        # right_arm_pd_joint_pos = PDJointPosControllerConfig(
        #     self.right_arm_joint_names,
        #     None,
        #     None,
        #     self.arm_stiffness,
        #     self.arm_damping,
        #     self.arm_force_limit,
        #     normalize_action=False,
        # )
        
        # --------------------------------------------------------------------------
        # 夹爪控制器
        # --------------------------------------------------------------------------
        # 左夹爪位置控制器
        gripper_left_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_left_joint_names,
            -0.01,  # 最小值
            0.05,    # 最大值
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )
        
        # # 右夹爪位置控制器
        # gripper_right_pd_joint_pos = PDJointPosMimicControllerConfig(
        #     self.gripper_right_joint_names,
        #     -0.01,  # 最小值
        #     0.05,    # 最大值
        #     self.gripper_stiffness,
        #     self.gripper_damping,
        #     self.gripper_force_limit,
        # )
        
        # --------------------------------------------------------------------------
        # 基座控制器
        # --------------------------------------------------------------------------
        # base_pd_joint_vel = PDBaseForwardVelControllerConfig(
        #     self.base_joint_names,
        #     lower=[-1, -3.14],
        #     upper=[1, 3.14],
        #     damping=1000,
        #     force_limit=500,
        # )
        
        # --------------------------------------------------------------------------
        # 头部控制器
        # --------------------------------------------------------------------------
        # head_pd_joint_pos = PDJointPosControllerConfig(
        #     self.head_joint_names,
        #     None,
        #     None,
        #     self.head_stiffness,
        #     self.head_damping,
        #     self.head_force_limit,
        #     normalize_action=False,
        # )

        # 组合所有控制器，创建双臂控制器config
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                left_arm=left_arm_pd_joint_pos,
                # right_arm=right_arm_pd_joint_pos,
                gripper_left=gripper_left_pd_joint_pos,
                # gripper_right=gripper_right_pd_joint_pos,
                # base=base_pd_joint_vel,
                # head=head_pd_joint_pos,
            ),
        )

        # 返回深拷贝的控制器配置，便于修改配置后复原
        return deepcopy(controller_configs)

    def _after_init(self):
        # 左臂末端执行器
        self.left_ee_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_left_link_name
        )
        
        # # 右臂末端执行器
        # self.right_ee_link: Link = sapien_utils.get_obj_by_name(
        #     self.robot.get_links(), self.ee_right_link_name
        # )
        
        # # 基座链接
        # self.base_link: Link = sapien_utils.get_obj_by_name(
        #     self.robot.get_links(), "base_link"
        # )
        
        # # 头部相机链接
        # self.head_camera_link: Link = sapien_utils.get_obj_by_name(
        #     self.robot.get_links(), "link6"
        # )
        
        # 设置碰撞位
        # # 标记基座碰撞位
        # self.base_link.set_collision_group_bit(
        #     group=2, bit_idx=PIPER_BASE_COLLISION_BIT, bit=1
        # )
        
        # # 如果有轮子，标记轮子碰撞位
        # for link in self.robot.get_links():
        #     if "wheel" in link.name:
        #         link.set_collision_group_bit(
        #             group=2, bit_idx=PIPER_WHEELS_COLLISION_BIT, bit=1
        #         )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85, arm="left"):
        """检查机器人是否抓取物体（左臂或右臂）"""
        if arm == "left":
            ee_link = self.left_ee_link
        # else:
        #     ee_link = self.right_ee_link
            
        contact_forces = self.scene.get_pairwise_contact_forces(ee_link, object)
        force = torch.linalg.norm(contact_forces, axis=1)
        return force >= min_force

    # def is_static(self, threshold: float = 0.2, base_threshold: float = 0.05):
    #     """检查机器人是否静止"""
    #     body_qvel = self.robot.get_qvel()[..., 3:-2]  # 排除基座和夹爪
    #     base_qvel = self.robot.get_qvel()[..., :3]    # 基座速度
    #     return torch.all(body_qvel <= threshold, dim=1) & torch.all(
    #         base_qvel <= base_threshold, dim=1
    #     )

    #似乎没有用到
    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        print("使用未做修改的build_grasp_pose方法")
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    @property
    def left_tcp_pos(self) -> Pose:
        """左臂末端位置"""
        return self.left_ee_link.pose.p

    @property
    def right_tcp_pos(self) -> Pose:
        """右臂末端位置"""
        return self.right_ee_link.pose.p

    @property
    def left_tcp_pose(self) -> Pose:
        """左臂末端位姿"""
        return self.left_ee_link.pose

    @property
    def right_tcp_pose(self) -> Pose:
        """右臂末端位姿"""
        return self.right_ee_link.pose
