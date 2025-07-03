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

# PIPER_WHEELS_COLLISION_BIT = 30
# """Collision bit of the piper robot wheel links"""
# PIPER_BASE_COLLISION_BIT = 31
# """Collision bit of the piper base"""


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
            #设置初始关节位置 基座[0] + 右臂(joint 1-6)[1-6] + 夹爪joint[7-8]
            qpos=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,]),   #9维
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
        
        # 右臂关节
        self.right_arm_joint_names = [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        ]
        
        
        # 夹爪关节
        self.gripper_right_joint_names = ["joint7", "joint8"]
        
        self.ee_right_link_name = "link7"  # 右臂夹爪1
        self.ee_right_link_name = "link8"  # 右臂夹爪2
        # 添加 TCP 链接名称 (工具中心点)
        self.tcp_link_name = "link6"  # 夹爪基座作为 TCP


        # 控制器参数
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100
        
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        # 末端执行器链接名称
        self.ee_right_link_name = "link7"  # 右臂夹爪1
        self.ee_right_link_name = "link8"  # 右臂夹爪2

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # --------------------------------------------------------------------------
        # 双臂控制器config
        # --------------------------------------------------------------------------
        # 右臂位置控制器
        right_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.right_arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        
        # --------------------------------------------------------------------------
        # 夹爪控制器
        # --------------------------------------------------------------------------
        # 右夹爪位置控制器
        gripper_right_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_right_joint_names,
            -0.01,  # 最小值
            0.05,    # 最大值
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )


        # 修改控制器配置为统一的控制器
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=PDJointPosControllerConfig(
                    self.right_arm_joint_names + self.gripper_right_joint_names,
                    None,
                    None,
                    self.arm_stiffness,
                    self.arm_damping,
                    self.arm_force_limit,
                    normalize_action=False,
                )
            ),

            # pd_joint_delta_pos_dual_arm=dict(         #从XLeRobot迁移过来的双臂控制方式config
            #     base=base_pd_joint_vel,
            #     arm1=arm_pd_joint_delta_pos,
            #     arm2=arm2_pd_joint_delta_pos,
            #     gripper1=gripper_pd_joint_pos,
            #     gripper2=gripper2_pd_joint_pos,
            #     body=body_pd_joint_delta_pos,
            # ),
        )

        # 返回深拷贝的控制器配置，便于修改配置后复原
        return deepcopy(controller_configs)

    def _after_init(self):
        # 右臂末端执行器
        self.right_ee_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_right_link_name
        )
        # TCP 链接
        self.tcp: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.tcp_link_name
        )



    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85, arm="right"):
        """检查机器人是否抓取物体（右臂）"""
        if arm == "right":
            ee_link = self.right_ee_link
 
        contact_forces = self.scene.get_pairwise_contact_forces(ee_link, object)
        force = torch.linalg.norm(contact_forces, axis=1)
        return force >= min_force



    @property
    def right_tcp_pos(self) -> Pose:
        """右臂末端位置"""
        return self.right_ee_link.pose.p

    @property
    def right_tcp_pose(self) -> Pose:
        """右臂末端位姿"""
        return self.right_ee_link.pose


    def tcp_pos(self):
        """TCP 位置"""
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        """TCP 位姿"""
        return self.tcp.pose