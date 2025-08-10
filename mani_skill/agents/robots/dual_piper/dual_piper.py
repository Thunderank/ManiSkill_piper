# 使用maniskill所需要引入
import numpy as np
import sapien

# 注册agent
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent

# 代码处理用到
import torch
from copy import deepcopy
from typing import Dict, Tuple

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array

#使用宜家raskog底座的双piper臂机器人
@register_agent()
class dual_Piper(BaseAgent):
    uid = "dual_piper"
    # urdf_path = f"{PACKAGE_ASSET_DIR}/robots/dual_piper/dual_piper_raskog.urdf"   #宜家raskog底座
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/dual_piper/dual_piper_tracer.urdf"   #mobile_aloha castor底座
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_arm_link7=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_arm_link8=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_arm_link7=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_arm_link8=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            # 设置关键帧，即初始关节位置。数量为urdf中非fixed的joint数，即可控关节
            qpos=np.array([0]*19),   
        )
    )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="left_wrist_camera",
                #p 第1维是z轴高度，负为向上 第3维是x轴前后，正为向前
                #q 第2、4维设为-1时是正向相机
                pose=Pose.create_from_pq([-0.05, 0, 0.05], [0, -1, 0, -1]),  # 此处设置maniskill中相机的绑定位姿
                width=128,
                height=128,
                fov=55,
                near=0.01,
                far=100,
                entity_uid="left_arm_camera",  # 使用相机链接
            ),
            CameraConfig(
                uid="right_wrist_camera",
                pose=Pose.create_from_pq([-0.05, 0, 0.05], [0, -1, 0, -1]),  # 位姿已在URDF中定义
                width=128,
                height=128,
                fov=55,
                near=0.01,
                far=100,
                entity_uid="right_arm_camera",  # 使用相机链接
            ),
            CameraConfig(
                uid="global_camera",
                # pose=Pose.create_from_pq([0, 0.0, 2.0], [0, 0, 1, 0]),
                pose=sapien_utils.look_at([0, 0, 1], [-0.5, 0.0, 0]),
                width=128,
                height=128,
                fov=78,
                near=0.01,
                far=100,
                entity_uid="global_camera_link",
            )
        ]

    def __init__(self, *args, **kwargs):
        # 左臂关节
        self.left_arm_joint_names = [
            "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", 
            "left_arm_joint4", "left_arm_joint5", "left_arm_joint6"
        ]
        
        # 左夹爪关节
        self.left_gripper_joint_names = ["left_arm_joint7", "left_arm_joint8"]
        
        # 右臂关节
        self.right_arm_joint_names = [
            "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", 
            "right_arm_joint4", "right_arm_joint5", "right_arm_joint6"
        ]
        
        # 右夹爪关节
        self.right_gripper_joint_names = ["right_arm_joint7", "right_arm_joint8"]

        # 末端执行器链接名称
        self.left_ee_link_name = "left_arm_gripper_base"
        self.right_ee_link_name = "right_arm_gripper_base"
        
        # TCP 链接名称 (工具中心点)
        self.left_tcp_link_name = "left_arm_gripper_base"
        self.right_tcp_link_name = "right_arm_gripper_base"

        # 控制器参数
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100
        
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # 左臂位置控制器
        left_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        
        # 左夹爪位置控制器
        left_gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.left_gripper_joint_names,
            [0, -0.035],  # 关节下限
            [0.035, 0],   # 关节上限
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )
        
        # 右臂位置控制器
        right_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.right_arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        
        # 右夹爪位置控制器
        right_gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.right_gripper_joint_names,
            [0, -0.035],  # 关节下限
            [0.035, 0],   # 关节上限
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )

        # 统一控制器配置
        controller_configs = dict(
            # 单臂控制
            pd_joint_delta_pos_left=dict(
                arm=left_arm_pd_joint_delta_pos,
                gripper=left_gripper_pd_joint_pos,
            ),
            pd_joint_delta_pos_right=dict(
                arm=right_arm_pd_joint_delta_pos,
                gripper=right_gripper_pd_joint_pos,
            ),
            
            # 双臂同时控制
            pd_joint_delta_pos_dual_arm=dict(
                left_arm=left_arm_pd_joint_delta_pos,
                left_gripper=left_gripper_pd_joint_pos,
                right_arm=right_arm_pd_joint_delta_pos,
                right_gripper=right_gripper_pd_joint_pos,
            )
        )

        # 返回深拷贝的控制器配置
        return deepcopy(controller_configs)

    def _after_init(self):
        # 左臂末端执行器
        self.left_ee_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.left_ee_link_name
        )
        
        # 右臂末端执行器
        self.right_ee_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.right_ee_link_name
        )
        
        # 左臂TCP
        self.left_tcp: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.left_tcp_link_name
        )
        
        # 右臂TCP
        self.right_tcp: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.right_tcp_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85, arm="right"):
        """检查机器人是否抓取物体"""
        if arm == "left":
            ee_link = self.left_ee_link
        else:
            ee_link = self.right_ee_link
            
        contact_forces = self.scene.get_pairwise_contact_forces(ee_link, object)
        force = torch.linalg.norm(contact_forces, axis=1)
        return force >= min_force

    @property
    def left_tcp_pos(self) -> Array:
        """左臂TCP位置 静态工具中心点（Stationary TCP）是以机器人本体以外的某个点作为中心点，机器人携带工件围绕该点做轨迹运动。 """
        return self.left_tcp.pose.p

    @property
    def left_tcp_pose(self) -> Pose:
        """左臂TCP位姿"""
        return self.left_tcp.pose

    @property
    def right_tcp_pos(self) -> Array:
        """右臂TCP位置"""
        return self.right_tcp.pose.p

    @property
    def right_tcp_pose(self) -> Pose:
        """右臂TCP位姿"""
        return self.right_tcp.pose

    @property
    def left_ee_pos(self) -> Array:
        """左臂末端位置"""
        return self.left_ee_link.pose.p

    @property
    def left_ee_pose(self) -> Pose:
        """左臂末端位姿"""
        return self.left_ee_link.pose

    @property
    def right_ee_pos(self) -> Array:
        """右臂末端位置"""
        return self.right_ee_link.pose.p

    @property
    def right_ee_pose(self) -> Pose:
        """右臂末端位姿"""
        return self.right_ee_link.pose
    

