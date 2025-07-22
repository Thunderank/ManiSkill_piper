import casadi                                                                       
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin                             
import time
from pinocchio import casadi as cpin                
from pinocchio.robot_wrapper import RobotWrapper    
from pinocchio.visualize import MeshcatVisualizer   
import os
import sys

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  #祖父目录,即maniskill
sys.path.append(parent2_dir)

from weighted_moving_filter import WeightedMovingFilter


class PIPER_ArmIK:
    def __init__(self, Unit_Test = False, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization
        # current_dir = os.path.dirname(os.path.abspath(__file__))    
        # #设定urdf目录
        # urdf_path = os.path.join(
        #     current_dir,
        #     "assets/rml63_with_gripper/rml_63_gripper.urdf"
        # )
        urdf_path = os.path.join(
            parent2_dir,
            "assets/robots/piper/piper.urdf"
        )
        package_dirs = os.path.join(parent2_dir, "assets")
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_dirs)
        
        #锁定不需要ik逆解计算的关节.此处为夹爪
        self.mixed_jointsToLockIDs = [
                                        "joint7" ,
                                        "joint8" ,
                                    ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        for i in range(self.reduced_robot.model.nframes):
            frame = self.reduced_robot.model.frames[i]
            frame_id = self.reduced_robot.model.getFrameId(frame.name)
            print(f"Frame ID: {frame_id}, Name: {frame.name}")
        
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.hand_id = self.reduced_robot.model.getFrameId("Link00")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.hand_id].translation - self.cTf[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.hand_id].rotation @ self.cTf[:3,:3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-6
            },
            'print_time':False,# print or not
            'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 6)
        self.vis = None

    # If the robot arm is not the same size as your arm :)
    def scale_arms(self, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        # scale_factor = robot_arm_length / human_arm_length
        scale_factor = 1.3
        robot_right_pose = human_right_pose.copy()
        robot_right_pose[:3, 3] *= scale_factor
        return robot_right_pose

    def solve_ik(self, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        right_wrist = self.scale_arms(right_wrist)
        # if self.Visualization:  
        #     self.vis.viewer['R_ee_target'].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            print(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)

    def compute_fk(self, q):
        """
        Input joint angle q, output end effector pose (SE3 object)
        :param q: joint angle vector (must be consistent with the simplified model degree of freedom)
        :return: end effector pose (SE3)
        """
        # Make sure the input dimensions match the model
        assert len(q) == self.reduced_robot.model.nq, "Joint angle dimensions do not match"
        
        # Initialize Data
        data = self.reduced_robot.data
        
        # Computational forward kinematics
        pin.forwardKinematics(self.reduced_robot.model, data, q)
        
        # Update the pose of the target frame (the frame_id here must be consistent with that in IK)s
        pin.updateFramePlacement(self.reduced_robot.model, data, self.hand_id)
        
        # Get the end pose (SE3 object)
        end_effector_pose = data.oMf[self.hand_id]

        x, y, z = end_effector_pose.translation
    
        # Extracting Quaternions
        rotation_matrix = end_effector_pose.rotation
        quat = pin.Quaternion(rotation_matrix)
        qx, qy, qz, qw = quat.coeffs()  # Note that the order is (x, y, z, w)
        
        return [x, y, z, qx, qy, qz, qw]   

class RML63_ArmIK:
    def __init__(self, Unit_Test = False, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(
            current_dir,
            "assets/rml63_with_gripper/rml_63_gripper.urdf"
        )
        # urdf_path = "../common/robot_devices/teleop/robot_arm/assets/rml63_with_gripper/rml_63_gripper.urdf"
        # package_dirs = [os.path.abspath("../common/robot_devices/teleop/robot_arm/assets/")]
        package_dirs = os.path.join(current_dir, "assets")
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_dirs)
        # self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=current_dir)
        
        self.mixed_jointsToLockIDs = [
                                        "joint01" ,
                                        "joint11" ,
                                        "joint02" ,
                                        "joint22" 
                                    ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        for i in range(self.reduced_robot.model.nframes):
            frame = self.reduced_robot.model.frames[i]
            frame_id = self.reduced_robot.model.getFrameId(frame.name)
            print(f"Frame ID: {frame_id}, Name: {frame.name}")
        
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.hand_id = self.reduced_robot.model.getFrameId("Link00")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.hand_id].translation - self.cTf[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.hand_id].rotation @ self.cTf[:3,:3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-6
            },
            'print_time':False,# print or not
            'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 6)
        self.vis = None

        # if self.Visualization:
        #     # Initialize the Meshcat visualizer for visualization
        #     self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        #     self.vis.initViewer(open=True) 
        #     self.vis.loadViewerModel("pinocchio") 
        #     self.vis.displayFrames(True, frame_ids=[101, 102], axis_length = 0.15, axis_width = 5)
        #     self.vis.display(pin.neutral(self.reduced_robot.model))

        #     # Enable the display of end effector target frames with short axis lengths and greater width.
        #     frame_viz_names = ['L_ee_target', 'R_ee_target']
        #     FRAME_AXIS_POSITIONS = (
        #         np.array([[0, 0, 0], [1, 0, 0],
        #                   [0, 0, 0], [0, 1, 0],
        #                   [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        #     )
        #     FRAME_AXIS_COLORS = (
        #         np.array([[1, 0, 0], [1, 0.6, 0],
        #                   [0, 1, 0], [0.6, 1, 0],
        #                   [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        #     )
        #     axis_length = 0.1
        #     axis_width = 10
        #     for frame_viz_name in frame_viz_names:
        #         self.vis.viewer[frame_viz_name].set_object(
        #             mg.LineSegments(
        #                 mg.PointsGeometry(
        #                     position=axis_length * FRAME_AXIS_POSITIONS,
        #                     color=FRAME_AXIS_COLORS,
        #                 ),
        #                 mg.LineBasicMaterial(
        #                     linewidth=axis_width,
        #                     vertexColors=True,
        #                 ),
        #             )
        #         )

    # If the robot arm is not the same size as your arm :)
    def scale_arms(self, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        # scale_factor = robot_arm_length / human_arm_length
        scale_factor = 1.3
        robot_right_pose = human_right_pose.copy()
        robot_right_pose[:3, 3] *= scale_factor
        return robot_right_pose

    def solve_ik(self, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        right_wrist = self.scale_arms(right_wrist)
        # if self.Visualization:  
        #     self.vis.viewer['R_ee_target'].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            print(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)

    def compute_fk(self, q):
        """
        Input joint angle q, output end effector pose (SE3 object)
        :param q: joint angle vector (must be consistent with the simplified model degree of freedom)
        :return: end effector pose (SE3)
        """
        # Make sure the input dimensions match the model
        assert len(q) == self.reduced_robot.model.nq, "Joint angle dimensions do not match"
        
        # Initialize Data
        data = self.reduced_robot.data
        
        # Computational forward kinematics
        pin.forwardKinematics(self.reduced_robot.model, data, q)
        
        # Update the pose of the target frame (the frame_id here must be consistent with that in IK)s
        pin.updateFramePlacement(self.reduced_robot.model, data, self.hand_id)
        
        # Get the end pose (SE3 object)
        end_effector_pose = data.oMf[self.hand_id]

        x, y, z = end_effector_pose.translation
    
        # Extracting Quaternions
        rotation_matrix = end_effector_pose.rotation
        quat = pin.Quaternion(rotation_matrix)
        qx, qy, qz, qw = quat.coeffs()  # Note that the order is (x, y, z, w)
        
        return [x, y, z, qx, qy, qz, qw]   


class RM75_ArmIK:
    def __init__(self, Unit_Test = False, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(
            current_dir,
            "assets/rm75_6f_gripper/rm_75_6f_gripper_description.urdf"
        )
        # urdf_path = "../common/robot_devices/teleop/robot_arm/assets/rml63_with_gripper/rml_63_gripper.urdf"
        # package_dirs = [os.path.abspath("../common/robot_devices/teleop/robot_arm/assets/")]
        package_dirs = os.path.join(current_dir, "assets")
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_dirs)
        # self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=current_dir)
        
        self.mixed_jointsToLockIDs = [
                                        "joint01" ,
                                        "joint11" ,
                                        "joint02" ,
                                        "joint22" 
                                    ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        for i in range(self.reduced_robot.model.nframes):
            frame = self.reduced_robot.model.frames[i]
            frame_id = self.reduced_robot.model.getFrameId(frame.name)
            print(f"Frame ID: {frame_id}, Name: {frame.name}")
        
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.hand_id = self.reduced_robot.model.getFrameId("Link00")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.hand_id].translation - self.cTf[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.hand_id].rotation @ self.cTf[:3,:3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-6
            },
            'print_time':False,# print or not
            'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 7)
        self.vis = None

    # If the robot arm is not the same size as your arm :)
    def scale_arms(self, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        # scale_factor = 1.3
        robot_right_pose = human_right_pose.copy()
        robot_right_pose[:3, 3] *= scale_factor
        return robot_right_pose

    def solve_ik(self, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        right_wrist = self.scale_arms(right_wrist)
        # if self.Visualization:  
        #     self.vis.viewer['R_ee_target'].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            print(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)

    def compute_fk(self, q):
        """
        Input joint angle q, output end effector pose (SE3 object)
        :param q: joint angle vector (must be consistent with the simplified model degree of freedom)
        :return: end effector pose (SE3)
        """
        # Make sure the input dimensions match the model
        assert len(q) == self.reduced_robot.model.nq, "Joint angle dimensions do not match"
        
        # Initialize Data
        data = self.reduced_robot.data
        
        # Computational forward kinematics
        pin.forwardKinematics(self.reduced_robot.model, data, q)
        
        # Update the pose of the target frame (the frame_id here must be consistent with that in IK)s
        pin.updateFramePlacement(self.reduced_robot.model, data, self.hand_id)
        
        # Get the end pose (SE3 object)
        end_effector_pose = data.oMf[self.hand_id]

        x, y, z = end_effector_pose.translation
    
        # Extracting Quaternions
        rotation_matrix = end_effector_pose.rotation
        quat = pin.Quaternion(rotation_matrix)
        qx, qy, qz, qw = quat.coeffs()  # Note that the order is (x, y, z, w)
        
        return [x, y, z, qx, qy, qz, qw]   

class Aloha_ArmIK:
    def __init__(self, Unit_Test = False, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization
        urdf_path = "assets/alhoa/piper_description.urdf"
        package_dirs = [os.path.abspath("assets/")]
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_dirs)
        
        self.mixed_jointsToLockIDs = [
                                        "joint7" ,
                                        "joint8" 
                                    ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        for i in range(self.reduced_robot.model.nframes):
            frame = self.reduced_robot.model.frames[i]
            frame_id = self.reduced_robot.model.getFrameId(frame.name)
            print(f"Frame ID: {frame_id}, Name: {frame.name}")
        
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.hand_id = self.reduced_robot.model.getFrameId("gripper_base")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.hand_id].translation - self.cTf[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.hand_id].rotation @ self.cTf[:3,:3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-6
            },
            'print_time':False,# print or not
            'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 6)
        self.vis = None

    # If the robot arm is not the same size as your arm :)
    def scale_arms(self, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_right_pose = human_right_pose.copy()
        robot_right_pose[:3, 3] *= scale_factor
        return robot_right_pose

    def solve_ik(self, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        # left_wrist, right_wrist = self.scale_arms(left_wrist, right_wrist)
        # if self.Visualization:  
        #     self.vis.viewer['R_ee_target'].set_transform(right_wrist)  # for visualization
        
        self.opti.set_value(self.param_tf, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            print(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)

    def compute_fk(self, q):
        """
        Input joint angle q, output end effector pose (SE3 object)
        :param q: joint angle vector (must be consistent with the simplified model degree of freedom)
        :return: end effector pose (SE3)
        """
        # Make sure the input dimensions match the model
        assert len(q) == self.reduced_robot.model.nq, "Joint angle dimensions do not match"
        
        # Initialize Data
        data = self.reduced_robot.data
        
        # Computational forward kinematics
        pin.forwardKinematics(self.reduced_robot.model, data, q)
        
        # Update the pose of the target frame (the frame_id here must be consistent with that in IK)s
        pin.updateFramePlacement(self.reduced_robot.model, data, self.hand_id)
        
        # Get the end pose (SE3 object)
        end_effector_pose = data.oMf[self.hand_id]

        x, y, z = end_effector_pose.translation
    
        # Extracting Quaternions
        rotation_matrix = end_effector_pose.rotation
        quat = pin.Quaternion(rotation_matrix)
        qx, qy, qz, qw = quat.coeffs()  # Note that the order is (x, y, z, w)
        
        return [x, y, z, qx, qy, qz, qw]   

if __name__ == "__main__":
    # arm_ik = RML63_ArmIK(Unit_Test = True, Visualization = False)
    arm_ik = Aloha_ArmIK(Unit_Test = True, Visualization = False)
    
    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(0.0026, -0.353, 0.002, 0.9353),
        np.array([-0.135, 0.0, 0.722]),
    )

    right_origin_pose = pin.SE3(
        pin.Quaternion(0.737273, 0, 0.675595, 0),
        np.array([0.054963, 0.0, 0.20332]),
    )

    rotation_speed = 0.005
    noise_amplitude_translation = 0.001
    noise_amplitude_rotation = 0.1

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):\n")
    if user_input.lower() == 's':
        step = 0
        while True:
            print("right_origin_pose:", [0.054963, 0.0, 0.20332, 0, 0.675595, 0, 0.737273])
            sol_q, sol_tauff = arm_ik.solve_ik(right_origin_pose.homogeneous)
            # print("sol_q",sol_q)
            # print("sol_tauff",sol_tauff)
            compute_right_pose = arm_ik.compute_fk(sol_q)
            print("compute_right_pose:", compute_right_pose)

            step += 1
            if step > 240:
                step = 0
            time.sleep(0.1)