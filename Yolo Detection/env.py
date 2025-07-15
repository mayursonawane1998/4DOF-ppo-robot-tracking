import random
import time
import numpy as np
import gym
import pybullet as p
import os
import math
import pybullet_data
import pybullet
import cv2
import torch
from gym import spaces
from collections import namedtuple
from attrdict import AttrDict
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
 
# x,y,z distance
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)
 
 
# x,y distance
def goal_distance2d(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)
 
ROBOT_URDF_PATH = r"C:\Users\Ankit\myenv_robot\Lib\site-packages\pybullet_data\Assem3\urdf\Assem31.urdf"
CUBE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "cube1.urdf")
 
# Initialize YOLO model
model = YOLO('yolov8s.pt')
 
class RobotGymEnv(gym.Env):
    def __init__(self,
                 camera_attached=False,
                 actionRepeat=1,
                 renders=False,
                 maxSteps=100,
                 simulatedGripper=False,
                 randObjPos=False,
                 task=0,
                 learning_param=0):
        # Initialize the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
 
        self.model = model
        print(f"YOLO model initialized: {self.model}")  # Debugging information
 
        self.renders = renders
        self.actionRepeat = actionRepeat
        self.camera_attached = camera_attached
 
        # Camera parameters
        self.camera_width = 640
        self.camera_height = 640
        self.camera_image = None
        self.camera_position = [0.1735, 0, -0.020006]
        self.camera_orientation = [0, 1.5708, 0]
        self.camera_joint_index = 4
        self.near_plane = 0.1
        self.far_plane = 100
        self.fov = 115
 
        # Setup pybullet sim:
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
 
        p.stepSimulation()
        time.sleep(1. / 240.)  # Adjust as needed
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(False)
 
        p.loadURDF("plane1.urdf")
 
        # Setup robot arm:
        self.end_effector_index = 4
        self.Robot = p.loadURDF(ROBOT_URDF_PATH, [0, 0, 0.15], [0, 0, 0, 1], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.Robot)
        self.control_joints = ["joint_1", "joint_2", "joint_3", "joint_4"]
        self.joint_type_list = ["REVOLUTE", "REVOLUTE", "REVOLUTE", "REVOLUTE", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "Velocity", "controllable"])
 
        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = p.getJointInfo(self.Robot, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointVelocity = info[10]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointVelocity, controllable)
            if info.type == "REVOLUTE":
                p.setJointMotorControl2(self.Robot, info.id, p.VELOCITY_CONTROL)
            self.joints[info.name] = info
 
        # Object:
        self.initial_obj_pos = [1.15, 0, 0.085]  # initial object pos
        self.obj = p.loadURDF(CUBE_URDF_PATH, self.initial_obj_pos, useFixedBase=True)
        p.changeVisualShape(self.obj, -1, rgbaColor=[0.8, 0.1, 0.1, 1.0])
 
        self.name = 'RobotGymEnv'
        self.simulatedGripper = simulatedGripper
        self.action_dim = 4
        self.stepCounter = 0
        self.maxSteps = maxSteps
        self.terminated = False
        self.randObjPos = randObjPos
        self.observation = np.array(0)
 
        self.task = task
        self.learning_param = learning_param
 
        self.obj_pos_range = {
            "x": (1.0, 1.25),
            "y": (-0.75, 0.75),
            "z": (0.05, 1.09)  # Ensure it's above the ground
        }
 
        self._action_bound = 0.5  # delta limits
        action_high = np.array([self._action_bound] * self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype='float32')
        self.reset()
 
        high = np.array([10]*self.observation.shape[0])  # Adjust size based on your observation features
        self.observation_space = spaces.Box(-high, high, shape=self.observation.shape, dtype=np.float32)
 
        self.model = model
 
        self.obj_pos = np.zeros(3)
        self.achieved_goal = np.zeros(6)
        self.desired_goal = np.zeros(3)
 
    def set_joint_angles(self, joint_angles):
        target_positions = joint_angles  # Extract target positions for controllable joints
        joint_indices = [self.joints[name].id for name in self.control_joints]  # Get joint indices
 
        p.setJointMotorControlArray(
            bodyUniqueId=self.Robot,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
        )
 
    def get_joint_angles(self):
        j = p.getJointStates(self.Robot, [0, 1, 2, 3])
        joints = [i[0] for i in j]
        return joints
 
    def check_collisions(self):
        collisions = p.getContactPoints()
        return len(collisions) > 0
 
    def calculate_ik(self, position, orientation):
        quaternion = p.getQuaternionFromEuler(orientation)
        lower_limits = [-1.5, -1, -1, -1.58]
        upper_limits = [1.5, 0.55, 1.484, 1.58]
        joint_ranges = [math.pi] * 4
        rest_poses = [0, 0, -0.719, -0.368]  # rest pose of our Robot robot
 
        joint_angles = p.calculateInverseKinematics(
            self.Robot, self.end_effector_index, position, quaternion,
            jointDamping=[0.01] * 4, upperLimits=upper_limits,
            lowerLimits=lower_limits, jointRanges=joint_ranges,
            restPoses=rest_poses
        )
        return joint_angles
 
    def get_current_pose(self):
        linkstate = p.getLinkState(self.Robot, self.end_effector_index, computeForwardKinematics=True)
        position = linkstate[0] if linkstate is not None else [0, 0, 0]
        orientation = linkstate[1] if linkstate is not None else [0, 0, 0]
        return (position, orientation)
 
    def reset(self):
        self.stepCounter = 0
        self.terminated = False
        self.Robot_or = [0, 1.5708, 0]
 
        # Randomize object position if required
        if self.randObjPos:
            self.initial_obj_pos = [
                np.random.uniform(*self.obj_pos_range["x"]),
                np.random.uniform(*self.obj_pos_range["y"]),
                np.random.uniform(*self.obj_pos_range["z"])
            ]
        else:
            #Set to default position if not randomizing
            self.initial_obj_pos = [1.05, 0, 0.09]
 
        p.resetBasePositionAndOrientation(self.obj, self.initial_obj_pos, [0., 0., 0., 1.0])
 
        joint_angles = (0, 0, -0.719, -0.368)
        self.set_joint_angles(joint_angles)
 
        for i in range(100):
            p.stepSimulation()
 
        self.getExtendedObservation()
        return self.observation
 
    def step(self, action):
        action = np.array(action)
        arm_action = action[0:self.action_dim-1].astype(float)
        cur_p = self.get_current_pose()
        new_p = np.array(cur_p[0]) + arm_action
 
        joint_angles = self.calculate_ik(new_p, self.Robot_or)
        self.set_joint_angles(joint_angles)
 
        for _ in range(self.actionRepeat):
            p.stepSimulation()
            if self.renders: 
                time.sleep(1. / 240.)
 
        self.getExtendedObservation()
        reward = self.compute_reward(self.achieved_goal, self.desired_goal, None)
        done = self.my_task_done()
 
        info = {'is_success': False}
        if self.terminated == self.task:
            info['is_success'] = True
 
        self.stepCounter += 1
        print(self.stepCounter)
 
        return self.observation, reward, done, info
 
    def capture_camera_image(self):
        end_effector_link_index = 4
        end_effector_state = p.getLinkState(bodyUniqueId=self.Robot, linkIndex=end_effector_link_index, computeForwardKinematics=True)
    
        if end_effector_state is not None:
            camera_position_relative = [0, 0, -0.01]  # Relative to end effector
            camera_orientation_relative = [0, 0, 0, 1]  # No rotation relative to the end effector
    
            # Compute the camera position and orientation in world space
            camera_position_world, camera_orientation_world = p.multiplyTransforms(
                end_effector_state[0],  # Position of the end effector in world space
                end_effector_state[1],  # Orientation of the end effector in world space
                camera_position_relative,  # Position of the camera relative to the end effector
                camera_orientation_relative  # Orientation of the camera relative to the end effector
            )
    
            
        
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_position_world,
                cameraTargetPosition=end_effector_state[0],  # Assuming camera looks towards the end effector
                cameraUpVector=[0, 0, 1]  # Up vector for the camera
            )
    
            projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.camera_width / self.camera_height, self.near_plane, self.far_plane)
    
            width, height, rgba_image_raw, depth_image_raw, _ = p.getCameraImage(
                width=self.camera_width,
                height=self.camera_height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_TINY_RENDERER
            )
    
            rgba_image_raw = np.array(rgba_image_raw).reshape((height, width, 4))
            depth_image_raw = np.array(depth_image_raw).reshape((height, width))
    
            rgb_image_raw = rgba_image_raw[:, :, :3]
    
            if rgb_image_raw.shape != (self.camera_height, self.camera_width, 3):
                raise ValueError(f"Captured image has invalid shape: {rgb_image_raw.shape}")
    
            return rgb_image_raw, depth_image_raw
        else:
            raise RuntimeError("Failed to get end effector state.")
 
 
 
    def getExtendedObservation(self):
        self.obj_pos = np.array([1.15, 0, 0.085])  # Default or initial position
        objects_pos = self.obj_pos
        rgb_image_raw, depth_image_raw = self.capture_camera_image()
        
        processed_image = self.preprocess_image(rgb_image_raw)
        input_image = np.transpose(processed_image, (1, 2, 0))
        
        results = self.model(input_image)  # YOLO detections in the new format
        boxes = []
        
        for result in results:
            if result.boxes is not None and len(result.boxes.xyxy) > 0:
                boxes = result.boxes.xyxy  # Assign detected boxes to the variable
        
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    print(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        cube_boxes = [box for box in boxes]  # Adjust class ID if needed
        
        if len(cube_boxes) == 0:
            self.cube_detected = False
            print("Cube not detected")
        else:
            self.cube_detected = True
            self.cube_detected = len(cube_boxes) > 0
            x1, y1, x2, y2 = cube_boxes[0][:4]
            cube_center_x = (x1 + x2) / 2
            cube_center_y = (y1 + y2) / 2
    
            # Print cube center from RGB image
            print(f"Cube Center (RGB): x={cube_center_x}, y={cube_center_y}")
    
            target_position = self.calculate_target_position([cube_center_x, cube_center_y], depth_image_raw)
            self.obj_pos = np.array(target_position)
            print(target_position)
            objects_pos = self.obj_pos
            print(objects_pos) 
    
        goal_pos = self.obj_pos
        tool_pos = self.get_current_pose()[0]  # Get current tool position
        self.observation = np.array(np.concatenate((tool_pos, objects_pos)))
        self.achieved_goal = np.array(np.concatenate((objects_pos, tool_pos)))
        self.desired_goal = np.array(goal_pos)
        self.obj_pos1, _ = pybullet.getBasePositionAndOrientation(self.obj)
        goal_pos1 = self.obj_pos1
        objects_pos1 = np.array(self.obj_pos1)
        print(objects_pos1)
        self.achieved_goal1 = np.concatenate((objects_pos1, tool_pos))
        self.desired_goal1 = np.array(goal_pos1)
        end_effector_pos = self.achieved_goal1[-3:]
        self.target_dist1 = goal_distance(end_effector_pos, self.desired_goal1)
        print(f"Target Distance Simulation: {self.target_dist1}")
        
 
    def preprocess_image(self, rgb_image_raw):
        original_shape = rgb_image_raw.shape[:2]
        target_shape = (640, 640)
        
        new_shape = (target_shape[0], int(original_shape[1] * target_shape[0] / original_shape[0]))
        if new_shape[1] > target_shape[1]:
            new_shape = (int(original_shape[0] * target_shape[1] / original_shape[1]), target_shape[1])
        resized_image = cv2.resize(rgb_image_raw, (new_shape[1], new_shape[0]))
        
        pad_top = (target_shape[0] - new_shape[0]) // 2
        pad_bottom = target_shape[0] - new_shape[0] - pad_top
        pad_left = (target_shape[1] - new_shape[1]) // 2
        pad_right = target_shape[1] - new_shape[1] - pad_left
        
        padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        normalized_image = padded_image / 255.0
    
        if normalized_image.shape[-1] == 1:
            normalized_image = cv2.cvtColor(padded_image, cv2.COLOR_GRAY2BGR)
        else:
            normalized_image = padded_image
    
        uint8_image = (normalized_image * 255).astype(np.uint8)
        transposed_image = np.transpose(uint8_image, (2, 0, 1))
        return transposed_image
 
    def pixel_to_3d(self, x, y, depth_image_raw):
        depth = depth_image_raw[y, x]
        print("Depth at (x, y):", depth)
        if depth == 0:
            return None
    
        self.fx = self.camera_width / (2 * np.tan(np.radians(self.fov / 2)))
        self.fy = self.fx  # Assuming square pixels
    
        self.cx = self.camera_width / 2
        self.cy = self.camera_height / 2
    
        # Calculate camera coordinates from pixel coordinates
        x_camera = (x - self.cx) * depth / self.fx
        y_camera = (y - self.cy) * depth / self.fy
        z_camera = depth  # Depth is along the camera's viewing direction
        
        # Adjust for camer0.4 to a orientation [0, 1.5708, 0] (rotation around Y-axis)
        x_world = z_camera
        y_world = -x_camera
        z_world = y_camera
    
        print(f"Calculated 3D Coordinates: x={x_world}, y={y_world}, z={z_world}")
        return [x_world, y_world, z_world]
 
 
 
 
    def calculate_target_position(self, cube_center, depth_image_raw):
        # Directly use the 3D position from pixel to 3D conversion
        target_3d = self.pixel_to_3d(int(cube_center[0]), int(cube_center[1]), depth_image_raw)
    
        if target_3d is None:
            print("Warning: Invalid depth data, using default position.")
            return self.initial_obj_pos
    
        return target_3d  # Return the correctly transformed 3D coordinates
 
 
 
 
 
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = 0
        
        # Only compute the target distance if the cube is detected
        if self.cube_detected:
            # Compute end-effector position from achieved_goal
            end_effector_pos = achieved_goal[-3:]
            self.target_dist = goal_distance(end_effector_pos, desired_goal)
            print(self.target_dist) 
            
            # Apply a dynamic offset correction based on target distance
            if 1 <= self.target_dist <= 1.65:
                self.corrected_target_dist = self.target_dist - 0.7  # Subtracting an average offset
            else:
                self.corrected_target_dist = self.target_dist
            
            # Initialize min_distance and max_distance for reward calculations
            min_distance = 0.25
            max_distance = 0.85
            heavy_penalty = 25
            
            # Reward logic based on distance
            if min_distance <= self.corrected_target_dist <= max_distance:
                reward += 40  # Reward for being within the desired distance range
            else:
                reward -= heavy_penalty  # Penalty for being outside the desired range
            
            # Penalize based on corrected distance from the goal
            reward -= self.corrected_target_dist * 5
            
            # Check if the goal is achieved
            if self.corrected_target_dist < min_distance:
                self.terminated = True
                reward += 30 
    
        else:
            reward -= 30  # Penalty for not detecting the cube
    
        # Reward or penalty based on cube detection
        if self.cube_detected:
            reward += 75  # Reward for detecting the cube
            
        
                
        # Check for collisions
        if self.check_collisions():
            reward -= 1  # Penalty for collisions
        
        # Print debugging information
        print(f"Target Distance: {self.corrected_target_dist if self.cube_detected else 'N/A'}, Reward: {reward}")
        
        return reward  # Return the computed reward
 
 
 
 
    def my_task_done(self):
        return self.terminated or self.stepCounter > self.maxSteps
 
    def close(self):
        #p.disconnect()
        print("PyBullet connection closed.")
