import random
import time
import numpy as np
import sys
from gym import spaces
import gym
import pybullet as p
import os
import math 
import pybullet
import pybullet_data
from datetime import datetime
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import cv2
 
ROBOT_URDF_PATH = r"C:\Users\Ankit\myenv_robot\Lib\site-packages\pybullet_data\Assem3\urdf\Assem31.urdf"
 
CUBE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "cube1.urdf")
 
# x,y,z distance
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)
 
 
# x,y distance
def goal_distance2d(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)
 
 
class RobotGymEnv(gym.Env):
    def __init__(self,
                 camera_attached=False,
                 actionRepeat=1,
                 renders=False,
                 maxSteps=100,
                 randObjPos=False,
                 task=0, # here target number
                 learning_param=0):
 
        self.renders = renders
        self.actionRepeat = actionRepeat
        self.camera_attached = camera_attached
 
        # Camera parameters
        self.camera_width = 480
        self.camera_height = 480
        self.camera_image = None  # Placeholder for the camera image
        # Camera parameters
        self.camera_position = [0.1735, 0, -0.020006]
        self.camera_orientation = [0, 1.5708, 0]
        self.camera_joint_index = 4
 
        
        self.near_plane = 0.1
        self.far_plane = 100
        self.fov = 115
 
         # setup pybullet sim:
        if self.renders:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
 
        # Step the simulation
        p.stepSimulation()
    
        # Sleep to control the simulation speed
        time.sleep(1. / 240.)  # Adjust as needed
        pybullet.setGravity(0,0,-10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setRealTimeSimulation(False)
 
        p.loadURDF("plane.urdf")
        
        # setup robot arm:
        self.end_effector_index = 4
        
        flags = pybullet.URDF_USE_SELF_COLLISION
        self.Robot = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0.15], [0, 0, 0, 1], useFixedBase=True)
        self.num_joints = pybullet.getNumJoints(self.Robot)
        self.control_joints = ["joint_1", "joint_2", "joint_3", "joint_4"]
        self.joint_type_list = ["REVOLUTE", "REVOLUTE", "REVOLUTE", "REVOLUTE", "FIXED"] 
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "Velocity", "controllable"])
 
 
        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.Robot, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit,  jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.Robot, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info
 
        # object:
        self.initial_obj_pos = [1.05, 0, 0.09] # initial object pos
        self.obj = pybullet.loadURDF(CUBE_URDF_PATH, self.initial_obj_pos, useFixedBase=True)
        p.changeVisualShape(self.obj, -1, rgbaColor=[0.8, 0.1, 0.1, 1.0])
 
        self.name = 'RobotGymEnv'
      
        self.action_dim = 4
        self.stepCounter = 0
        self.maxSteps = maxSteps
        self.terminated = False
        self.randObjPos = randObjPos
        self.observation = np.array(0)
 
        self.task = task
        self.learning_param = learning_param
 
        # Tracking lists for analysis:
        self.goal_positions = []  # Store goal positions for each episode
        self.end_effector_positions = []  # Store end-effector positions for each episode
        self.positional_errors = []  # Store positional errors for each episode
        self.target_distances = []  # Store target distances for each episode
 
        # Define object position range
        self.obj_pos_range = {
            "x": (1.1, 1.2),
            "y": (-0.7, 0.7),
            "z": (0.05, 1.0)  # Ensure it's above the ground
        }
     
        self._action_bound = 0.1 # delta limits
        action_high = np.array([self._action_bound] * self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype='float32')
        self.reset()
        image_dim = 480 * 480 * 4  # 480x480 image with 4 channels (RGBA)
        obs_dim = 6 + image_dim  # 6 for tool_pos and objects_pos + flattened image dimensions
        high = np.array([10]*obs_dim)
        self.observation_space = spaces.Box(-high, high, dtype='float32')
        #print(f"self.observation_space: {self.observation_space.shape}")
        #print(f"Updated observation dtype: {self.observation_space.dtype}")
 
    def set_joint_angles(self, joint_angles):
        target_positions = joint_angles  # Extract target positions for controllable joints
        joint_indices = [self.joints[name].id for name in self.control_joints]  # Get joint indices
 
        pybullet.setJointMotorControlArray(
            bodyUniqueId=self.Robot,
            jointIndices=joint_indices,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=target_positions,
        )
 
    def get_joint_angles(self):
        j = pybullet.getJointStates(self.Robot, [1,2,3,4])
        joints = [i[0] for i in j]
        return joints
    
 
    def check_collisions(self):
        collisions = pybullet.getContactPoints()
        if len(collisions) > 0:
            # print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False
 
 
    def calculate_ik(self, position, orientation):
     
        quaternion = pybullet.getQuaternionFromEuler(orientation)
        lower_limits = [-1.57 -0.95, -1.571, -1.58]
        upper_limits = [1.57, 1.9, 1.484, 1.58]
        joint_ranges = [math.pi] * 4
        rest_poses = [0, 0, -0.719, -0.368]  # rest pose of our Robot robot
        joint_angles = pybullet.calculateInverseKinematics(
            self.Robot, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*4, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )
        return joint_angles
       
        
    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.Robot, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)
 
 
    def reset(self):
        self.stepCounter = 0
        self.terminated = False
        self.Robot_or = [0, 1.5708, 0]
 
        # Reset tracking lists at the start of each episode:
        self.goal_positions = []
        self.end_effector_positions = []
        self.positional_errors = [] 
        self.target_distances = []  # Reset target distances
 
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
 
        pybullet.resetBasePositionAndOrientation(self.obj, self.initial_obj_pos, [0., 0., 0., 1.0])
 
 
        # reset robot simulation and position:
        joint_angles = (0, 0, -0.719, -0.368) # pi/2 = 1.5707
        self.set_joint_angles(joint_angles)
 
        # step simualator:
        for i in range(100):
            pybullet.stepSimulation()
 
        # get obs and return:
        self.getExtendedObservation()
        return self.observation
    
    
    def step(self, action):
        action = np.array(action)
        arm_action = action[0:self.action_dim-1].astype(float) # dX, dY, dZ - range: [-1,1]
    
        # get current position:
        cur_p = self.get_current_pose()
        # add delta position:
        new_p = np.array(cur_p[0]) + arm_action
        # actuate: 
        joint_angles = self.calculate_ik(new_p, self.Robot_or) # XYZ and angles set to zero
        self.set_joint_angles(joint_angles)
        
        # step simualator:
        for i in range(self.actionRepeat):
            pybullet.stepSimulation()
            if self.renders: time.sleep(1./240.)
        
        self.getExtendedObservation()
        # Record the current goal and end-effector positions
        end_effector_pos = self.get_current_pose()[0]
        target_distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.desired_goal))
        self.target_distances.append(target_distance)
 
        self.end_effector_positions.append(end_effector_pos)
        self.goal_positions.append(self.desired_goal)
 
        # Calculate and record the positional error
        positional_error = np.linalg.norm(np.array(end_effector_pos) - np.array(self.desired_goal))
        self.positional_errors.append(positional_error)
        observation_input = self.final_obs
        
        reward = self.compute_reward(self.achieved_goal, self.desired_goal, None)
        done = self.my_task_done()
 
        info = {'is_success': False}
        if self.terminated == self.task:
            info['is_success'] = True
 
        self.stepCounter += 1
 
        return observation_input, reward, done, info
 
    def capture_camera_image(self):
        # Set camera position and orientation based on the end effector state
        end_effector_link_index = 4  # Assuming the end effector is Link4
        end_effector_state = p.getLinkState(bodyUniqueId=self.Robot, linkIndex=end_effector_link_index, computeForwardKinematics=True)
        
        if end_effector_state is not None:
            # Get camera position and orientation relative to the end effector
            camera_position_relative = [0, 0, -0.01]  # Relative camera position from the end effector
            camera_orientation_relative = [0, 0, 0, 1]  # Relative camera orientation from the end effector (quaternion)
    
            # Convert relative camera position to world coordinate
            camera_position_world, _ = p.multiplyTransforms(
                end_effector_state[0],  # Position of the end effector in world coordinates
                end_effector_state[1],  # Orientation of the end effector in world coordinates (quaternion)
                camera_position_relative,  # Relative camera position from the end effector
                camera_orientation_relative  # Relative camera orientation from the end effector (quaternion)
            )
    
            # Compute view matrix with adjusted camera position and orientation
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_position_world,
                cameraTargetPosition=end_effector_state[0],  # No target position, capture the scene without focus
                cameraUpVector=[0, 0, 1]
            )
    
            projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.camera_width / self.camera_height, self.near_plane, self.far_plane)
            width, height, rgba_image_raw, _, _ = p.getCameraImage(
                width=self.camera_width,
                height=self.camera_height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_TINY_RENDERER
            )
    
            rgba_image_raw = np.array(rgba_image_raw).reshape((height, width, 4))
            # Print debugging information
            #print("Captured image shape:", rgba_image_raw.shape)
            #print("Captured image dtype:", rgba_image_raw.dtype)
           
    
            return rgba_image_raw
        else:
            raise RuntimeError("Failed to get end effector state.")
    
          
 
 
    def preprocess_image(self, image):
        # Preprocess the captured camera image
        # Resize the image to a consistent size
        image = cv2.resize(image, (self.camera_width, self.camera_height))
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        # Convert image to grayscale if needed
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
 
 
 
 
    def getExtendedObservation(self):
        tool_pos = np.array(self.get_current_pose()[0])
        self.obj_pos, _ = pybullet.getBasePositionAndOrientation(self.obj)
        objects_pos = np.array(self.obj_pos)
        goal_pos = np.array(self.obj_pos)
    
        # Capture and preprocess the image
        camera_image = self.capture_camera_image()
        processed_image = self.preprocess_image(camera_image)
    
        # Flatten the preprocessed image
        preprocessed_image = processed_image.flatten()
    
        # Combine all observation components
        full_observation = np.concatenate((tool_pos, objects_pos, preprocessed_image))
    
        # Store the full observation 
        self.observation = full_observation
    
        # Process the observation but only use the first 6 elements
        self.final_obs = self._process_observation(self.observation)
    
        # Debug information (optional, could be removed)
        #print(f"Full observation shape: {self.observation.shape}")
        #print(f"Processed observation shape: {self.final_obs.shape}")
    
        self.achieved_goal = np.concatenate((objects_pos, tool_pos))
        self.desired_goal = np.array(goal_pos)
 
    def _process_observation(self, observation):
        
        noise = np.random.randn(len(observation)) * 0.0001
        observation_with_noise = observation + noise
    
        
        padded_observation = np.zeros_like(observation_with_noise)
        padded_observation[:6] = observation_with_noise[:6]
        
        return padded_observation
 
 
    def my_task_done(self):
        
        c = (self.terminated == True or self.stepCounter > self.maxSteps)
        return c
 
 
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = np.zeros(1)
        
        end_effector_pos = achieved_goal[-3:]
        self.target_dist = goal_distance(end_effector_pos, desired_goal)
        
        # Define desired distance range
        self.min_distance = 0.2 # Minimum acceptable distance
        self.max_distance = 0.4  # Maximum acceptable distance
        heavy_penalty = 10    # Reduced heavy penalty
        
        # Reward for being within the acceptable distance range
        if self.min_distance <= self.target_dist <= self.max_distance:
            reward += 10  # Increased positive reward for staying within the range
        else:
            # Penalize if the distance is outside the acceptable range
            reward -= heavy_penalty
        
        # Reward based on distance from target
        reward += -self.target_dist * 5  # Less punitive
    
        # Reward for improvement in distance
        previous_distance = self.previous_target_dist if hasattr(self, 'previous_target_dist') else self.target_dist
        distance_difference = previous_distance - self.target_dist
        reward += distance_difference * 10  # Reward for improvement
        
        # Task completion
        if self.target_dist < self.min_distance:
            reward += 50  # Significant reward for reaching the goal
            self.terminated = True
    
        # Penalize collisions
        if self.check_collisions(): 
            reward -= 1  # Small penalty for collision
        
        # Store the current distance for the next step
        self.previous_target_dist = self.target_dist
    
        return reward
