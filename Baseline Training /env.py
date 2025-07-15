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
from collections import namedtuple
from attrdict import AttrDict
import cv2
 
# Define paths to URDF files for the robot and the cube object
ROBOT_URDF_PATH = r"C:\Users\Ankit\myenv_robot\Lib\site-packages\pybullet_data\Assem3\urdf\Assem31.urdf"
CUBE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "cube1.urdf")
 
# Calculate the Euclidean distance between two 3D points
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)
 
# Calculate the Euclidean distance between two 2D points (x, y)
def goal_distance2d(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)
 
# Custom Gym environment for the Robot
class RobotGymEnv(gym.Env):
    def __init__(self,
                 camera_attached=False,
                 actionRepeat=1,
                 renders=False,
                 maxSteps=150,
                 randObjPos=False,
                 task=0, # target task number
                 learning_param=0):
 
        self.renders = renders
        self.actionRepeat = actionRepeat
        
        # Setup pybullet simulation
        if self.renders:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
 
        # Step the simulation and set gravity
        p.stepSimulation()
        time.sleep(1. / 240.)  # Adjust simulation speed as needed
        pybullet.setGravity(0,0,-10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setRealTimeSimulation(False)
        p.loadURDF("plane.urdf")
        
        # Setup robot arm (Robot):
        self.end_effector_index = 4  # Index of the end-effector link
        
        # Load the robot's URDF with self-collision enabled
        flags = pybullet.URDF_USE_SELF_COLLISION
        self.robot = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0.15], [0, 0, 0, 1], useFixedBase=True)
        self.num_joints = pybullet.getNumJoints(self.robot)
        self.control_joints = ["joint_1", "joint_2", "joint_3", "joint_4"]  # Controllable joints
        self.joint_type_list = ["REVOLUTE", "REVOLUTE", "REVOLUTE", "REVOLUTE", "FIXED"]  # Types of joints
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "Velocity", "controllable"])
 
        # Initialize joints
        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.robot, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                # Disable default position control for revolute joints
                pybullet.setJointMotorControl2(self.robot, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info
 
        # Load and position the object (cube) in the environment
        self.initial_obj_pos = [1.05, 0, 0.09]  # Initial object position
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
            "x": (1.05, 1.2),
            "y": (-0.7, 0.7),
            "z": (0.05, 1.1)  # Ensure it's above the ground
        }
     
        self._action_bound = 0.1  # Delta limits
        action_high = np.array([self._action_bound] * self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype='float32')
        self.reset()
        high = np.array([10]*self.observation.shape[0])
        self.observation_space = spaces.Box(-high, high, dtype='float32')
 
    # Set the joint angles of the robot
    def set_joint_angles(self, joint_angles):
        target_positions = joint_angles  # Extract target positions for controllable joints
        joint_indices = [self.joints[name].id for name in self.control_joints]  # Get joint indices
 
        pybullet.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=joint_indices,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=target_positions,
        )
 
    # Get the current joint angles of the robot
    def get_joint_angles(self):
        j = pybullet.getJointStates(self.robot, [1, 2, 3, 4])
        joints = [i[0] for i in j]
        return joints
    
    # Check for collisions in the environment
    def check_collisions(self):
        collisions = pybullet.getContactPoints()
        if len(collisions) > 0:
            return True
        return False
 
    # Calculate inverse kinematics to determine joint angles for a desired position and orientation
    def calculate_ik(self, position, orientation):
        quaternion = pybullet.getQuaternionFromEuler(orientation)
      
        lower_limits = [-1.57, -0.95, -1, -1.58]
        upper_limits = [1.57, 1.9, 1.484, 1.58]
        joint_ranges = [math.pi] * 4
        rest_poses = [0, 0, -0.719, -0.368]  # Rest pose of the robot
        joint_angles = pybullet.calculateInverseKinematics(
            self.robot, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*4, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )
        return joint_angles
       
    # Get the current position and orientation of the robot's end-effector
    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.robot, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)
 
    # Reset the environment
    def reset(self):
        self.stepCounter = 0
        self.terminated = False
        self.robot_or = [0, 1.5708, 0]
 
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
            # Set to default position if not randomizing
            self.initial_obj_pos = [1.05, 0, 0.09]
 
        pybullet.resetBasePositionAndOrientation(self.obj, self.initial_obj_pos, [0., 0., 0., 1.0])
 
        # Reset robot simulation and position:
        joint_angles = (0, 0, -0.719, -0.368)  # Default joint angles
        self.set_joint_angles(joint_angles) #
 
        # Step simulation to update the robot's state:
        for i in range(100):
            pybullet.stepSimulation()
 
        # Get observation and return:
        self.end_effector_trajectory = []
        self.cube_trajectory = []
        self.getExtendedObservation()
        return self.observation
    
    # Execute a step in the environment based on the given action
    def step(self, action):
        action = np.array(action)
        arm_action = action[0:self.action_dim-1].astype(float)  # dX, dY, dZ - range: [-1, 1]
    
        # Get current position:
        cur_p = self.get_current_pose()
        # Add delta position:
        new_p = np.array(cur_p[0]) + arm_action
        # Actuate: 
        joint_angles = self.calculate_ik(new_p, self.robot_or)  # XYZ and angles set to zero
        self.set_joint_angles(joint_angles)
        
        # Step simulation:
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
        
        reward = self.compute_reward(self.achieved_goal, self.desired_goal, None)
        done = self.my_task_done()
 
        info = {'is_success': False}
        if self.terminated == self.task:
            info['is_success'] = True
 
        self.stepCounter += 1
 
        return self.observation, reward, done, info
    
    # Get an extended observation combining tool and object positions
    def getExtendedObservation(self):
        tool_pos = self.get_current_pose()[0] 
        self.obj_pos, _ = pybullet.getBasePositionAndOrientation(self.obj)
        objects_pos = self.obj_pos       
        goal_pos = self.obj_pos
 
        self.observation = np.array(np.concatenate((tool_pos, objects_pos)))
        self.achieved_goal = np.array(np.concatenate((objects_pos, tool_pos)))
        self.desired_goal = np.array(goal_pos)
 
    # Determine if the task is done
    def my_task_done(self):
        return self.terminated == True or self.stepCounter > self.maxSteps
 
    # Compute the reward based on the distance between the achieved and desired goals
    # 4 REWARD FUNCTION
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = np.zeros(1)
        
        end_effector_pos = achieved_goal[-3:]
        self.target_dist = goal_distance(end_effector_pos, desired_goal)
        
        # Define desired distance range
        self.min_distance = 0.15  # Minimum acceptable distance
        self.max_distance = 0.35  # Maximum acceptable distance
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
 
    # 3RD REWARD FUNCTION
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = np.zeros(1)
        
        end_effector_pos = achieved_goal[-3:]
        self.target_dist = goal_distance(end_effector_pos, desired_goal)
        
        # Define desired distance range
        min_distance = 0.15 # Minimum acceptable distance
        max_distance = 0.35  # Maximum acceptable distance
        heavy_penalty = 10  # Heavy penalty for violating the distance range
    
        # Reward for being within the acceptable distance range
        if min_distance <= self.target_dist <= max_distance:
            reward += 10  # Positive reward for staying within the range
        else:
            # Penalize heavily if the distance is outside the acceptable range
            reward -= heavy_penalty
    
        # Reward for being close to the target
        reward += -self.target_dist * 5
        
        # Task 0: reach object
        if self.target_dist < min_distance:
            self.terminated = True
    
        # Penalize collisions
        if self.check_collisions(): 
            reward -= 1
    
        return reward
 
    #2ND REWARD FUNCTION 
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = np.zeros(1)
 
        end_effector_pos = achieved_goal[-3:]
            
        self.target_dist = goal_distance(end_effector_pos, desired_goal)
        # print(end_effector_pos, desired_goal, self.target_dist)
 
 
        reward += -self.target_dist * 10
            
        
        # Define a penalty for being too close to the target
        close_threshold = 0.15  # Example threshold distance
        penalty_factor = 5  # Example penalty factor
        
        # Penalty for getting too close to the target
        penalty_close = penalty_factor * max(0, close_threshold - self.target_dist)
        
        # Combine reward and pena
        reward -= penalty_close
 
        # task 0: reach object:
        if self.target_dist < close_threshold :
            self.terminated = True
            # print('Successful!')
 
       
        # check collisions:
        if self.check_collisions(): 
            reward += -10
            # print('Collision!')
 
        # print(target_dist, reward)
        # input()
 
        return reward
 
    #1ST REWARD FUNCTION
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = np.zeros(1)
 
        end_effector_pos = achieved_goal[-3:]
            
        self.target_dist = goal_distance(end_effector_pos, desired_goal)
        # print(end_effector_pos, desired_goal, self.target_dist)
 
        
 
        reward += -self.target_dist * 10
 
        # task 0: reach object:
        if self.target_dist < self.learning_param:# and approach_velocity < 0.05:
            self.terminated = True
            # print('Successful!')
 
        
 
        # check collisions:
        if self.check_collisions(): 
            reward += -1
            # print('Collision!')
 
        
        return reward
