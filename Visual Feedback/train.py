import os
import argparse
from datetime import datetime
import numpy as np
import torch
import random
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend to avoid issues with 'tkinter' for plotting on servers
import matplotlib.pyplot as plt
from ppo import PPO, Memory  # Import PPO agent and Memory class
from gym_env import RobotGymEnv  # Import custom Gym environment
 
# Title for the experiment
title = 'PyBullet Mayur Robot'
 
# Function to parse command-line arguments
def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    # Environment parameters
    arg('--render', action='store_true', default=False, help='render the environment')
    arg('--randObjPos', action='store_true', default=True, help='randomize object position')
    arg('--mel', type=int, default=460, help='max episode length')
    arg('--repeat', type=int, default=1, help='repeat action')
    arg('--task', type=int, default=0, help='task to learn: 0 move')
    arg('--lp', type=float, default=0.0001, help='learning parameter for task')
    # Training parameters
    arg('--seed', type=int, default=987, help='random seed')
    arg('--emb_size',   type=int, default=128, help='embedding size')
    
    arg('--log_interval', type=int, default=10, help='interval for log')
    arg('--save_interval', type=int, default=100, help='interval for saving model')
    arg('--max_episodes', type=int, default=10000, help='max training episodes')
    arg('--update_timestep', type=int, default=1100, help='update policy every n timesteps')
    arg('--action_std', type=float, default=1.0, help='constant std for action distribution (Multivariate Normal)')
    arg('--K_epochs', type=int, default=23, help='update policy for K epochs')
    arg('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
    arg('--gamma', type=float, default=0.99, help='discount factor')
    arg('--lr', type=float, default=1e-3, help='parameters for Adam optimizer')
    arg('--betas', type=float, default=(0.9, 0.999), help='betas for Adam optimizer')
    arg('--loss_entropy_c', type=float, default=0.02, help='coefficient for entropy term in loss')
    arg('--loss_value_c', type=float, default=0.5, help='coefficient for value term in loss')
    arg('--save_dir', type=str, default='saved_rl_models(0.00001)stdCamera', help='path to save the models')
    arg('--cuda', dest='cuda', action='store_true', default=False, help='Use cuda to train model')
    arg('--device_num', type=str, default=0, help='GPU number to use')
 
    args = parser.parse_args()
    return args
 
# Parse the arguments
args = get_args()  # Holds all the input arguments
 
# Set the precision for printing numpy and torch arrays
np.set_printoptions(precision=2)
torch.set_printoptions(profile="full", precision=2)
 
# Filename for training log
args.filename_tl = 'training_log.txt'  # log file
 
# Set the device (CPU or GPU)
args.device = torch.device('cuda:'+str(args.device_num) if args.cuda else 'cpu')
print('Using device:', 'cuda' if args.cuda else 'cpu', ', device number:', args.device_num, ', GPUs in system:', torch.cuda.device_count())
 
# Function to smooth data using a moving average
def smooth_data(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
 
# Function to downsample data by a given factor
def downsample_data(data, factor=10):
    return data[::factor]
 
# Function to save the plot of training rewards
def save_plot(episode_rewards, save_dir, suffix=''):
    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards)
    plt.xlabel('Episodes (x{})'.format(args.log_interval))
    plt.ylabel('Average Reward')
    plt.title('Training Rewards')
    save_path = os.path.join(save_dir, f'training_rewards{suffix}.png')
    plt.savefig(save_path)
    plt.close()
 
# Function to save the plot of goal vs end-effector positions over time
def save_position_plot(goal_positions, end_effector_positions, save_dir, suffix=''):
    plt.figure(figsize=(10, 6))  # Start with a fresh figure
 
    # Convert lists to arrays
    goal_positions = np.array(goal_positions)
    end_effector_positions = np.array(end_effector_positions)
 
    # Downsample the data for clarity
    downsampled_goal_positions = downsample_data(goal_positions, factor=100)
    downsampled_end_effector_positions = downsample_data(end_effector_positions, factor=100)
 
    # Calculate the overall (Euclidean) distance of the goal and end-effector positions
    goal_distances = np.linalg.norm(downsampled_goal_positions, axis=1)
    end_effector_distances = np.linalg.norm(downsampled_end_effector_positions, axis=1)
 
    # Plot the overall distances
    plt.plot(goal_distances, label='Goal Position', color='green', alpha=0.7)
    plt.plot(end_effector_distances, label='End-Effector Position', color='blue', linestyle='--', alpha=0.7)
 
    # Set labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Distance')  
    plt.title('Goal vs End-Effector Position Over Time')
 
    # Add the legend
    plt.legend()
 
    # Display grid and save the plot
    plt.grid(True)
    save_path = os.path.join(save_dir, f'position_comparison{suffix}.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to prevent overlap with the next one
 
# Function to save the plot of smoothed positional errors during training
def save_error_plot(mean_errors, std_errors, save_dir, suffix=''):
    # Apply smoothing to the mean and standard deviation of positional errors
    smoothed_mean_errors = smooth_data(mean_errors)
    smoothed_std_errors = smooth_data(std_errors)
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(smoothed_mean_errors)), smoothed_mean_errors, yerr=smoothed_std_errors, fmt='-o', label='Smoothed Positional Error (Mean ± Std)', ecolor='lightgray', elinewidth=2, capsize=3)
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Positional Error (Mean ± Std)')
    plt.title('Smoothed Positional Errors During Training')
    plt.legend()
    save_path = os.path.join(save_dir, f'smoothed_positional_errors{suffix}.png')
    plt.savefig(save_path)
    plt.close()
 
# Function to save the plot of average target distance over episodes
def save_target_distance_over_episodes_plot(all_target_distances, save_dir, suffix=''):
    # all_target_distances should be a list of lists, where each inner list contains the target distances for one episode
 
    # Calculate average target distance per episode
    avg_target_distances = [np.mean(distances) for distances in all_target_distances]
 
    plt.figure(figsize=(10, 6))
    plt.plot(avg_target_distances, label='Average Target Distance per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Distance (End-Effector to Cube)')
    plt.title('Average Target Distance Over Episodes')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, f'target_distance_over_episodes{suffix}.png')
    plt.savefig(save_path)
    plt.close()
 
# Main function to train the PPO agent
def main():
    args.env_name = title
    print('Environment name:', args.env_name)
 
    # Create directory for saving models if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Initialize the custom robot environment
    env = RobotGymEnv(renders=args.render, maxSteps=args.mel, 
                    actionRepeat=args.repeat, task=args.task, randObjPos=args.randObjPos, learning_param=args.lp)
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
 
    # Initialize memory and PPO agent
    memory = Memory()
    ppo = PPO(args, env)
 
    # Logging variables
    running_reward = 0
    time_step = 0
 
    # Lists to store positional data and metrics for plotting
    episode_rewards = []
    all_goal_positions = []
    all_end_effector_positions = []
    mean_errors = []
    std_errors = []
    all_target_distances = []  # List to collect target distances for all episodes
 
    print('Starting training with learning_param:', args.lp)
    for i_episode in range(1, args.max_episodes + 1):
        try:
            state = env.reset()  # Reset environment at the start of each episode
            episode_reward = 0
 
            for t in range(args.mel):
                time_step += 1
 
                action = ppo.select_action(state, memory)  # Select action using the PPO agent
                state, reward, done, _ = env.step(action)  # Take a step in the environment
                
                # Process rewards based on the modified reward function
                reward = np.array([reward])  # Ensure reward is in numpy array format
                reward = torch.tensor(reward, dtype=torch.float32).to(args.device)  # Convert reward to tensor
                
                # Save reward and terminal state
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
 
                # Update policy if it's time
                if time_step % args.update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()
                    time_step = 0
                episode_reward += reward.item()  # Accumulate episode reward
 
                if done:
                    break
            
            # Update policy if there are enough rewards in memory
            if len(memory.rewards) >= args.update_timestep:
                ppo.update(memory)
                memory.clear_memory()
            
            # Logging
            running_reward += episode_reward
            
            # Calculate positional error metrics for the episode
            mean_error = np.mean(env.positional_errors)
            std_error = np.std(env.positional_errors)
            mean_errors.append(mean_error)
            std_errors.append(std_error)
 
            # Store the positions for plotting
            all_goal_positions.extend(env.goal_positions)
            all_end_effector_positions.extend(env.end_effector_positions)
 
            # Store the target distances for this episode
            all_target_distances.append(env.target_distances.copy())
 
            # Log the results every log_interval episodes
            if i_episode % args.log_interval == 0:
                avg_reward = int(running_reward / args.log_interval)
                print(f'Episode {i_episode} \t Avg reward: {avg_reward} \t Mean Error: {mean_error:.4f} \t Std Error: {std_error:.4f}')
                episode_rewards.append(avg_reward)
                running_reward = 0
 
            # Save plots periodically every 500 episodes
            if i_episode % 500 == 0:
                save_plot(episode_rewards, args.save_dir, suffix=f'_{i_episode}')
                save_position_plot(all_goal_positions, all_end_effector_positions, args.save_dir, suffix=f'_{i_episode}')
                save_error_plot(mean_errors, std_errors, args.save_dir, suffix=f'_{i_episode}')
                save_target_distance_over_episodes_plot(all_target_distances, args.save_dir, suffix=f'_{i_episode}')
 
            # Save model periodically every save_interval episodes
            if i_episode % args.save_interval == 0:
                torch.save(ppo.policy.state_dict(), os.path.join(args.save_dir, f'model_epoch_{int(i_episode / args.save_interval)}.pth'))
 
        except Exception as e:
            print(f"Error in episode {i_episode}: {e}")
            env.close()  # Close the environment in case of an error
            break
 
    # Final save of plots at the end of training
    save_plot(episode_rewards, args.save_dir)
    save_position_plot(all_goal_positions, all_end_effector_positions, args.save_dir)
    save_error_plot(mean_errors, std_errors, args.save_dir)
    save_target_distance_over_episodes_plot(all_target_distances, args.save_dir)  # Final save
    env.close()  # Close the environment
 
# Entry point of the script
if __name__ == '__main__':
    main()
 
