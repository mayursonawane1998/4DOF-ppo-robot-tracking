import os
import argparse
import numpy as np
import torch
import random
from ppo import PPO, Memory
from gym_env import RobotGymEnv
 
title = 'PyBullet Robot'
 
def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    # env
    arg('--render', action='store_true', default=False, help='render the environment')
    arg('--randObjPos', action='store_true', default=True, help='randomize object position')
    arg('--mel', type=int, default=90, help='max episode length')
    arg('--repeat', type=int, default=1, help='repeat action')
    arg('--task', type=int, default=0, help='task to learn: 0 move')
    arg('--lp', type=float, default=0.001, help='learning parameter for task')
    # train:
    arg('--seed', type=int, default=987, help='random seed')
    arg('--emb_size',   type=int, default=128, help='embedding size')
    arg('--solved_reward', type=int, default=0, help='stop training if avg_reward > solved_reward')
    arg('--log_interval', type=int, default=10, help='interval for log')
    arg('--save_interval', type=int, default=25, help='interval for saving model')
    arg('--max_episodes', type=int, default=2000, help='max training episodes')
    arg('--update_timestep', type=int, default=1000, help='update policy every n timesteps')
    arg('--action_std', type=float, default=1.0, help='constant std for action distribution (Multivariate Normal)')
    arg('--K_epochs', type=int, default=15, help='update policy for K epochs')
    arg('--eps_clip', type=float, default=0.15, help='clip parameter for PPO')
    arg('--gamma', type=float, default=0.99, help='discount factor')
    arg('--lr', type=float, default=1e-3, help='parameters for Adam optimizer')
    arg('--betas', type=float, default=(0.9, 0.999), help='')
    arg('--loss_entropy_c', type=float, default=0.02, help='coefficient for entropy term in loss')
    arg('--loss_value_c', type=float, default=0.5, help='coefficient for value term in loss')
    arg('--save_dir', type=str, default='saved_rl_models(0.00001)6d5', help='path to save the models')
    arg('--cuda', dest='cuda', action='store_true', default=False, help='Use cuda to train model')
    arg('--device_num', type=str, default=0, help='GPU number to use')
 
    args = parser.parse_args()
    return args
 
args = get_args()  # Holds all the input arguments
 
np.set_printoptions(precision=2)
torch.set_printoptions(profile="full", precision=2)
 
# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_B = '\033[34m'
CP_Y = '\033[33m'
CP_C = '\033[0m'
 
def write_file(filepath, data, mode):
    with open(filepath, mode) as f:
        f.write(data)
 
args.filename_tl = 'training_log.txt'  # log file
 
args.device = torch.device('cuda:'+str(args.device_num) if args.cuda else 'cpu')
print('Using device:', 'cuda' if args.cuda else 'cpu', ', device number:', args.device_num, ', GPUs in system:', torch.cuda.device_count())
 
def main():
    args.env_name = title
    print(CP_G + 'Environment name:', args.env_name, ''+CP_C)
 
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    env = RobotGymEnv(renders=args.render, maxSteps=args.mel, 
                      actionRepeat=args.repeat, task=args.task, randObjPos=args.randObjPos, learning_param=args.lp)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
 
    memory = Memory()
    ppo = PPO(args, env)
 
    # logging variables
    running_reward = 0
    time_step = 0
 
    # List to store the results
    episode_rewards = []
 
    # Initialize lists to store target distances for each episode
    target_distances_detection = []
    target_distances_simulation = []
 
    
 
    print('Starting training with learning_param:', args.lp)
    for i_episode in range(1, args.max_episodes + 1):
        try:
            state = env.reset()
            episode_reward = 0
 
            # Initialize lists to store distances for the current episode
            episode_detection_distances = []
            episode_simulation_distances = []
 
            for t in range(args.mel):
                time_step += 1
 
                action = ppo.select_action(state, memory)
                state, reward, done, _ = env.step(action)
 
                # Append the detection-based target distance if valid
                if env.cube_detected and env.corrected_target_dist is not None:
                    episode_detection_distances.append(env.corrected_target_dist)
 
                # Append the simulation-based target distance if valid
                if env.target_dist1 is not None:
                    episode_simulation_distances.append(env.target_dist1)
 
                # Process rewards based on the modified reward function
                reward = np.array([reward])
                reward = torch.tensor(reward, dtype=torch.float32).to(args.device)
                
                # Saving reward and is_terminals:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
 
                # Learning step
                if time_step % args.update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()
                    time_step = 0
 
                episode_reward += reward.item()
 
                if done:
                    break
 
            # After the episode ends, store the episode's average distances
            if episode_detection_distances:
                target_distances_detection.append(np.mean(episode_detection_distances))
            if episode_simulation_distances:
                target_distances_simulation.append(np.mean(episode_simulation_distances))
 
            # Update policy if there are enough rewards in memory
            if len(memory.rewards) >= args.update_timestep:
                ppo.update(memory)
                memory.clear_memory()
            
            # logging
            running_reward += episode_reward
            
            if i_episode % args.log_interval == 0:
                avg_reward = int(running_reward / args.log_interval)
                print(f'Episode {i_episode} \t Avg reward: {avg_reward}')
                episode_rewards.append(avg_reward)
                running_reward = 0
 
            # Save model every save_interval episodes
            if i_episode % args.save_interval == 0:
                torch.save(ppo.policy.state_dict(), os.path.join(args.save_dir, f'model_epoch_{int(i_episode / args.save_interval)}.pth'))
 
        except Exception as e:
            print(f"Error in episode {i_episode}: {e}")
            env.close()
            break
 
    env.close()
 
if __name__ == '__main__':
    main()
