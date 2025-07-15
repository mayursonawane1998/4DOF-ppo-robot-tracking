#PPO Code#
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
 
# Memory class for storing experiences during training
class Memory:
    def __init__(self):
        self.actions = []         # Stores actions taken
        self.states = []          # Stores states encountered
        self.logprobs = []        # Stores log probabilities of actions
        self.rewards = []         # Stores rewards obtained
        self.is_terminals = []    # Stores terminal state flags
    
    # Clears the stored experiences after each update
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
 
 
# Custom Actor-Critic model combining policy (actor) and value (critic) networks
class CustomActorCritic(nn.Module):
    def __init__(self, device, state_dim, emb_size, action_dim, action_std):
        self.device = device
        super(CustomActorCritic, self).__init__()
        
        # Actor network definition: maps states to action means
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, emb_size),
                nn.Tanh(),
                nn.Linear(emb_size, emb_size),
                nn.Tanh(),
                # nn.Linear(emb_size, emb_size),
                # nn.Tanh(),
                nn.Linear(emb_size, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, emb_size),
                nn.Tanh(),
                nn.Linear(emb_size, emb_size),
                nn.Tanh(),
                # nn.Linear(emb_size, emb_size),
                # nn.Tanh(),
                nn.Linear(emb_size, 1)
                )
        
        # Action variance (used for exploration in continuous action spaces)
        self.action_var = torch.full((action_dim,), action_std*action_std).to(self.device)
        
    # Forward pass is not used directly in this model
    def forward(self):
        raise NotImplementedError
    
    # Function to determine the action to take given a state
    def act(self, state, memory):
        print("State shape in act method:", state.shape)
        action_mean = self.actor(state)  # Get the mean of the action distribution
        print("Actor Output Shape (Action Mean):", action_mean.shape)
        cov_mat = torch.diag(self.action_var).to(self.device)  # Covariance matrix for the action distribution
        
        # Create a multivariate normal distribution and sample an action
        distribution = MultivariateNormal(action_mean, cov_mat)
        action = distribution.sample()
        print("Action Sampled Shape:", action.shape)
        action_logprob = distribution.log_prob(action)
        
        # Store the state, action, and log probability of the action in memory
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
 
        print("Action Shape after sampling:", action.shape)  # Ensure action shape matches action_dim
        
        # Return the sampled action, detached from the computation graph
        return action.detach()
    
    # Function to evaluate actions taken in the past
    def evaluate(self, state, action): 
        print("State shape in evaluate method:", state.shape)
        action_mean = self.actor(state)  # Get the mean of the action distribution
        print("Actor Output Shape in evaluate method (Action Mean):", action_mean.shape)
        
        # Covariance matrix for the action distribution
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        # Create a multivariate normal distribution
        distribution = MultivariateNormal(action_mean, cov_mat)
        
        # Get the log probabilities, entropy, and state value
        action_logprobs = distribution.log_prob(action)
        distribution_entropy = distribution.entropy()
        state_value = self.critic(state)
        print("State Value Shape in evaluate method:", state_value.shape)
        
        return action_logprobs, torch.squeeze(state_value), distribution_entropy
 
 
# PPO (Proximal Policy Optimization) agent
class PPO:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.device = self.args.device
 
        self.state_dim = self.env.observation_space.shape[0]  # State dimension
        self.action_dim = self.env.action_space.shape[0]      # Action dimension
        
        # Policy network (current)
        self.policy = CustomActorCritic(self.device ,self.state_dim, self.args.emb_size, self.action_dim, self.args.action_std).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr, betas=self.args.betas)
        
        # Old policy network (used for stable updates)
        self.policy_old = CustomActorCritic(self.device, self.state_dim, self.args.emb_size, self.action_dim, self.args.action_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Loss function for the critic
        self.MseLoss = nn.MSELoss()
    
    # Function to select action using the old policy network
    def select_action(self, state, memory):
        print("State before preprocessing:", state.shape)
        # Convert state to float tensor and move to device
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        print("FloatTensor state shape:", state.shape)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    # Function to update the policy using collected experiences
    def update(self, memory):
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize the rewards
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards.float().squeeze()
        
        # Convert list of tensors to a single tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()
 
        print("Old states shape in update:", old_states.shape)
        print("Old actions shape in update:", old_actions.shape)
        print("Old logprobs shape in update:", old_logprobs.shape)
        
        # Optimize policy for K epochs
        for _ in range(self.args.K_epochs):
            # Evaluate old actions and values using the current policy
            logprobs, state_values, distribution_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Calculate the ratio (new policy / old policy)
            ratios = torch.exp(logprobs - old_logprobs.detach())
 
            # Calculate the surrogate loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + \
                    + self.args.loss_value_c*self.MseLoss(state_values, rewards) + \
                    - self.args.loss_entropy_c*distribution_entropy
            
            # Perform a gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
