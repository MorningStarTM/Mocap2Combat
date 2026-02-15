import os
import time
import json
import numpy as np
from datetime import datetime
import torch
import csv
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from src.mocap2combat.utils.logger import logger
from safetensors.torch import save_file, load_file
from src.mocap2combat.network.mlp import MLP
from src.mocap2combat.network.sequence import RNNNet
from typing import List, Tuple, Optional




class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def __len__(self):
        return len(self.rewards)
    



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dims, n_actions_per_dim=4):
        """
        action_dims = 22 for MultiDiscrete([4]*22)
        n_actions_per_dim = 4
        """
        super().__init__()
        self.action_dims = action_dims
        self.n_actions_per_dim = n_actions_per_dim

        # actor outputs (action_dims * n_actions_per_dim) logits
        self.actor = MLP(
            state_dim,
            hidden_sizes=[512, 256, 64, 64],
            output_size=action_dims * n_actions_per_dim,
            activation=nn.Tanh
        )

        self.critic = MLP(
            state_dim,
            hidden_sizes=[512, 256, 128, 64],
            output_size=1,
            activation=nn.Tanh
        )

    def _dist(self, state):
        # logits: (B, action_dims*n_actions_per_dim) or (action_dims*n_actions_per_dim,)
        logits = self.actor(state)

        if logits.dim() == 1:
            logits = logits.view(self.action_dims, self.n_actions_per_dim)          # (22,4)
        else:
            logits = logits.view(-1, self.action_dims, self.n_actions_per_dim)      # (B,22,4)

        return logits

    def act(self, state):
        logits = self._dist(state)  # (22,4)
        dists = [Categorical(logits=logits[i]) for i in range(self.action_dims)]

        actions = torch.stack([d.sample() for d in dists], dim=0)                  # (22,)
        logprob = torch.stack([d.log_prob(a) for d, a in zip(dists, actions)]).sum()
        entropy = torch.stack([d.entropy() for d in dists]).sum()

        state_val = self.critic(state)

        return actions.detach(), logprob.detach(), state_val.detach(), entropy.detach()

    def evaluate(self, states, actions):
        """
        states:  (T,B?) usually (T, state_dim) in your code => (N,state_dim)
        actions: (N, action_dims)  long
        """
        logits = self._dist(states)  # (N,22,4)

        # build per-dim distributions across the batch
        # logits[:, i, :] is (N,4)
        logprob_list = []
        entropy_list = []

        for i in range(self.action_dims):
            dist_i = Categorical(logits=logits[:, i, :])
            logprob_list.append(dist_i.log_prob(actions[:, i]))
            entropy_list.append(dist_i.entropy())

        action_logprobs = torch.stack(logprob_list, dim=1).sum(dim=1)  # (N,)
        dist_entropy   = torch.stack(entropy_list, dim=1).sum(dim=1)   # (N,)

        state_values = self.critic(states).squeeze(-1)                 # (N,)

        return action_logprobs, state_values, dist_entropy

    



class PPO:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.gamma = self.config['gamma']
        self.eps_clip = self.config['eps_clip']
        self.K_epochs = self.config['K_epochs']
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.config['lr_actor']},
                        {'params': self.policy.critic.parameters(), 'lr': self.config['lr_critic']}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def get_buffer_size(self):
        return len(self.buffer)

    def select_action(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)

            action_vec, action_logprob, state_val, _ = self.policy_old.act(state_t)

            self.buffer.states.append(state_t)
            self.buffer.actions.append(action_vec)        # IMPORTANT: store vector (22,)
            self.buffer.logprobs.append(action_logprob)   # scalar
            self.buffer.state_values.append(state_val)    # (1,) or scalar

            # env.step expects array-like of shape (22,)
            return action_vec.cpu().numpy(), action_logprob, state_val


    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(self.device)        # (N, state_dim)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(self.device)      # (N, 22)
        old_actions = old_actions.long()

        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(self.device)    # (N,)
        old_state_values = torch.stack(self.buffer.state_values, dim=0).detach().to(self.device).squeeze(-1)  # (N,)


        # calculate advantages
        #logger.info(f"rewards shape: {rewards.shape}, old_state_values shape: {old_state_values.shape}")
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path, filename="ppo_checkpoint.pth"):
        torch.save(self.policy_old.state_dict(), os.path.join(checkpoint_path, filename))
   
    def load(self, checkpoint_path, filename="ppo_checkpoint.pth"):
        file = os.path.join(checkpoint_path, filename)
        self.policy_old.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))


    def save_safetensors(self, checkpoint_path, filename="ppo_policy.safetensors", meta_filename="ppo_meta.json"):
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save policy_old (the one you actually use for acting)
        state = self.policy_old.state_dict()

        # safetensors needs a plain {str: Tensor} dict (CPU tensors are safest)
        tensor_dict = {k: v.detach().cpu().contiguous() for k, v in state.items()}

        save_file(tensor_dict, os.path.join(checkpoint_path, filename))

        # Optional: store non-tensor metadata separately (JSON)
        meta = {
            "state_dim": None,
            "action_dim": None,
            "gamma": float(self.gamma),
            "eps_clip": float(self.eps_clip),
            "K_epochs": int(self.K_epochs),
            "lr_actor": float(self.config.get("lr_actor", 0.0)),
            "lr_critic": float(self.config.get("lr_critic", 0.0)),
        }
        with open(os.path.join(checkpoint_path, meta_filename), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_safetensors(self, checkpoint_path, filename="ppo_policy.safetensors", strict=True):
        file_path = os.path.join(checkpoint_path, filename)

        # load_file returns CPU tensors
        tensor_dict = load_file(file_path)

        # Load into both policies (like your torch.load version)
        self.policy_old.load_state_dict(tensor_dict, strict=strict)
        self.policy.load_state_dict(tensor_dict, strict=strict)

        # Make sure models are on the right device
        self.policy_old.to(self.device)
        self.policy.to(self.device)

    
    def decay_lr(self, gamma: float, min_lr: float = 1e-6):
        """
        Multiply actor/critic learning rates by gamma (clamped to min_lr).
        Your optimizer has 2 param_groups:
        group 0 -> actor lr (lr_actor)
        group 1 -> critic lr (lr_critic)
        Returns dict of new lrs for logging.
        """
        if self.optimizer is None:
            return {}

        # Safety: if someone changes param_groups ordering later, this still works
        out = {}

        # group 0: actor
        if len(self.optimizer.param_groups) >= 1:
            pg0 = self.optimizer.param_groups[0]
            pg0["lr"] = max(min_lr, pg0["lr"] * gamma)
            out["actor_lr"] = pg0["lr"]

        # group 1: critic
        if len(self.optimizer.param_groups) >= 2:
            pg1 = self.optimizer.param_groups[1]
            pg1["lr"] = max(min_lr, pg1["lr"] * gamma)
            out["critic_lr"] = pg1["lr"]

        return out