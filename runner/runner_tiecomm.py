import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam,RMSprop
from modules.utils import merge_dict, multinomials_log_density
import time
from runner import Runner
from termcolor import cprint

import sys
import argparse

Transition = namedtuple('Transition', ('action_outs', 'actions', 'rewards', 'values', 'episode_masks', 'episode_agent_masks'))
God_Transition = namedtuple('God_Transition', ('god_action_out', 'god_action', 'god_reward', 'god_value', 'episode_masks',))


class RunnerTiecomm(Runner):
    def __init__(self, config, env, agent):
        super().__init__(config, env, agent)

        print("IT'S GOING THROUGH RunnerTiecomm.py")

        self.optimizer_agent_ac = RMSprop(self.agent.agent.parameters(), lr = 0.003, alpha=0.97, eps=1e-6)
        self.optimizer_god_ac = RMSprop(self.agent.god.parameters(), lr = 0.006, alpha=0.97, eps=1e-6)

        self.n_nodes = int(self.n_agents * (self.n_agents - 1) / 2)
        self.interval = self.args.interval




    def optimizer_zero_grad(self):
        self.optimizer_agent_ac.zero_grad()
        self.optimizer_god_ac.zero_grad()



    def optimizer_step(self):
        #nn.utils.clip_grad_norm_(self.params, max_norm=5.0)
        self.optimizer_agent_ac.step()
        self.optimizer_god_ac.step()


    def compute_grad(self, batch):
        log=dict()
        # cprint(type(batch), 'white', 'on_green')
        # print('(Transition, Transition)')
        # print(len(batch))
        # cprint(type(batch[1]), 'white', 'on_light_green')
        # print("God_Transition(''god_action_out', 'god_action', 'god_reward', 'god_value', 'episode_masks'')")
        # cprint(type(batch[1].god_action), 'white', 'on_yellow')
        # print(batch[1].god_action)

        # cprint(batch, 'white', 'on_red')
        reformat = self.reformat_batch(batch[0])
        batch = (reformat, batch[1])

        agent_log = self.compute_agent_grad(batch[0])
        god_log = self.compute_god_grad(batch[1])

        merge_dict(agent_log, log)
        merge_dict(god_log, log)
        return log


    def reformat_batch(self, batch):

        # cprint(type(batch), 'white', 'on_green')
        # print('(Transition, Transition)')
        # print(len(batch))
        # cprint(type(batch[0]), 'white', 'on_light_green')
        # print("Transition('action_outs', 'actions', 'rewards', 'values', 'episode_masks', 'episode_agent_masks')")
        # cprint(type(batch[0].actions), 'white', 'on_yellow')
        # print(batch[0].actions)


        # Extract actions from batch (tuple of dictionaries)
        action_dicts = batch.actions  # This is a tuple of dictionaries
        
        # Determine agent keys dynamically
        agent_keys = sorted(action_dicts[0].keys())  # Ensure consistent ordering
        
        # Convert each dictionary to a NumPy array, maintaining the tuple structure
        formatted_actions = [np.array([[action_dict[agent] for agent in agent_keys] for action_dict in action_dicts])]  # List of length 1
        
        # Create a new batch with formatted actions
        new_batch = Transition(
            action_outs=batch.action_outs,
            actions=tuple(formatted_actions),  # Ensure it remains a tuple
            rewards=batch.rewards,
            values=batch.values,
            episode_masks=batch.episode_masks,
            episode_agent_masks=batch.episode_agent_masks
        )
        
        return new_batch





    def train_batch(self, batch_size): #THIS IS THE ONE BEING USED 
        batch_data, batch_log = self.collect_batch_data(batch_size)
        self.optimizer_zero_grad()
        train_log = self.compute_grad(batch_data)
        merge_dict(batch_log, train_log)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= batch_log['num_steps']
        self.optimizer_step()
        return train_log





    def collect_batch_data(self, batch_size):
        batch_data = []
        god_batch_data = []
        batch_log = dict()
        num_episodes = 0
        cprint('batch_size is: ' + str(batch_size), 'white', 'on_red')
        while len(batch_data) < batch_size:
            cprint('length of batch_data is: ' + str(len(batch_data)), 'white', 'on_red')
            episode_data,episode_log = self.run_an_episode()
            batch_data += episode_data[0]
            god_batch_data += episode_data[1]
            merge_dict(episode_log, batch_log)
            num_episodes += 1

        batch_log['num_episodes'] = num_episodes
        batch_log['num_steps'] = len(batch_data)
        batch_data = Transition(*zip(*batch_data))
        god_batch_data = God_Transition(*zip(*god_batch_data))

        # cprint("collect_batch_data complete", "white", 'on_red')
        return (batch_data, god_batch_data), batch_log


    def run_an_episode(self):

        log = dict()

        memory = []
        god_memory = []

        self.reset()
        obs = self.env.get_obs()
        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
        graph = self.env.get_graph()

        # cprint("obs: " + str(len(obs[0])), 'red') 
        # cprint("obs_tensor shape: " + str(obs_tensor.shape), 'red')

        god_action_out, god_value = self.agent.god(obs_tensor, graph)
        god_action = self.choose_action(god_action_out)
        god_action = [god_action[0].reshape(1)]
        g, set = self.agent.graph_partition(graph, god_action)

        god_reward_list = []
        god_reward = np.zeros(1)

        step = 1
        num_group = 0
        episode_return = 0
        done = False
        while not done and step <= self.args.episode_length:
            cprint("Step: " + str(step), 'white', 'on_blue')
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)

            if step % self.interval == 0:
                graph = self.env.get_graph()
                god_action_out, god_value = self.agent.god(obs_tensor, graph)
                god_action = self.choose_action(god_action_out)
                god_action = [god_action[0].reshape(1)]
                g, set = self.agent.graph_partition(graph, god_action)


            after_comm = self.agent.communicate(obs_tensor, g, set)
            action_outs, values = self.agent.agent(after_comm)
            actions = self.choose_action(action_outs)
            actions = self.env.reformat_action(actions)

            # rewards, done, env_info = self.env.step(actions)
            obs, rewards_pre, terminations, truncations, info = self.env.step(actions)
            done = self.env.done_adapter(terminations=terminations, truncations=truncations)
            env_info = self.env.info_adapter(terminations=terminations, truncations=truncations, info=info)
            
            rewards = self.env.reformat_rewards(rewards_pre)
            god_reward_list.append(np.mean(rewards).reshape(1))

            if step % self.interval == 0:
                god_reward = np.mean(god_reward_list).reshape(1)
                god_reward_list = []


            next_obs = self.env.get_obs()

            done = done or step == self.args.episode_length

            episode_mask = np.ones(rewards.shape)
            episode_agent_mask = np.ones(rewards.shape)
            god_episode_mask = np.ones(1)
            if done:
                episode_mask = np.zeros(rewards.shape)
                god_episode_mask = np.zeros(1)
            elif 'completed_agent' in env_info:
                episode_agent_mask = 1 - np.array(env_info['completed_agent']).reshape(-1)


            trans = Transition(action_outs, actions, rewards, values, episode_mask, episode_agent_mask)
            memory.append(trans)


            if step % self.interval == 0:
                god_trans = God_Transition(god_action_out, god_action, god_reward, god_value, god_episode_mask)
                god_memory.append(god_trans)

            obs = next_obs
            episode_return += int(np.sum(rewards))
            step += 1
            num_group += len(set[1])


        log['episode_return'] = episode_return
        log['episode_steps'] = [step-1]
        log['num_groups'] = num_group / (step - 1)

        if 'num_collisions' in env_info:
            log['num_collisions'] = int(env_info['num_collisions'])

        # if self.args.env == 'tj':
        #     merge_dict(self.env.get_stat(),log)

        return (memory, god_memory), log





    def compute_god_grad(self, batch):

        log = dict()
        batch_size = len(batch.god_value)
        n = 1

        rewards = torch.Tensor(np.array(batch.god_reward))
        actions = torch.Tensor(np.array(batch.god_action))
        actions = actions.transpose(1, 2).view(-1, n, 1)


        episode_masks = torch.Tensor(np.array(batch.episode_masks))

        values = torch.cat(batch.god_value, dim=0)
        action_outs = torch.stack(batch.god_action_out, dim=0)



        returns = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)
        prev_returns = 0


        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + self.args.gamma * prev_returns * episode_masks[i]
            prev_returns = returns[i].clone()


        for i in reversed(range(batch_size)):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()


        log_p_a = [action_outs.view(-1, 10)]
        actions = actions.contiguous().view(-1, 1)
        log_prob = multinomials_log_density(actions, log_p_a)
        action_loss = -advantages.view(-1) * log_prob.squeeze()
        actor_loss = action_loss.sum()


        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        critic_loss = value_loss.sum()


        total_loss = actor_loss + self.args.value_coeff * critic_loss
        total_loss.backward()

        log['god_action_loss'] = actor_loss.item()
        log['god_value_loss'] = critic_loss.item()
        log['god_total_loss'] = total_loss.item()

        return log