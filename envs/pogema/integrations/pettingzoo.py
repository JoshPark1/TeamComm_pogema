import functools
from enum import Enum

import numpy as np
import wandb

from ..animation import AnimationConfig
from ..grid_config import GridConfig, EnvMode, RewardScheme
from pogema.envs import _make_pogema
from pettingzoo import ParallelEnv
import gymnasium

import xml.etree.ElementTree as ET
# import cairosvg
import os

import networkx as nx
import numpy as np
from termcolor import cprint
from gym.spaces import flatdim


def parallel_env(grid_config: GridConfig = GridConfig()):
    return PogemaParallel(grid_config)


class PogemaParallel(ParallelEnv):
    def state(self):
        anm = self.agent_name_mapping
        sz = self.pogema.grid_config.size
        s = np.zeros((2 * len(self.possible_agents) + 1) * sz * sz)

        # put global obstacle memory in state
        if self.full_global_state:
            s[:sz*sz] = np.ndarray.flatten(self.pogema.grid.global_obstacles).astype(float)
        else:
            s[:sz*sz] = np.ndarray.flatten((self.pogema.grid.globally_observed_mask & self.pogema.grid.global_obstacles)).astype(float)
            # cprint(self.pogema.grid, 'white', 'on_red')

        for agent in self.possible_agents:
            i = anm[agent]
            obs = self.pogema.get_agents_obs(agent_id=i, masked=(not self.full_global_state))

            # put the agent position component of observation in state, clamped to ignore -1 entries since global state knows all agent positions
            s[(2*i + 1) * sz * sz : (2*i + 2) * sz * sz] = np.clip(obs[sz*sz:2*sz*sz], 0, None)

            # put target position in global state
            s[(2*i + 2) * sz * sz : (2*i + 3) * sz * sz] = obs[2*sz*sz:]

        return s

    def __init__(self, grid_config: GridConfig, render_mode='ansi'):
        # cprint(grid_config, 'black', 'on_yellow')
        grid_config = GridConfig()
        # cprint(grid_config, 'black', 'on_magenta')
        # Environment type flags
        self.centralized = grid_config.env_mode == EnvMode.CENTRALIZED_CONCATENATED_CRITIC or grid_config.env_mode == EnvMode.CENTRALIZED_FULL_GLOBAL_CRITIC
        self.full_global_state = grid_config.env_mode == EnvMode.CENTRALIZED_FULL_GLOBAL_CRITIC
        self.comm_action = grid_config.env_mode == EnvMode.DECENTRALIZED_COMM

        # self.centralized = True
        # self.full_global_state = False
        # self.comm_action = False

        

        self.metadata = {'render_modes': ['ansi'], "name": "pogema"}
        self.render_mode = render_mode
        self.pogema = _make_pogema(grid_config)
        self.max_episode_steps = grid_config.max_episode_steps
        self.possible_agents = ["agents_" + str(r) for r in range(self.pogema.get_num_agents())]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.agents = None
        self.num_moves = None
        self.size = grid_config.size
        self.state_space = gymnasium.spaces.Box(-1.0, 1.0, shape=((2 * len(self.possible_agents) + 1) * self.size * self.size,))
        
        """JOSH CODE"""
        # self.graph = None

        # For testing policy
        self.render_steps = False

        # cprint(grid_config.observation_type, 'white','on_red')

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        assert agent in self.possible_agents
        return self.pogema.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        assert agent in self.possible_agents
        return self.pogema.action_space

    def render(self, mode="human"):
        # assert mode == 'human'
        self.pogema.save_animation('rendered_rollout/out-static.svg', AnimationConfig(static=True, save_every_idx_episode=None))
        self.pogema.save_animation('rendered_rollout/out-static-ego.svg', AnimationConfig(egocentric_idx=0, static=True))
        self.pogema.save_animation('rendered_rollout/out-static-no-agents.svg', AnimationConfig(show_agents=False, static=True))
        self.pogema.save_animation("rendered_rollout/out.svg")
        self.pogema.save_animation("rendered_rollout/out-ego.svg", AnimationConfig(egocentric_idx=0))

        # tree = ET.parse('out.svg')
        # root = tree.getroot()
        #
        #
        # animates = root.findall('.//{http://www.w3.org/2000/svg}animate')

        output_dir = 'frames'
        os.makedirs(output_dir, exist_ok=True)

        # # List to store WandB images for logging
        # wandb_images = []
        #
        # # Iterate through each <animate> tag or frame and log as images
        # for idx, animate in enumerate(animates):
        #     # Convert the current SVG frame to PNG using CairoSVG
        #     svg_frame_str = ET.tostring(root, encoding='unicode')
        #     png_filename = os.path.join(output_dir, f'frame_{idx}.png')
        #
        #     # Render the SVG string to PNG using CairoSVG
        #     cairosvg.svg2png(bytestring=svg_frame_str.encode('utf-8'), write_to=png_filename)
        #
        #     # Log the generated PNG to WandB
        #     wandb_images.append(wandb.Image(png_filename, caption=f"Frame {idx}"))
        #
        # # After all frames are rendered, log them as a sequence to WandB
        # wandb.log({"animation": wandb_images})
        return self.pogema.render()

    def reset(self, seed=None, options=None):
        anm = self.agent_name_mapping
        
        observations, infos = self.pogema.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent: observations[self.agent_name_mapping[agent]].astype(np.float32) for agent in self.agents}
        info = {agent: infos[anm[agent]] for agent in self.agents}

        if self.render_steps:
            self.render()
        return observations, info

    def step(self, actions):
        anm = self.agent_name_mapping

        actions = [actions[agent] if agent in actions else 0 for agent in self.possible_agents]
        if self.render_steps:
            print('Action:', actions)
        observations, rewards, terminated, truncated, infos = self.pogema.step(actions)
        d_observations = {agent: observations[anm[agent]].astype(np.float32) for agent in
                          self.possible_agents}
        d_rewards = {agent: rewards[anm[agent]] for agent in self.possible_agents}
        d_terminated = {agent: terminated[anm[agent]] for agent in self.possible_agents}
        d_truncated = {agent: truncated[anm[agent]] for agent in self.possible_agents}
        d_infos = {agent: infos[anm[agent]] for agent in self.possible_agents}
        d_infos['graph'] = self._get_graph()

        for agent, idx in anm.items():
            if (not self.pogema.grid.is_active[idx] or all(truncated) or all(terminated)) and agent in self.agents:
                self.agents.remove(agent)

        if self.render_steps:
            self.render()

        # cprint(self._get_graph(), 'white', 'on_red')
        # self.graph = self._get_graph()

        return d_observations, d_rewards, d_terminated, d_truncated, d_infos

    @property
    def unwrapped(self):
        return self

    def close(self):
        pass

    """ JOSH CODE"""
    def _get_graph(self):
        G = nx.Graph()
        G.add_nodes_from([i for i in range (self.num_agents)])

        for i in range (self.num_agents):
            for j in range (self.num_agents):
                if i != j:
                    if np.linalg.norm(np.array(self.pogema.grid.get_agents_xy()[i]) - np.array(self.pogema.grid.get_agents_xy()[j]))<=2.0:
                        G.add_edge(i,j)

        return G
        # return NotImplementedError()

    def get_state_size(self):
        """ Returns the shape of the state"""

        return np.size(self.state())

    def get_env_info(self):
        """
        Returns the environment information:
        state_shape
        obs_shape
        n_actions
        n_agents
        epsiode_length
        """
        
        cprint(self.pogema.grid, 'white', 'on_red')
        env_info = {
            'state_shape' : self.get_state_size(),
        }

        # cprint(env_info, 'white', 'on_red')
        # return env_info
        return env_info