from typing import Optional

import numpy as np
import gymnasium
from gymnasium.error import ResetNeeded

from pogema.animation import AnimationMonitor
from pogema.grid import Grid, GridLifeLong
from pogema.grid_config import GridConfig, EnvMode, RewardScheme
from pogema.wrappers.metrics import LifeLongAverageThroughputMetric, NonDisappearEpLengthMetric, \
    NonDisappearCSRMetric, NonDisappearISRMetric, EpLengthMetric, ISRMetric, CSRMetric
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.generator import generate_new_target
from pogema.wrappers.persistence import PersistentWrapper

from termcolor import cprint

class ActionsSampler:
    """
    Samples the random actions for the given number of agents using the given seed.
    """

    def __init__(self, num_actions, seed=42):
        self._num_actions = num_actions
        self._rnd = None
        self.update_seed(seed)

    def update_seed(self, seed=None):
        self._rnd = np.random.default_rng(seed)

    def sample_actions(self, dim=1):
        return self._rnd.integers(self._num_actions, size=dim)


class PogemaBase(gymnasium.Env):
    """
    Abstract class of the Pogema environment.
    """
    metadata = {"render_modes": ["ansi"], }

    def step(self, action):
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None, ):
        raise NotImplementedError

    def __init__(self, grid_config: GridConfig = GridConfig()):
        # noinspection PyTypeChecker
        self.grid: Grid = None
        self.grid_config = grid_config

        # Add a hover move that corresponds to comm
        if grid_config.env_mode == EnvMode.DECENTRALIZED_COMM:
            grid_config.MOVES.append([0, 0])

        self.action_space: gymnasium.spaces.Discrete = gymnasium.spaces.Discrete(len(self.grid_config.MOVES))
        self._multi_action_sampler = ActionsSampler(self.action_space.n, seed=self.grid_config.seed)

    def get_agents_obs(self, agent_id=0, masked=True):
        """
        Returns the observation of the agent with the given id.
        :param agent_id: (which agent)
        :param masked: (used for global state generation)
        :return:
        """
        if not masked:
            # do not mask based on what the agent has been able to view, just give the full-perfect observation
            # assuming global FOV
            return np.ndarray.flatten(np.concatenate([
                self.grid.get_obstacles_for_agent(agent_id)[None], 
                self.grid.get_positions(agent_id)[None],
                self.grid.get_square_target(agent_id)[None],
            ])).astype(float)

        total_history_mask, just_now_mask = self.grid.make_mask(agent_id)

        if self.grid_config.env_mode != EnvMode.DECENTRALIZED_COMM:
            return np.ndarray.flatten(np.concatenate([
                total_history_mask & self.grid.get_obstacles_for_agent(agent_id)[None], 
                -just_now_mask & self.grid.get_positions(agent_id)[None],
                total_history_mask & self.grid.get_square_target(agent_id)[None],
            ])).astype(float)
        else:
            print(agent_id, np.concatenate([
                total_history_mask & self.grid.get_obstacles_for_agent(agent_id)[None], 
                -just_now_mask & self.grid.get_positions(agent_id)[None],
                -total_history_mask & self.grid.get_square_target(agent_id)[None],
                np.expand_dims(self.grid.comm_masks[agent_id], 0)
            ]))
            return np.ndarray.flatten(np.concatenate([
                total_history_mask & self.grid.get_obstacles_for_agent(agent_id)[None], 
                -just_now_mask & self.grid.get_positions(agent_id)[None],
                -total_history_mask & self.grid.get_square_target(agent_id)[None],
                np.expand_dims(self.grid.comm_masks[agent_id], 0)
            ])).astype(float)

    def check_reset(self):
        """
        Checks if the reset needed.
        :return:
        """
        if self.grid is None:
            raise ResetNeeded("Please reset environment first!")

    def render(self, mode='human'):
        """
        Renders the environment using ascii graphics.
        :param mode:
        :return:
        """
        self.check_reset()
        return self.grid.render(mode=mode)

    def sample_actions(self):
        """
        Samples the random actions for the given number of agents.
        :return:
        """
        return self._multi_action_sampler.sample_actions(dim=self.grid_config.num_agents)

    def get_num_agents(self):
        """
        Returns the number of agents in the environment.
        :return:
        """
        return self.grid_config.num_agents


class Pogema(PogemaBase):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)
        self.was_on_goal = None
        full_size = self.grid_config.obs_radius * 2 + 1
        if self.grid_config.observation_type == 'default':
            self.observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=((3 if grid_config.env_mode != EnvMode.DECENTRALIZED_COMM else 4) * self.grid_config.size * self.grid_config.size,))
        elif self.grid_config.observation_type == 'POMAPF':
            self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
                obstacles=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                agents=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
            )
        elif self.grid_config.observation_type == 'MAPF':
            self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
                obstacles=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                agents=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                # global_obstacles=None, # todo define shapes of global state variables
                # global_xy=None,
                # global_target_xy=None,
            )
        else:
            raise ValueError(f"Unknown observation type: {self.grid.config.observation_type}")

    def comm_broadcast(self, agent_idx):
        for i in range(self.grid_config.num_agents):
            # update what this agent knows for future observations
            self.grid.observed_masks[i] = self.grid.observed_masks[i] | self.grid.observed_masks[agent_idx]

            # update this agent's comm mask to indicate what has now been communicated
            self.grid.comm_masks[i] = self.grid.comm_masks[i] | self.grid.observed_masks[agent_idx]

            # note: globally observed mask does not have to change as comm does not affect what's globally known

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents
        rewards = []
        target_achieved = []
        terminated = []

        # physically move the agents
        collision = self.move_agents(action)
        self.update_was_on_goal()

        # check for any communication and enact if so
        if self.grid_config.env_mode == EnvMode.DECENTRALIZED_COMM:
            for agent_idx in range(self.grid_config.num_agents):
                if action[agent_idx] == 5: # comm action
                    self.comm_broadcast(agent_idx)

        all_done = True
        num_done = 0
        for agent_idx in range(self.grid_config.num_agents):
            on_goal = self.grid.on_goal(agent_idx)
            if not on_goal and self.grid.is_active[agent_idx]:
                all_done = False
            
            if on_goal and self.grid.is_active[agent_idx] and not collision:
                target_achieved.append(1.0)
                num_done += 1
            else:
                target_achieved.append(0.0)
            
        terminated = [all_done or collision] * self.grid_config.num_agents

        if collision:
            rewards = [0.0] * self.grid_config.num_agents
        elif self.grid_config.reward_scheme == RewardScheme.INDEPENDENT:
            rewards = target_achieved
        elif self.grid_config.reward_scheme == RewardScheme.SHARED_ON_TARGET:
            rewards = [num_done / self.grid_config.num_agents if num_done > 0 else 0.0] * self.grid_config.num_agents
        elif self.grid_config.reward_scheme == RewardScheme.SHARED_ALL_REACH_TARGET:
            rewards = [1.0 if all_done else 0.0] * self.grid_config.num_agents
        elif self.grid_config.reward_scheme == RewardScheme.INDEPENDENT_AND_COOPERATIVE_SHARED:
            rewards = [(1.0 if all_done else 0.0) + 0.2 * t for t in target_achieved]

        for agent_idx in range(self.grid_config.num_agents):
            if self.grid.on_goal(agent_idx):
                self.grid.hide_agent(agent_idx)
                self.grid.is_active[agent_idx] = False

        infos = self._get_infos()

        observations = self._obs()
        truncated = [False] * self.grid_config.num_agents
        return observations, rewards, terminated, truncated, infos

    def _initialize_grid(self):
        self.grid: Grid = Grid(grid_config=self.grid_config)

    def update_was_on_goal(self):
        self.was_on_goal = [self.grid.on_goal(agent_idx) and self.grid.is_active[agent_idx]
                            for agent_idx in range(self.grid_config.num_agents)]

    def reset(self, seed: Optional[int] = None, return_info: bool = True, options: Optional[dict] = None, ):
        self._initialize_grid()
        self.update_was_on_goal()

        if seed is not None:
            self.grid.seed = seed

        if return_info:
            return self._obs(), self._get_infos()
        return self._obs()

    def _obs(self):
        if self.grid_config.observation_type == 'default':
            return [self.get_agents_obs(index) for index in range(self.grid_config.num_agents)]
        elif self.grid_config.observation_type == 'POMAPF':
            return self._pomapf_obs()

        elif self.grid_config.observation_type == 'MAPF':
            results = self._pomapf_obs()
            global_obstacles = self.grid.get_obstacles()
            global_agents_xy = self.grid.get_agents_xy()
            global_targets_xy = self.grid.get_targets_xy()

            for agent_idx in range(self.grid_config.num_agents):
                result = results[agent_idx]
                result.update(global_obstacles=global_obstacles)
                result['global_xy'] = global_agents_xy[agent_idx]
                result['global_target_xy'] = global_targets_xy[agent_idx]

            return results
        else:
            raise ValueError(f"Unknown observation type: {self.grid.config.observation_type}")

    def _pomapf_obs(self):
        results = []
        agents_xy_relative = self.grid.get_agents_xy_relative()
        targets_xy_relative = self.grid.get_targets_xy_relative()

        for agent_idx in range(self.grid_config.num_agents):
            result = {'obstacles': self.grid.get_obstacles_for_agent(agent_idx),
                      'agents': self.grid.get_positions(agent_idx),
                      'xy': agents_xy_relative[agent_idx],
                      'target_xy': targets_xy_relative[agent_idx]}

            results.append(result)
        return results

    def _get_infos(self):
        infos = [dict() for _ in range(self.grid_config.num_agents)]
        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]
            
            # add action mask to info dict
            if not infos[agent_idx]['is_active']: # an inactive agent can't do anything
                infos[agent_idx]['action_mask'] = np.zeros(5).astype(int)
                continue
            infos[agent_idx]['action_mask'] = np.ones(5).astype(int) # all actions possible initially
            pos = self.grid.get_agents_xy()[agent_idx]
            for (a, move) in enumerate(self.grid_config.MOVES):
                # check if doing this move would cause a collision with obstacle
                if self.grid.has_obstacle(pos[0] + move[0], pos[1] + move[1]):
                    infos[agent_idx]['action_mask'][a] = 0
        return infos

    def _revert_action(self, agent_idx, used_cells, cell, actions):
        actions[agent_idx] = 0
        used_cells[cell].remove(agent_idx)
        new_cell = self.grid.positions_xy[agent_idx]
        if new_cell in used_cells and len(used_cells[new_cell]) > 0:
            used_cells[new_cell].append(agent_idx)
            return self._revert_action(used_cells[new_cell][0], used_cells, new_cell, actions)
        else:
            used_cells.setdefault(new_cell, []).append(agent_idx)
        return actions, used_cells

    
    # Returns True if a collision occured
    # Based on original pogema 'soft' collision system logic
    def move_agents(self, actions):
        used_cells = dict()
        used_edges = dict()
        agents_xy = self.grid.get_agents_xy()
        for agent_idx, (x, y) in enumerate(agents_xy):
            if self.grid.is_active[agent_idx]:
                dx, dy = self.grid.config.MOVES[actions[agent_idx]]
                used_cells.setdefault((x + dx, y + dy), []).append(agent_idx)
                used_edges[x, y, x + dx, y + dy] = [agent_idx]
                if dx != 0 or dy != 0:
                    used_edges.setdefault((x + dx, y + dy, x, y), []).append(agent_idx)
        for agent_idx, (x, y) in enumerate(agents_xy):
            if self.grid.is_active[agent_idx]:
                dx, dy = self.grid.config.MOVES[actions[agent_idx]]
                if len(used_edges[x, y, x + dx, y + dy]) > 1:
                    #used_cells[x + dx, y + dy].remove(agent_idx)
                    #used_cells.setdefault((x, y), []).append(agent_idx)
                    #actions[agent_idx] = 0
                    return True
        for agent_idx in reversed(range(len(agents_xy))):
            x, y = agents_xy[agent_idx]
            if self.grid.is_active[agent_idx]:
                dx, dy = self.grid.config.MOVES[actions[agent_idx]]
                if len(used_cells[x + dx, y + dy]) > 1 or self.grid.has_obstacle(x + dx, y + dy):
                    # actions, used_cells = self._revert_action(agent_idx, used_cells, (x + dx, y + dy), actions)
                    return True
        for agent_idx in range(self.grid_config.num_agents):
            if self.grid.is_active[agent_idx]:
                self.grid.move_without_checks(agent_idx, actions[agent_idx])
        
        return False

    def get_agents_xy_relative(self):
        return self.grid.get_agents_xy_relative()

    def get_targets_xy_relative(self):
        return self.grid.get_targets_xy_relative()

    def get_obstacles(self, ignore_borders=False):
        return self.grid.get_obstacles(ignore_borders=ignore_borders)

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_agents_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_targets_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_state(self, ignore_borders=False, as_dict=False):
        return self.grid.get_state(ignore_borders=ignore_borders, as_dict=as_dict)


class PogemaLifeLong(Pogema):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)

    def _initialize_grid(self):
        self.grid: GridLifeLong = GridLifeLong(grid_config=self.grid_config)

        main_rng = np.random.default_rng(self.grid_config.seed)
        seeds = main_rng.integers(np.iinfo(np.int32).max, size=self.grid_config.num_agents)
        self.random_generators = [np.random.default_rng(seed) for seed in seeds]

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents
        rewards = []

        infos = [dict() for _ in range(self.grid_config.num_agents)]

        self.move_agents(action)
        self.update_was_on_goal()

        for agent_idx in range(self.grid_config.num_agents):
            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.grid.is_active[agent_idx]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

            if self.grid.on_goal(agent_idx):
                self.grid.finishes_xy[agent_idx] = generate_new_target(self.random_generators[agent_idx],
                                                                       self.grid.point_to_component,
                                                                       self.grid.component_to_points,
                                                                       self.grid.positions_xy[agent_idx])

        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]

        obs = self._obs()

        terminated = [False] * self.grid_config.num_agents
        truncated = [False] * self.grid_config.num_agents
        return obs, rewards, terminated, truncated, infos


class PogemaCoopFinish(Pogema):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)
        self.num_agents = self.grid_config.num_agents
        self.is_multiagent = True

    def _initialize_grid(self):
        self.grid: Grid = Grid(grid_config=self.grid_config)

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents

        infos = [dict() for _ in range(self.grid_config.num_agents)]

        self.move_agents(action)
        self.update_was_on_goal()

        is_task_solved = all(self.was_on_goal)
        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]

        obs = self._obs()

        terminated = [is_task_solved] * self.grid_config.num_agents
        truncated = [False] * self.grid_config.num_agents
        rewards = [1.0 if is_task_solved else 0.0 for _ in range(self.grid_config.num_agents)]
        return obs, rewards, terminated, truncated, infos


def _make_pogema(grid_config):
    if grid_config.on_target == 'restart':
        env = PogemaLifeLong(grid_config=grid_config)
        env = AnimationMonitor(env)
    elif grid_config.on_target == 'nothing':
        env = PogemaCoopFinish(grid_config=grid_config)
        env = AnimationMonitor(env)
    elif grid_config.on_target == 'finish':
        env = Pogema(grid_config=grid_config)
        env = AnimationMonitor(env)
    else:
        raise KeyError(f'Unknown on_target option: {grid_config.on_target}')

    env = MultiTimeLimit(env, grid_config.max_episode_steps)

    """
    if env.grid_config.persistent:
        env = PersistentWrapper(env)
    else:
        # adding metrics wrappers
        if grid_config.on_target == 'restart':
            env = LifeLongAverageThroughputMetric(env)
        elif grid_config.on_target == 'nothing':
            env = NonDisappearISRMetric(env)
            env = NonDisappearCSRMetric(env)
            env = NonDisappearEpLengthMetric(env)
        elif grid_config.on_target == 'finish':
            env = ISRMetric(env)
            env = CSRMetric(env)
            env = EpLengthMetric(env)
        else:
            raise KeyError(f'Unknown on_target option: {grid_config.on_target}')
    """

    return env
