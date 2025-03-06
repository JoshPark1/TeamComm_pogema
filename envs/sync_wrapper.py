from .syncorsink.Orion.forage_main.cool_forage_env import PettingZooForageEnv
import torch
from torch import nn
from envs.syncorsink.Orion.forage_main import utils
import argparse
import numpy as np
import networkx as nx
from .syncorsink.Orion.forage_main.forage_env_wrapper import ForageEnvironment

from termcolor import cprint

class TiecommPettingZoo(PettingZooForageEnv):
    """
    PettingZooForageEnv has agents and players and will instatiate everything automatically.
    Once the game is done, it removes the agents. So agents will BECOME an empty array while players stay the same.
    The game STARTS with no players so you have to call game.addplayer() to add players (it does for you in reset())

    CRITICAL DISTINCTION:
    The game only has "players" but they get removed when done (kinda like agents in the wrapper)
    """

    def __init__(self, config):
        super().__init__()
        self.reset() #To add the players to the game

        self.args = argparse.Namespace(**config)
        time_limit = self.args.time_limit
        self.episode_limit = time_limit

        # self.concat_obs = None
        

        # cprint(self.agents, "white", "on_green")
        # cprint(self.reset()[0]['agents_0'].keys(), "white", "on_green")

        # MINIMAP_HISTORY = 3
        # FOV_HISTORY = 3
        # GRID_SIZE = 200
        # self.mini_map_cnn = nn.Sequential(
        #     nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(MINIMAP_HISTORY, 3, 3), stride=1),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )
        # self.mini_map_output_dim = 16 * (GRID_SIZE - 2) * (GRID_SIZE - 2)
        # self.fov_cnn = nn.Sequential(
        #     nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(FOV_HISTORY, 3, 3), stride=1),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )
        # self.fov_output_dim = 16 * (GRID_SIZE - 2) * (GRID_SIZE - 2)

    def get_state_size(self):
        """ Returns the shape of the state from the observation"""
        obs = self._observe()
        obs_size = len(self._adjust_obs(obs)['agents_0'])
        state_size = obs_size * len(self.players)
        return state_size
        

    def _adjust_obs(self, obs):
        '''
        Convolves the minimap and fov and flattens it all into a 1D array
        '''
        # mini_map, fov, position, relative_position, holding_crate, value_of_crate, target_score, group_score, delivery_points = self.preprocess_observations(
        #         obs, self.players, mini_map_history=3, fov_history=3
        #     )
        # for i, agent in enumerate(obs.keys()):
        #     obs[agent]['minimap'] = self.mini_map_cnn(mini_map[0][i])   # TODO figure out how to index with batch size
        #     obs[agent]['fov'] = self.fov_cnn(fov[0][i])
        new_obs = {}
        for agent in obs:
            res = []
            for field, value in obs[agent].items():
                res.extend(utils.flatten_array(value))
            new_obs[agent] = res
        return new_obs

    # def preprocess_observations(self, observations, agents, mini_map_history, fov_history):
    #     mini_map_list, fov_list = [], []
    #     position_list, relative_position_list = [], []
    #     holding_crate_list, value_of_crate_list = [], []
    #     target_score_list, group_score_list = [], []
    #     delivery_points = None
    #     all_delivery_points = []
    #     for agent in agents:
    #         obs = observations[agent]
    #         mini_map_list.append(obs['minimap'])
    #         fov_list.append(obs['fov'])
    #         position_list.append(obs['position'])
    #         relative_position_list.append(obs['relative_delivery_position'])
    #         holding_crate_list.append([obs['holding_crate']])
    #         value_of_crate_list.append(obs['crate_value'])
    #         target_score_list.append([obs['target_score']])
    #         group_score_list.append([obs['group_score']])
    #         all_delivery_points.append(obs['delivery_points'])
    #         if delivery_points is None:
    #             delivery_points = obs['delivery_points']
    #     mini_map = torch.tensor(mini_map_list).unsqueeze(0).float() / 255.0
    #     fov = torch.tensor(fov_list).unsqueeze(0).float() / 255.0
    #     position = torch.tensor(position_list).unsqueeze(0).float()
    #     relative_position = torch.tensor(relative_position_list).unsqueeze(0).float()
    #     holding_crate = torch.tensor(holding_crate_list).unsqueeze(0).float()
    #     value_of_crate = torch.tensor(value_of_crate_list).unsqueeze(0).float()
    #     target_score = torch.tensor(target_score_list).unsqueeze(0).float()
    #     group_score = torch.tensor(group_score_list).unsqueeze(0).float()
    #     delivery_points = torch.tensor(delivery_points).unsqueeze(0).float()
    #     return mini_map, fov, position, relative_position, holding_crate, value_of_crate, target_score, group_score, delivery_points
        

    def get_obs_size(self):

        obs = self._observe()
        obs_size = len(self._adjust_obs(obs)['agents_0'])
        
        return obs_size
    

    def get_obs(self):
        '''
        Returns the observations given the actions.
        '''
        obs = self._observe()

        # cprint(obs, "blue")

        output_obs = {}
        all_observations = []
        for agent in self.agents:
            output_obs[agent] = {}
            # processed_obs = self._obs_processing(obs[agent])
            processed_obs = self._adjust_obs(obs)[agent]
            output_obs[agent]['observation'] = processed_obs
            # all_observations.append(processed_obs.detach().numpy())  # Detach and convert to numpy array
            all_observations.append(processed_obs)

        return all_observations
    

    def get_graph(self):

        # cprint(self.game.get_player_data()['agents_0']['position'], "white", "on_green")
        # cprint(self.agents, "white", "on_green")

        G = nx.Graph()
        G.add_nodes_from([i for i in range (self.num_agents)])
        agent_data = self.game.get_player_data()
        # for i in range(self.ncar):
        #     G.add_node(i, feature = np.array(self.obs[i]))

        for i, agent_i in enumerate(self.agents):
            for j, agent_j in enumerate(self.agents):
                if agent_i != agent_j:
                    # test = np.linalg.norm(np.array(self.world.agents[i].state.p_pos) - np.array(self.world.agents[j].state.p_pos))
                    # print(test)
                    if np.linalg.norm(np.array(agent_data[agent_i]['position']) - np.array(agent_data[agent_j]['position']))<=2.0:
                        G.add_edge(i,j)
                    # if self.scenario.group_indices[i] == self.scenario.group_indices[j] or \
                    #         np.linalg.norm(np.array(self.world.agents[i].state.p_pos) - np.array(self.world.agents[j].state.p_pos))<= 2.0:
                    #         G.add_edge(i, j)

        # nx.draw(G, with_labels=True, node_color='#A0CBE2', edge_color='#A0CBE2', node_size=100, width=1)
        # plt.show()
        
        # set = cd.louvain(G).communities
        # print(set)
        
        # g = nx.Graph()
        # g.add_nodes_from(G.nodes(data=False))
        
        # for e in G.edges():
        #     strength = measure_strength(G, e[0], e[1])
        #     print(strength)
        #     if strength > 0.8:
        #         g.add_edge(e[0], e[1])
        
        # #set = [list(c) for c in nx.connected_components(g)]
        
        # subax1 = plt.subplot(121)
        # nx.draw(G, with_labels=True, node_color='#A0CBE2', edge_color='#A0CBE2', node_size=100, width=1)
        # subax2 = plt.subplot(122)
        # nx.draw(g, pos=nx.spring_layout(g), with_labels=True, node_color='#A0CBE2', edge_color='#A0CBE2',
        #         node_size=100, edge_cmap=plt.cm.Blues, width=1)
        # plt.show()

        return G
    
    def reformat_action(self, action):
        """
        Reformat the action from a list to a dictionary
        """
        reformatted_action = {}
        for i, agent in enumerate(self.agents):
            reformatted_action[agent] = action[0][i]

        return reformatted_action
        

    def get_env_info(self):
        # cprint(self.action_spaces['agents_0'].n, "white", "on_green")
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.action_spaces['agents_0'].n, 
            "n_agents": len(self.players),
            "episode_length": self.episode_limit,
        }
    
    
    def step_func_adapter(self, terminations, truncations, infos):
        """
        Turns inputs (terminations, truncations, infos) outputted from cool_forage_env 
        into format {Timelimit.truncated: True} have to find what different kinds of
        keys there can be
        """
        adapted_info = {}
        for agents in truncations:
            if truncations[agents]:
                adapted_info['Timelimit.truncated'] = True
        
        return adapted_info
    