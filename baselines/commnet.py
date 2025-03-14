import argparse

import torch
import torch.nn.functional as F
from torch import nn

from .models import MLP
from .utils import select_action, translate_action
import sys
sys.path.append("../src")


class CommNetAgent(nn.Module):
    """
    MLP based CommNet. Uses communication vector to communicate info
    between agents
    """
    def __init__(self, agent_config):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MLP {object} -- Self
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
        """

        super(CommNetAgent, self).__init__()
        self.args = argparse.Namespace(**agent_config)
        # self.n_agents = args.n_agents
        # self.hid_size = args.hid_size
        # self.comm_passes = args.comm_passes
        # self.recurrent = args.recurrent
        if self.args.hard_attn:
            if not isinstance(self.args.n_actions, (list, tuple)):  # single action case
                self.args.n_actions = [self.args.n_actions]
            # self.dim_actions = self.args.env.dim_actions+1
            self.args.n_actions = [*self.args.n_actions, 2]
            self.n_action_heads = [int(self.args.n_actions[i]) for i in range(2)]
            self.heads = nn.ModuleList([nn.Linear(self.args.hid_size, o)
                                    for o in self.n_action_heads])
        else:
            self.head = nn.Linear(self.args.hid_size, self.args.n_actions)
        self.init_std = self.args.init_std if hasattr(self.args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.args.n_agents, self.args.n_agents)
        else:
            self.comm_mask = torch.ones(self.args.n_agents, self.args.n_agents) \
                            - torch.eye(self.args.n_agents, self.args.n_agents)


        # Since linear layers in PyTorch now accept * as any number of dimensions
        # between last and first dim, num_agents dimension will be covered.
        # The network below is function r in the paper for encoding
        # initial environment stage
        self.encoder = nn.Linear(self.args.obs_shape, self.args.hid_size)

        # if self.args.env_name == 'starcraft':
        #     self.state_encoder = nn.Linear(num_inputs, num_inputs)
        #     self.encoder = nn.Linear(num_inputs * 2, args.hid_size)
        if self.args.recurrent:
            self.hidd_encoder = nn.Linear(self.args.hid_size, self.args.hid_size)

        if self.args.recurrent:
            self.init_hidden(self.args.batch_size)
            self.f_module = nn.LSTMCell(self.args.hid_size, self.args.hid_size)

        else:
            if self.args.share_weights:
                self.f_module = nn.Linear(self.args.hid_size, self.args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(self.args.hid_size, self.args.hid_size)
                                                for _ in range(self.args.comm_passes)])
        # else:
            # raise RuntimeError("Unsupported RNN type.")

        # Our main function for converting current hidden state to next state
        # self.f = nn.Linear(args.hid_size, args.hid_size)
        if self.args.share_weights:
            self.C_module = nn.Linear(self.args.hid_size, self.args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.args.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(self.args.hid_size, self.args.hid_size)
                                            for _ in range(self.args.comm_passes)])
        # self.C = nn.Linear(args.hid_size, args.hid_size)

        # initialise weights as 0
        if self.args.comm_init == 'zeros':
            for i in range(self.args.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        # print(self.C)
        # self.C.weight.data.zero_()
        # Init weights for linear layers
        # self.apply(self.init_weights)

        self.value_head = nn.Linear(self.args.hid_size, 1)


    def get_agent_mask(self, batch_size, info):
        n = self.args.n_agents
        #
        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1).clone() # clone gives the full tensor and avoid the error

        return num_agents_alive, agent_mask

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, extras = x
            x = self.encoder(x)

            # if self.args.rnn_type == 'LSTM':
            hidden_state, cell_state = extras
            # else:
            #     hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
        else:
            x = x.unsqueeze(0)  #
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state


    def forward(self, x, info={}):
        # TODO: Update dimensions
        """Forward function for CommNet class, expects state, previous hidden
        and communication tensor.
        B: Batch Size: Normally 1 in case of episode
        N: number of agents

        Arguments:
            x {tensor} -- State of the agents (N x num_inputs)
            prev_hidden_state {tensor} -- Previous hidden state for the networks in
            case of multiple passes (1 x N x hid_size)
            comm_in {tensor} -- Communication tensor for the network. (1 x N x N x hid_size)

        Returns:
            tuple -- Contains
                next_hidden {tensor}: Next hidden state for network
                comm_out {tensor}: Next communication tensor
                action_data: Data needed for taking next action (Discrete values in
                case of discrete, mean and std in case of continuous)
                v: value head
        """

        # if self.args.env_name == 'starcraft':
        #     maxi = x.max(dim=-2)[0]
        #     x = self.state_encoder(x)
        #     x = x.sum(dim=-2)
        #     x = torch.cat([x, maxi], dim=-1)
        #     x = self.tanh(x)

        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size()[0]
        n = self.args.n_agents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            agent_mask *= comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)

        for i in range(self.args.comm_passes):
            # Choose current or prev depending on recurrent
            comm = hidden_state.view(batch_size, n, self.args.hid_size) if self.args.recurrent else hidden_state
            #comm = comm.unsqueeze(0)
            # Get the next communication vector based on next hidden state
            comm = comm.unsqueeze(-2).expand(-1, n, n, self.args.hid_size)

            # Create mask for masking self communication
            mask = self.comm_mask.view(1, n, n)
            mask = mask.expand(comm.shape[0], n, n)
            mask = mask.unsqueeze(-1)

            mask = mask.expand_as(comm)
            comm = comm * mask

            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
                and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)

            # Mask comm_in
            # Mask communcation from dead agents
            comm = comm * agent_mask
            # Mask communication to dead agents
            comm = comm * agent_mask_transpose

            # Combine all of C_j for an ith agent which essentially are h_j
            comm_sum = comm.sum(dim=1)
            c = self.C_modules[i](comm_sum)


            if self.args.recurrent:
                # skip connection - combine comm. matrix and encoded input for all agents
                inp = x + c

                inp = inp.view(batch_size * n, self.args.hid_size)

                output = self.f_module(inp, (hidden_state, cell_state))

                hidden_state = output[0]
                cell_state = output[1]

            else: # MLP|RNN
                # Get next hidden state from f node
                # and Add skip connection from start and sum them
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)
        hidden_state= hidden_state.squeeze(0)#
        # v = torch.stack([self.value_head(hidden_state[:, i, :]) for i in range(n)])
        # v = v.view(hidden_state.size(0), n, -1)
        value_head = self.value_head(hidden_state)
        #h = hidden_state.view(batch_size, n, self.hid_size)
        h = hidden_state.view(n, self.args.hid_size)#
        #h = hidden_state.view(batch_size, n, self.hid_size)
        # if self.continuous:
        #     action_mean = self.action_mean(h)
        #     action_log_std = self.action_log_std.expand_as(action_mean)
        #     action_std = torch.exp(action_log_std)
        #     # will be used later to sample
        #     action = (action_mean, action_log_std, action_std)
        # else:
            # discrete actions
        if self.args.hard_attn:
            action = [F.log_softmax(head(h), dim=-1) for head in self.heads]
        else:
            action = F.log_softmax(self.head(h), dim=-1)#
        #action = [F.log_softmax(head(h), dim=-1) for head in self.heads]

        if self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone())
        else:
            return action, value_head

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.args.n_agents, self.args.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.args.n_agents, self.args.hid_size, requires_grad=True)))

