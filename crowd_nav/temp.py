#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:37:37 2022

@author: amin
"""


import torch
import numpy as np
#import gym
import os
#from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
import configparser


def get_attention_weights(data):
    

    policy = policy_factory['sarl']()
    # model = policy.get_model()
    
    
    model_dir = 'data_crowd/output'
    model_weights = os.path.join(model_dir, 'rl_model.pth')
    
    policy_config_file = os.path.join(model_dir, os.path.basename('configs/policy.config'))
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    model = policy.get_model()
    model.eval()
    model.load_state_dict(torch.load(model_weights))
    #policy.set_phase('test')
    #policy.set_device('cpu')
    # model = policy.get_model()
    # z = torch.tensor([[ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  1.20e+00,
    #         -9.12e+00,  7.00e-03,  7.00e-03,  1.00e+00],
    #         [ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  1.37e+00,
    #         -8.27e+00,  0.00e+00,  0.00e+00,  1.00e+00],
    #         [ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  1.85e+00,
    #         -3.51e+00, -1.20e-02,  2.30e-02,  1.00e+00],
    #         [ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  2.49e+00,
    #         -3.60e+00, -9.00e-03,  2.30e-02,  1.00e+00],
    #         [ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  3.72e+00,
    #         -8.77e+00,  0.00e+00,  0.00e+00,  1.00e+00]])
    # z = torch.reshape(z,(1,5,10))
    
    size = np.shape(data)
    data = torch.tensor(data)
    data = torch.reshape(data,(1,size[0],size[1]))
    output = model(data.float())
    
    return model.attention_weights

# z = torch.tensor([[ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  1.20e+00,
#             -9.12e+00,  7.00e-03,  7.00e-03,  1.00e+00],
#             [ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  1.37e+00,
#             -8.27e+00,  0.00e+00,  0.00e+00,  1.00e+00],
#             [ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  1.85e+00,
#             -3.51e+00, -1.20e-02,  2.30e-02,  1.00e+00],
#             [ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  2.49e+00,
#             -3.60e+00, -9.00e-03,  2.30e-02,  1.00e+00],
#             [ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  3.72e+00,
#             -8.77e+00,  0.00e+00,  0.00e+00,  1.00e+00],
#             [ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  3.26e+00, 
#              -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00],
#             [ 0.0, 0.0,  0.0, 0.0,  0.0,  0.0, 
#              0.0,  0.0, 0.0,  0.0]])
# #z = torch.reshape(z,(1,6,10))
# output = get_attention_weights(z)
# print(output)
###############################################################################

# import torch
# import torch.nn as nn
# from torch.nn.functional import softmax
# import logging
# from crowd_nav.policy.cadrl import mlp
# from crowd_nav.policy.multi_human_rl import MultiHumanRL
# import numpy as np


# class ValueNetwork(nn.Module):
#     def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
#                  cell_size, cell_num):
#         super().__init__()
#         self.self_state_dim = self_state_dim
#         self.global_state_dim = mlp1_dims[-1]
#         self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
#         self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
#         self.with_global_state = with_global_state
#         if with_global_state:
#             self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
#         else:
#             self.attention = mlp(mlp1_dims[-1], attention_dims)
#         self.cell_size = cell_size
#         self.cell_num = cell_num
#         mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
#         self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
#         self.attention_weights = None

#     def forward(self, state):
#         """
#         First transform the world coordinates to self-centric coordinates and then do forward computation

#         :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
#         :return:
#         """
#         if np.shape(state)[2]>10:
#             state = torch.cat((state[:,:,:5],state[:,:,8:]),dim=2)
        
#         size = state.shape
#         self_state = state[:, 0, :self.self_state_dim]
#         mlp1_output = self.mlp1(state.view((-1, size[2])))
#         mlp2_output = self.mlp2(mlp1_output)

#         if self.with_global_state:
#             # compute attention scores
#             global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
#             global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
#                 contiguous().view(-1, self.global_state_dim)
#             attention_input = torch.cat([mlp1_output, global_state], dim=1)
#         else:
#             attention_input = mlp1_output
#         scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

#         # masked softmax
#         # weights = softmax(scores, dim=1).unsqueeze(2)
#         scores_exp = torch.exp(scores) * (scores != 0).float()
#         weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
#         self.attention_weights = weights[0, :, 0].data.cpu().numpy()

#         # output feature is a linear combination of input features
#         features = mlp2_output.view(size[0], size[1], -1)
#         # for converting to onnx
#         # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
#         weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

#         # concatenate agent's state with global weighted humans' state
#         joint_state = torch.cat([self_state, weighted_feature], dim=1)
#         value = self.mlp3(joint_state)
#         return value


# class SARL(MultiHumanRL):
#     def __init__(self):
#         super().__init__()
#         self.name = 'SARL'

#     def configure(self, config):
#         self.set_common_parameters(config)
#         mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
#         mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
#         mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
#         attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
#         self.with_om = config.getboolean('sarl', 'with_om')
#         with_global_state = config.getboolean('sarl', 'with_global_state')
#         self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
#                                   attention_dims, with_global_state, self.cell_size, self.cell_num)
#         self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
#         if self.with_om:
#             self.name = 'OM-SARL'
#         logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

#     def get_attention_weights(self):
#         return self.model.attention_weights

# z = torch.tensor([[ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  1.20e+00,
#         -9.12e+00,  7.00e-03,  7.00e-03,  1.00e+00],
#         [ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  1.37e+00,
#         -8.27e+00,  0.00e+00,  0.00e+00,  1.00e+00],
#         [ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  1.85e+00,
#         -3.51e+00, -1.20e-02,  2.30e-02,  1.00e+00],
#         [ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  2.49e+00,
#         -3.60e+00, -9.00e-03,  2.30e-02,  1.00e+00],
#         [ 3.26e+00, -5.19e+00,  9.00e-03, -6.80e-02,  1.00e+00,  3.72e+00,
#         -8.77e+00,  0.00e+00,  0.00e+00,  1.00e+00]])
# z = torch.reshape(z,(1,5,10))

# model = SARL()
# model_dir = 'data/output'
# model_weights = os.path.join(model_dir, 'rl_model.pth')
# model.load_state_dict(torch.load(model_weights))
# #out = model(z)





















