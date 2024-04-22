from typing import Dict, List
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.torch.misc import normc_initializer, same_padding, SlimConv2d, SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.framework import TensorType


from .gnn import adversarialGraphML as gml_adv
from .gnn import graphML as gml
from .gnn import graphTools
import numpy as np
import copy
from torch.distributions.categorical import Categorical

import tree
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
import pdb
import json

torch, nn = try_import_torch()
from torchsummary import summary

# https://ray.readthedocs.io/en/latest/using-ray-with-pytorch.html

REWARDS_ROLE = "rewards_role"
# PREV_REWARDS_ROLE = "prev_rewards_role"
# VALUES_ROLE = "values_role"
ACTIONS_ROLE = "actions_role"
# PREV_ACTIONS_ROLE = "prev_actions_role"
ACTIONS_PRIMITIVE = "actions_primitive"
# PREV_ACTIONS_PRIMITIVE = "prev_actions_primitive"
VF_PREDS_ROLE = "vf_preds_role"
# ADVANTAGES_ROLE = "advantages_role"
# VALUE_TARGETS_ROLE = "value_targets_role"

VARIFY = False
EXPLORE = False
COVERAGE = False

DEFAULT_OPTIONS = {
    "activation": "relu",
    "agent_split": 1,
    "cnn_compression": 512,
    "cnn_filters": [[32, [8, 8], 4], [64, [4, 4], 2], [128, [4, 4], 2]],
    "cnn_residual": False,
    "freeze_coop": True,
    "freeze_coop_value": False,
    "freeze_greedy": False,
    "freeze_greedy_value": False,
    "graph_edge_features": 1,
    "graph_features": 512,
    "graph_layers": 1,
    "graph_tabs": 3,
    "relative": True,
    "value_cnn_compression": 512,
    "value_cnn_filters": [[32, [8, 8], 2], [64, [4, 4], 2], [128, [4, 4], 2]],
    "forward_values": True
}


class RoleModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):#,
        #graph_layers, graph_features, graph_tabs, graph_edge_features, cnn_filters, value_cnn_filters, value_cnn_compression, cnn_compression, relative, activation):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(model_config['custom_model_config'])

        #self.cfg = model_config['custom_options']
        self.n_agents = len(obs_space.original_space['agents'])
        self.graph_features = self.cfg['graph_features']
        self.cnn_compression = self.cfg['cnn_compression']
        self.activation = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU
        }[self.cfg['activation']]

        ################## Local encoder #####################
        layers = []
        input_shape = obs_space.original_space['agents'][0]['map'].shape
        (w, h, in_channels) = input_shape

        in_size = [w, h]
        for out_channels, kernel, stride in self.cfg['cnn_filters'][:-1]:
            padding, out_size = same_padding(in_size, kernel, [stride, stride])
            layers.append(SlimConv2d(in_channels, out_channels, kernel, stride, padding, activation_fn=self.activation))
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = self.cfg['cnn_filters'][-1]
        layers.append(
            SlimConv2d(in_channels, out_channels, kernel, stride, None))
        layers.append(nn.Flatten(1, -1))
        #if isinstance(cnn_compression, int):
        #    layers.append(nn.Linear(cnn_compression, self.cfg['graph_features']-2)) # reserve 2 for pos
        #    layers.append(self.activation{))
        self.coop_convs = nn.Sequential(*layers)
        self.greedy_convs = copy.deepcopy(self.coop_convs)

        self.coop_value_obs_convs = copy.deepcopy(self.coop_convs)
        self.greedy_value_obs_convs = copy.deepcopy(self.coop_convs)

        self.primitive_coop_convs = copy.deepcopy(self.coop_convs)
        self.primitive_greedy_convs = copy.deepcopy(self.coop_convs)

        self.primitive_coop_value_obs_convs = copy.deepcopy(self.coop_convs)
        self.primitive_greedy_value_obs_convs = copy.deepcopy(self.coop_convs)

        summary(self.coop_convs, device="cpu", input_size=(input_shape[2], input_shape[0], input_shape[1]))
        ################## Local encoder #####################

        ############## communication   #######################
        gfl = []
        for i in range(self.cfg['graph_layers']):
            gfl.append(gml_adv.GraphFilterBatchGSOA(self.graph_features, self.graph_features, self.cfg['graph_tabs'], self.cfg['agent_split'], self.cfg['graph_edge_features'], False))
            #gfl.append(gml.GraphFilterBatchGSO(self.graph_features, self.graph_features, self.cfg['graph_tabs'], self.cfg['graph_edge_features'], False))
            gfl.append(self.activation())

        self.GFL = nn.Sequential(*gfl)
        ############## communication   #######################

        ########## This is the logist of local encoders of role#################
        logits_inp_features = self.graph_features
        if self.cfg['cnn_residual']:
            logits_inp_features += self.cnn_compression

        post_logits = [
            nn.Linear(logits_inp_features, 64),
            self.activation(),
            nn.Linear(64, 32),
            self.activation()
        ]
        logit_linear = nn.Linear(32, 2)
        nn.init.xavier_uniform_(logit_linear.weight)
        nn.init.constant_(logit_linear.bias, 0)
        post_logits.append(logit_linear)
        self.coop_logits = nn.Sequential(*post_logits)
        self.greedy_logits = copy.deepcopy(self.coop_logits)
        self.primitive_greedy_logits = copy.deepcopy(self.coop_logits)
        summary(self.coop_logits, device="cpu", input_size=(logits_inp_features,))
        ########## This is the logist of local encoders of role#################
        

        ########## This is the logist of local encoders of primitive#################
        logits_inp_features_ = self.cnn_compression + 1

        post_logits = [
            nn.Linear(logits_inp_features_, 64),
            self.activation(),
            nn.Linear(64, 32),
            self.activation()
        ]
        logit_linear = nn.Linear(32, 5)
        nn.init.xavier_uniform_(logit_linear.weight)
        nn.init.constant_(logit_linear.bias, 0)
        post_logits.append(logit_linear)
        self.primitive_greedy_logits = nn.Sequential(*post_logits)
        self.primitive_coop_logits = copy.deepcopy(self.primitive_greedy_logits)
        summary(self.primitive_greedy_logits, device="cpu", input_size=(logits_inp_features_,))
        ########## This is the logist of local encoders of primitive #################
        
        ##################  This is the state encoder      ############################
        layers = []
        input_shape = np.array(obs_space.original_space['state'].shape)
        (w, h, in_channels) = input_shape

        in_size = [w, h]
        for out_channels, kernel, stride in self.cfg['value_cnn_filters'][:-1]:
            padding, out_size = same_padding(in_size, kernel, [stride, stride])
            layers.append(SlimConv2d(in_channels, out_channels, kernel, stride, padding, activation_fn=self.activation))
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = self.cfg['value_cnn_filters'][-1]
        layers.append(
            SlimConv2d(in_channels, out_channels, kernel, stride, None))
        layers.append(nn.Flatten(1, -1))

        self.primitive_coop_value_cnns = nn.Sequential(*layers)
        self.primitive_greedy_value_cnns = copy.deepcopy(self.primitive_coop_value_cnns)
        self.coop_value_cnns = copy.deepcopy(self.primitive_coop_value_cnns)
        self.greedy_value_cnns = copy.deepcopy(self.primitive_coop_value_cnns)
        summary(self.primitive_coop_value_cnns, device="cpu", input_size=(input_shape[2], input_shape[0], input_shape[1]))
        ##################  This is the state encoder      ############################

        ############## This is the logist of state encoders in primitive model #####################
        layers = [
            nn.Linear(self.cnn_compression + self.cfg['value_cnn_compression'] + 1, 64),
            self.activation(),
            nn.Linear(64, 32),
            self.activation()
        ]
        values_linear = nn.Linear(32, 1)
        normc_initializer()(values_linear.weight)
        nn.init.constant_(values_linear.bias, 0)
        layers.append(values_linear)

        self.primitive_coop_value_branch = nn.Sequential(*layers)
        self.primitive_greedy_value_branch = copy.deepcopy(self.primitive_coop_value_branch)
        summary(self.primitive_coop_value_branch, device="cpu", input_size=(self.cnn_compression + self.cfg['value_cnn_compression'] + 1,))
        ############## This is the logist of state encoders in primitive model  #####################

        ############## This is the logist of state encoders in primitive model #####################
        layers = [
            nn.Linear(self.cnn_compression + self.cfg['value_cnn_compression'], 64),
            self.activation(),
            nn.Linear(64, 32),
            self.activation()
        ]
        values_linear = nn.Linear(32, 1)
        normc_initializer()(values_linear.weight)
        nn.init.constant_(values_linear.bias, 0)
        layers.append(values_linear)

        self.coop_value_branch = nn.Sequential(*layers)
        self.greedy_value_branch = copy.deepcopy(self.coop_value_branch)
        summary(self.coop_value_branch, device="cpu", input_size=(self.cnn_compression + self.cfg['value_cnn_compression'],))
        ############## This is the logist of state encoders in primitive model  #####################

        ################ global information (2channel) fed into 3layers CNN ################
        layers = []
        input_channel = obs_space.original_space['merged_states'].shape[-1]
        input_shape = np.array(obs_space.original_space['merged_states'].shape)
        layers.append(nn.Conv2d(in_channels=input_channel,out_channels=16,kernel_size=5,stride=2,padding=1))
        layers.append(nn.BatchNorm2d(num_features=16))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        layers.append(nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1))
        layers.append(nn.BatchNorm2d(num_features=32))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1))
        layers.append(nn.BatchNorm2d(num_features=64))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1))
        layers.append(nn.BatchNorm2d(num_features=64))
        layers.append(nn.ReLU())
        layers.append(nn.Flatten(1, -1))

        self.coop_convs_global = nn.Sequential(*layers)
        summary(self.coop_convs_global,device='cpu',input_size=(input_shape[2], input_shape[0], input_shape[1]))
        ################# global information (2channel) fed into 3layers CNN ####################
        
        ### global and local mixed and fed into MLP
        layers = [
            nn.Linear(self.cfg['cnn_compression'] + 64,128),
            self.activation(),
            nn.Linear(128,64),
            self.activation(),
            nn.Linear(64,32),
            self.activation()          
        ]
        
        self.global_local_mlp = nn.Sequential(*layers)
        summary(self.global_local_mlp,device='cpu',input_size=(self.cfg['cnn_compression'] + 64,))

        self._cur_value = None
        self._cur_value_role = None

        self.freeze_coop_value(self.cfg['freeze_coop_value'])
        self.freeze_greedy_value(self.cfg['freeze_greedy_value'])
        self.freeze_coop(self.cfg['freeze_coop'])
        self.freeze_greedy(self.cfg['freeze_greedy'])

        
        self.view_requirements[REWARDS_ROLE] = ViewRequirement()
        # self.view_requirements[PREV_REWARDS_ROLE] = ViewRequirement()
        self.view_requirements[ACTIONS_ROLE] = ViewRequirement(space=self.action_space['role'])
        # self.view_requirements[PREV_ACTIONS_ROLE] = ViewRequirement(data_col=self.view_requirements[ACTIONS_ROLE],
        #         shift=-1,
        #         space=self.action_space['role'])
        self.view_requirements[ACTIONS_PRIMITIVE] = ViewRequirement(space=self.action_space['primitive'])
        # self.view_requirements[PREV_ACTIONS_PRIMITIVE] = ViewRequirement(data_col=self.view_requirements[ACTIONS_PRIMITIVE],
        #         shift=-1,
        #         space=self.action_space['primitive'])
        self.view_requirements[VF_PREDS_ROLE] = ViewRequirement()

 
        # self.primitive_model = _PrimitiveModel()


    def freeze_coop(self, freeze):
        all_params = \
            list(self.coop_convs.parameters()) + \
            [self.GFL[0].weight1] + \
            list(self.coop_logits.parameters()) + \
            list(self.primitive_coop_convs.parameters()) + \
            list(self.primitive_coop_logits.parameters())

        for param in all_params:
            param.requires_grad = not freeze

    def freeze_greedy(self, freeze):
        all_params = \
            list(self.greedy_logits.parameters()) + \
            list(self.greedy_convs.parameters()) + \
            list(self.primitive_greedy_logits.parameters()) + \
            list(self.primitive_greedy_convs.parameters()) + \
            [self.GFL[0].weight0]

        for param in all_params:
            param.requires_grad = not freeze

    def freeze_greedy_value(self, freeze):
        all_params = \
            list(self.primitive_greedy_value_branch.parameters()) + \
            list(self.primitive_greedy_value_cnns.parameters()) + \
            list(self.primitive_greedy_value_obs_convs.parameters()) +\
            list(self.greedy_value_branch.parameters()) + \
            list(self.greedy_value_cnns.parameters()) + \
            list(self.greedy_value_obs_convs.parameters())

        for param in all_params:
            param.requires_grad = not freeze

    def freeze_coop_value(self, freeze):
        all_params = \
            list(self.primitive_coop_value_cnns.parameters()) + \
            list(self.primitive_coop_value_branch.parameters()) + \
            list(self.primitive_coop_value_obs_convs.parameters()) +\
            list(self.coop_value_cnns.parameters()) + \
            list(self.coop_value_branch.parameters()) + \
            list(self.coop_value_obs_convs.parameters())

        for param in all_params:
            param.requires_grad = not freeze
    

    @override(ModelV2)
    def forward(self,input_dict,state, seq_lens):
        batch_size = input_dict["obs"]['gso'].shape[0]
        o_as = input_dict["obs"]['agents']
        global_as = input_dict["obs"]['merged_states']
        # print('global shape is: {} and state shape is: {} '.format(global_as.shape,input_dict["obs"]["state"].shape))
        # global shape is: torch.Size([32, 25, 25, 2]) and state shape is: torch.Size([32, 25, 25, 5]) 

        gso = input_dict["obs"]['gso'].unsqueeze(1)
        device = gso.device

        for i in range(len(self.GFL)//2):
            self.GFL[i*2].addGSO(gso)

        def role_logits_get():              

            greedy_cnn = self.greedy_convs(o_as[0]['map'].permute(0, 3, 1, 2)) # torch.Size([1, 32])
            coop_agents_cnn_local = {id_agent: self.coop_convs(o_as[id_agent]['map'].permute(0, 3, 1, 2)) for id_agent in range(1, len(o_as))}

            # global feature
            coop_agents_cnn_global = self.coop_convs_global(global_as.permute(0, 3, 1, 2)) # torch.Size([32, 64])
            coop_agents_cnn = {id_agent: self.global_local_mlp(torch.cat([coop_agents_cnn_local[id_agent],coop_agents_cnn_global],dim=1)) for id_agent in range(1, len(o_as))} # torch.Size([32, 32])

            extract_feature_map = torch.zeros(batch_size, self.graph_features, self.n_agents).to(device)
            extract_feature_map[:, :self.cnn_compression, 0] = greedy_cnn
            for id_agent in range(1, len(o_as)):
                extract_feature_map[:, :self.cnn_compression, id_agent] = coop_agents_cnn[id_agent]

            shared_feature = self.GFL(extract_feature_map)
            # for id_agent in range(1, len(o_as)):
            #     this_entity = shared_feature[..., id_agent]

            logits = torch.empty(batch_size, self.n_agents, 2).to(device)
            values = torch.empty(batch_size, self.n_agents).to(device)

            greedy_value_obs_cnn = self.greedy_value_obs_convs(o_as[0]['map'].permute(0, 3, 1, 2))
            coop_value_obs_cnn = {id_agent: self.coop_value_obs_convs(o_as[id_agent]['map'].permute(0, 3, 1, 2)) for id_agent in range(1, len(o_as))}

            if self.cfg['forward_values']:
                greedy_value_cnn = self.greedy_value_cnns(input_dict["obs"]["state"].permute(0, 3, 1, 2))
                coop_value_cnn = self.coop_value_cnns(input_dict["obs"]["state"].permute(0, 3, 1, 2))

                values[:, 0] = self.greedy_value_branch(torch.cat([greedy_value_obs_cnn, greedy_value_cnn], dim=1)).squeeze(1)

            logits_inp = shared_feature[..., 0]
            logits[:, 0] = self.greedy_logits(logits_inp)

            for id_agent in range(1, len(o_as)):
                this_entity = shared_feature[..., id_agent]
                logits[:, id_agent] = self.coop_logits(this_entity)
                if self.cfg['forward_values']:
                    value_cat = torch.cat([coop_value_cnn, coop_value_obs_cnn[id_agent]], dim=1)
                    values[:, id_agent] = self.coop_value_branch(value_cat).squeeze(1)

            self._cur_value_role = values
            role_dist = logits.view(batch_size, self.n_agents*2)

            return role_dist            

        role_dist = role_logits_get()   
        # pdb.set_trace()
        split_inputs = torch.split(role_dist,[2]*len(o_as),dim=1)
    
        role_dist_ = [Categorical(logits=split_inputs[i]) for i in range(len(split_inputs))]
        roles = [role_dist_[i].sample() for i in range(len(split_inputs))] # 5* [32] # roles_action

        # cat the dist and role_action
        roles_explore = []
        # roles_coverage = []
        roles_ = []
        if VARIFY:
            if EXPLORE:
                for l in range(len(roles)):
                    # roles_.append(torch.tensor(roles[l],dtype = torch.float32))
                    roles_.append(torch.full((roles[l].shape),10).to(device))
            elif COVERAGE:
                for l in range(len(roles)):
                    roles_.append(torch.ones_like(roles[l],dtype = torch.float32))
        
            roles = roles_

        # for l in range(len(roles)):
        # #     roles_explore.append(torch.zeros_like(roles[l],dtype=torch.float32))
        ##     roles_coverage.append(torch.ones_like(roles[l],dtype=torch.float32))
        
        role_dist_and_action = torch.cat([role_dist,torch.stack(roles,dim=-1)],dim=-1)

        # local encoder for actor
        # primitive_greedy_cnn = self.primitive_greedy_convs(torch.full((o_as[0]['map'].permute(0,3,1,2).shape),100).float().to(device))
        # primitive_coop_agents_cnn = {id_agent: self.primitive_coop_convs(torch.full((o_as[id_agent]['map'].permute(0,3,1,2).shape),100).float().to(device)) for id_agent in range(1, len(o_as))}
        primitive_greedy_cnn = self.primitive_greedy_convs(o_as[0]['map'].permute(0,3,1,2))
        primitive_coop_agents_cnn = {id_agent: self.primitive_coop_convs(o_as[id_agent]['map'].permute(0,3,1,2)) for id_agent in range(1, len(o_as))}

        # local encoder for critic
        primitive_greedy_value_obs_cnn = self.primitive_greedy_value_obs_convs(o_as[0]['map'].permute(0, 3, 1, 2))
        primitive_coop_value_obs_cnn = {id_agent: self.primitive_coop_value_obs_convs(o_as[id_agent]['map'].permute(0, 3, 1, 2)) for id_agent in range(1, len(o_as))}

        primitive_logits = torch.empty(batch_size, self.n_agents, 5).to(device)
        primitive_logits_coverage = torch.empty(batch_size, self.n_agents, 5).to(device)  # test distribution
        primitive_values = torch.empty(batch_size, self.n_agents).to(device)

        primitive_logits[:, 0] = self.primitive_greedy_logits(torch.cat([primitive_greedy_cnn,roles[0].reshape(-1,1)],dim=1))
        # primitive_logits_coverage[:, 0] = self.primitive_greedy_logits(torch.cat([primitive_greedy_cnn,roles_coverage[0].reshape(-1,1)],dim=1)) # test distribution

        if self.cfg['forward_values']:
            # global encoder for critic
            primitive_greedy_value_cnn = self.primitive_greedy_value_cnns(input_dict["obs"]["state"].permute(0, 3, 1, 2))
            primitive_coop_value_cnn = self.primitive_coop_value_cnns(input_dict["obs"]["state"].permute(0, 3, 1, 2))

            primitive_values[:, 0] = self.primitive_greedy_value_branch(torch.cat([torch.cat([primitive_greedy_value_obs_cnn, primitive_greedy_value_cnn], dim=1),roles[0].reshape(-1,1)],dim=1)).squeeze(1)

        for id_agent in range(1, len(o_as)):
            # pdb.set_trace()
            primitive_logits[:, id_agent] = self.primitive_coop_logits(torch.cat([primitive_coop_agents_cnn[id_agent],roles[id_agent].reshape(-1,1)],dim=1))
            # primitive_logits_coverage[:, id_agent] = self.primitive_coop_logits(torch.cat([primitive_coop_agents_cnn[id_agent],roles_coverage[id_agent].reshape(-1,1)],dim=1)) ####### test distribution
            if self.cfg['forward_values']:
                primitive_value_cat = torch.cat([torch.cat([primitive_coop_value_cnn, primitive_coop_value_obs_cnn[id_agent]],dim=1), roles[id_agent].reshape(-1,1)], dim=1)
                primitive_values[:, id_agent] = self.primitive_coop_value_branch(primitive_value_cat).squeeze(1)

        self._cur_value = primitive_values

        primitive_dist = primitive_logits.view(batch_size, self.n_agents*5)
        # primitive_dist_coverage = primitive_logits_coverage.view(batch_size, self.n_agents*5) ####### test distribution

        # dictObj = {
        # 'explore' :primitive_dist.tolist(),
        # 'coverage' :primitive_dist_coverage.tolist()
        # }
        # jsObj = json.dumps(dictObj)
        
        # fileObject = open('/home/zln/repos/adversarial_comms-master/adversarial_comms/environments/jsonFile.json','w')
        # fileObject.write(jsObj)
        # fileObject.close()

        # pdb.set_trace()
        logist_totle = torch.cat([primitive_dist,role_dist_and_action],dim=-1)
        # pdb.set_trace()
        return logist_totle, state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        values = {}
        values['primitive'] = self._cur_value
        values['role'] = self._cur_value_role
        return values
    
    def value_function_role(self):
        assert self._cur_value_role is not None, "must call forward() first"
        return self._cur_value_role

# class PrimitiveModel(RoleModel, nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         RoleModel.__init__(self, obs_space, action_space['role'], num_outputs, model_config, name)
#         nn.Module.__init__(self)

#         self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
#         self.cfg.update(model_config['custom_model_config'])

#         self.n_agents = len(obs_space.original_space['agents'])
        
#         self.activation = {
#             'relu': nn.ReLU,
#             'leakyrelu': nn.LeakyReLU
#         }[self.cfg['activation']]

#         ########### This is the encoder of local infomation ############
#         layers = []
#         input_shape = obs_space.original_space['agents'][0]['map'].shape
#         (w,h,in_channels) = input_shape

#         in_size = [w, h]
#         for out_channels, kernel, stride in self.cfg['cnn_filters'][:-1]:
#             padding, out_size = same_padding(in_size, kernel, [stride, stride])
#             layers.append(SlimConv2d(in_channels, out_channels, kernel, stride, padding, activation_fn=self.activation))
#             in_channels = out_channels
#             in_size = out_size

#         out_channels, kernel, stride = self.cfg['cnn_filters'][-1]
#         layers.append(
#             SlimConv2d(in_channels, out_channels, kernel, stride,None))
#         layers.append(nn.Flatten(1,-1))

#         self.coop_convs = nn.Sequential(*layers)
#         self.greedy_convs = copy.deepcopy(self.coop_convs)

#         self.coop_value_obs_convs = copy.deepcopy(self.coop_convs)
#         self.greedy_value_obs_convs = copy.deepcopy(self.coop_convs)

#         summary(self.coop_convs, device="cpu", input_size=(input_shape[2], input_shape[0], input_shape[1]))
#         ########### This is the encoder of local infomation ############

#         ########### This is the mlp of local infomation ############
#         logits_inp_features = self.cnn_compression
#         post_logits = [
#             nn.Linear(logits_inp_features, 64),
#             self.activation(),
#             nn.Linear(64, 32),
#             self.activation()
#         ]
#         logit_linear = nn.Linear(32, 2)
#         nn.init.xavier_uniform_(logit_linear.weight)
#         nn.init.constant_(logit_linear.bias, 0)
#         post_logits.append(logit_linear)
#         self.coop_logits = nn.Sequential(*post_logits)
#         self.greedy_logits = copy.deepcopy(self.coop_logits)
#         summary(self.coop_logits, device="cpu", input_size=(logits_inp_features,))
#         ########### This is the mlp of local infomation ############
        
#         ########### This is the cnn of local infomation ############
#         layers = []
#         input_shape = np.array(obs_space.original_space['state'].shape)
#         (w, h, in_channels) = input_shape

#         in_size = [w, h]
#         for out_channels, kernel, stride in self.cfg['value_cnn_filters'][:-1]:
#             padding, out_size = same_padding(in_size, kernel, [stride, stride])
#             layers.append(SlimConv2d(in_channels, out_channels, kernel, stride, padding, activation_fn=self.activation))
#             in_channels = out_channels
#             in_size = out_size

#         out_channels, kernel, stride = self.cfg['value_cnn_filters'][-1]
#         layers.append(
#             SlimConv2d(in_channels, out_channels, kernel, stride, None))
#         layers.append(nn.Flatten(1, -1))

#         self.greedy_value_cnn = nn.Sequential(*layers)
#         self.coop_value_cnn = copy.deepcopy(self.greedy_value_cnn)
#         summary(self.greedy_value_cnn, device="cpu", input_size=(input_shape[2], input_shape[0], input_shape[1]))
#          ########### This is the cnn of local infomation ############

#         ########### This is the cnn of global infomation ############
#         layers = [
#             nn.Linear(self.cfg['value_cnn_compression'], 64),
#             self.activation(),
#             nn.Linear(64, 32),
#             self.activation()
#         ]
#         values_linear = nn.Linear(32, 1)
#         normc_initializer()(values_linear.weight)
#         nn.init.constant_(values_linear.bias, 0)
#         layers.append(values_linear)

#         self.coop_value_branch = nn.Sequential(*layers)
#         self.greedy_value_branch = copy.deepcopy(self.coop_value_branch)
#         summary(self.coop_value_branch, device="cpu", input_size=(self.cfg['value_cnn_compression'],))
#         ########### This is the cnn of global infomation ############

#         self._cur_value = None

#         self.freeze_coop_value(self.cfg['freeze_coop_value'])
#         self.freeze_greedy_value(self.cfg['freeze_greedy_value'])
#         self.freeze_coop(self.cfg['freeze_coop'])
#         self.freeze_greedy(self.cfg['freeze_greedy'])
    
#     @override(ModelV2)
#     def forward(self, input_dict, state, seq_lens, roles):
#         batch_size = input_dict["obs"]['gso'].shape[0]
#         o_as = input_dict["obs"]['agents']

#         gso = input_dict["obs"]['gso'].unsqueeze(1)
#         device = gso.device

#         primitive_greedy_obs_cnn = self.greedy_convs(o_as[0]['map']).permute(0,3,1,2)
#         primitive_coop_agents_obs_cnn = {id_agent: self.coop_convs(o_as[id_agent]['map'].permute(0,3,1,2)) for id_agent in range(1, len(o_as))}

#         primitive_greedy_value_obs_cnn = self.greedy_value_obs_convs(o_as[0]['map'].permute(0, 3, 1, 2))
#         primitive_coop_value_obs_cnn = {id_agent: self.coop_value_obs_convs(o_as[id_agent]['map'].permute(0, 3, 1, 2)) for id_agent in range(1, len(o_as))}

        
#         primitive_logits = torch.empty(batch_size, self.n_agents, 5).to(device)
#         primitive_values = torch.empty(batch_size, self.n_agents).to(device)

#         primitive_logits[:, 0] = self.greedy_logits(primitive_greedy_obs_cnn,roles[0])


#         if self.cfg['forward_values']:
#             primitive_greedy_value_cnn = self.greedy_value_cnn(input_dict["obs"]["state"].permute(0, 3, 1, 2))
#             primitive_coop_value_cnn = self.coop_value_cnn(input_dict["obs"]["state"].permute(0, 3, 1, 2))

#             primitive_values[:, 0] = self.greedy_value_branch(torch.cat([primitive_greedy_value_obs_cnn, primitive_greedy_value_cnn, roles[0]], dim=1)).squeeze(1)

#         for id_agent in range(1, len(o_as)):
#             primitive_logits[:, id_agent] = self.coop_logits(primitive_coop_agents_obs_cnn,roles[1][id_agent])

#             if self.cfg['forward_values']:
#                 primitive_value_cat = torch.cat([primitive_coop_value_cnn, primitive_coop_value_obs_cnn[id_agent], roles[1][id_agent]], dim=1)
#                 primitive_values[:, id_agent] = self.coop_value_branch(primitive_value_cat).squeeze(1)

#         self._cur_value = primitive_values
#         return primitive_logits.view(batch_size, self.n_agents*5), state

#     def freeze_coop(self, freeze):
#         all_params = \
#             list(self.coop_convs.parameters()) + \
#             list(self.coop_logits.parameters())

#         for param in all_params:
#             param.requires_grad = not freeze

#     def freeze_greedy(self, freeze):
#         all_params = \
#             list(self.greedy_logits.parameters()) + \
#             list(self.greedy_convs.parameters())

#         for param in all_params:
#             param.requires_grad = not freeze

#     def freeze_greedy_value(self, freeze):
#         all_params = \
#             list(self.greedy_value_branch.parameters()) + \
#             list(self.greedy_value_cnn.parameters()) + \
#             list(self.greedy_value_obs_convs)

#         for param in all_params:
#             param.requires_grad = not freeze

#     def freeze_coop_value(self, freeze):
#         all_params = \
#             list(self.coop_value_cnn.parameters()) + \
#             list(self.coop_value_branch.parameters()) + \
#             list(self.coop_value_obs_convs)

#         for param in all_params:
#             param.requires_grad = not freeze



# class AdversarialModel():
#     def __init__(self,obs_space, action_space_1, action_space_2,num_outputs_1, num_outputs_2, model_config_1, model_config_2, name_1, name_2):
#         self.role_model = RoleModel(obs_space, action_space_1, num_outputs_1, model_config_1, name_1)
#         self.primitive_model = PrimitiveModel(obs_space, action_space_2, num_outputs_2, model_config_2, name_2)

#     def forward(self,input_dict, state, seq_lens): #加两类
#         role_output = self.role_model(input_dict, state, seq_lens)
#         # roles = role_selector(role_output)      2->1

#         primitive_output = self.primitive_model(input_dict, state, seq_lens, roles.float())

#         return role_output,primitive_output
    
    