import gym
import numpy as np
import tree
from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution,TorchCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
import pdb

torch, nn = try_import_torch()

VARIFY = False
EXPLORE = False
COVERAGE = False


class InvalidActionSpace(Exception):
    """Raised when the action space is invalid"""

    pass


class TorchHomogeneousMultiActionDistribution(TorchMultiActionDistribution):
    def __init__(self, inputs, model, *, child_distributions, input_lens, action_space):
        # pdb.set_trace()
        self.role_action = inputs[:,-5:].long()
        super().__init__(inputs[:,:-5], model, child_distributions=child_distributions, input_lens=input_lens, action_space=action_space)      
        
    
    @override(TorchMultiActionDistribution)
    def deterministic_sample(self):
        # First, sample role_action
        # pdb.set_trace()
        child_distributions = tree.unflatten_as(self.action_space_struct,
                                                self.flat_child_distributions)
        actions = {}
        actions['role'] = []
        actions['primitive'] = tree.map_structure(lambda s: s.deterministic_sample(),child_distributions['primitive'])
        for i in range(len(actions['primitive'])):       
            action_role = self.role_action.squeeze(dim=0)[:,i]
            actions['role'].append(action_role)
        actions['role'] = tuple(actions['role'])


        if VARIFY:
            if COVERAGE:
                for l in range(len(actions['role'])):
                    for j in range(actions['role'][l].shape[0]):
                        actions['role'][l][j] = torch.ones_like(actions['role'][l][j])
            if EXPLORE:
                for l in range(len(actions['role'])):
                    for j in range(actions['role'][l].shape[0]):
                        # pdb.set_trace()
                        actions['role'][l][j] = torch.zeros_like(actions['role'][l][j])
        return actions

    @override(TorchMultiActionDistribution)
    def sample(self):
        # pdb.set_trace()
        child_distributions = tree.unflatten_as(self.action_space_struct,
                                                self.flat_child_distributions)
        
        # device = actions['role'][0][0].device
        actions = {}
        # for key in child_distributions.keys():
        actions['primitive'] = tree.map_structure(lambda s: s.sample(),child_distributions['primitive'])
        actions['role'] = torch.split(self.role_action.squeeze(dim=0),1,dim=-1)
        if VARIFY:
            if COVERAGE:
                for l in range(len(actions['role'])):
                    for j in range(actions['role'][l].shape[0]):
                        actions['role'][l][j] = torch.ones_like(actions['role'][l][j])
            if EXPLORE:
                for l in range(len(actions['role'])):
                    for j in range(actions['role'][l].shape[0]):
                        # actions['role'][l][j] = torch.full((actions[l][j].shape),-50).to(device)
                        actions['role'][l][j] = torch.zeros_like(actions['role'][l][j])
        # print('*'*80)
        # print("sample_actions",actions)
        return actions
    # 还没改
    @override(TorchMultiActionDistribution)
    def logp(self, x):
        child_distributions = tree.unflatten_as(self.action_space_struct,self.flat_child_distributions)
        logps_ = {}
        len = x.shape[1]
        for key in child_distributions.keys():
            logps = []
            for i, (d, action_space) in enumerate(zip(child_distributions[key], self.action_space_struct[key])):
                if isinstance(action_space, gym.spaces.box.Box):
                    assert len(action_space.shape) == 1
                    a_w = action_space.shape[0]
                    x_sel = x[:, a_w * i : a_w * (i + 1)]
                elif isinstance(action_space, gym.spaces.discrete.Discrete):
                    if key == "primitive":
                        x_sel = x[:, i]
                    else:
                        x_sel = x[:, int(len/2)+i]
                else:
                    raise InvalidActionSpace(
                        "Expect gym.spaces.box or gym.spaces.discrete action space"
                    )
                logps.append(d.logp(x_sel))
            logps_[key] = torch.stack(logps, axis=1)

        return logps_

    @override(TorchMultiActionDistribution)
    def entropy(self):

        child_distributions = tree.unflatten_as(self.action_space_struct,self.flat_child_distributions)
        entropy_totle = tree.map_structure(lambda s: s.entropy(),child_distributions)
        entropy_ = {}
        for key in child_distributions.keys():
                entropy_[key] = torch.stack([entropy_totle[key][i] for i in range(len(child_distributions[key]))],dim=-1)
        
        return entropy_
        # return torch.stack(
            # [d.entropy() for d in self.flat_child_distributions], axis=-1
        # )

    @override(TorchMultiActionDistribution)
    def sampled_action_logp(self):
        child_distributions = tree.unflatten_as(self.action_space_struct,self.flat_child_distributions)
        logp = {}
        sampled_action_logp_ = {}

        logp['primitive'] = tree.map_structure(lambda s: s.sampled_action_logp(),child_distributions['primitive'])
        sampled_action_logp_['primitive'] = torch.stack([logp['primitive'][i] for i in range(len(child_distributions['primitive']))],dim=-1)

        sampled_action_logp_['role'] = tuple([child_distributions['role'][i].logp(self.role_action[:,i]) for i in range(len(child_distributions['role']))])

        sampled_action_logp_['role'] = torch.stack([sampled_action_logp_['role'][i] for i in range(len(child_distributions['role']))],dim=-1)
        return sampled_action_logp_

    @override(TorchMultiActionDistribution)
    def kl(self, other):
        child_distributions = tree.unflatten_as(self.action_space_struct,self.flat_child_distributions)
        other_child_distributions = tree.unflatten_as(self.action_space_struct,other.flat_child_distributions)
        
        kls_ = {}
        for key in child_distributions.keys():
            kls = []
            for d, o in zip(child_distributions[key], other_child_distributions[key]):
                kls.append(d.kl(o))
            kls_[key] = torch.stack(kls,axis=-1)

        return kls_