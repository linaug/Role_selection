"""
PyTorch policy class used for PPO.
"""
import gym
import logging
import numpy as np
from typing import Dict, List, Optional, Type, Union
from .constant import *

import ray
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.agents.ppo.ppo_torch_policy import kl_and_loss_stats, \
    vf_preds_fetches, setup_mixins, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, \
    LearningRateSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    convert_to_torch_tensor, explained_variance, sequence_mask
from ray.rllib.utils.typing import TensorType, TrainerConfigDict, AgentID
from ray.rllib.evaluation.postprocessing import discount_cumsum
import sys


import pdb


torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

class InvalidActionSpace(Exception):
    """Raised when the action space is invalid"""
    pass

def compute_advantage_(rollout: SampleBatch, last_r:float, gamma:float, lambda_:float,use_gae: bool = True, use_critic: bool = True):
    #########################  role_GAE  ##################################
    vpred_t_role = np.concatenate([rollout[SampleBatch.VF_PREDS_ROLE], np.array([last_r])])
    delta_t_role = (rollout[SampleBatch.REWARDS_ROLE] + gamma * vpred_t_role[1:] - vpred_t_role[:-1]) # (32,)
    rollout[Postprocessing.ADVANTAGES_ROLE] = discount_cumsum(delta_t_role, gamma * lambda_)   # (32,)
    rollout[Postprocessing.VALUE_TARGETS_ROLE] = (rollout[Postprocessing.ADVANTAGES_ROLE] + rollout[SampleBatch.VF_PREDS_ROLE]).astype(np.float32)
    rollout[Postprocessing.ADVANTAGES_ROLE] = rollout[Postprocessing.ADVANTAGES_ROLE].astype(np.float32)          

    #########################  primitive_GAE  ##################################
    vpred_t = np.concatenate([rollout[SampleBatch.VF_PREDS],np.array([last_r])])          
    delta_t = (rollout[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1]) # (32,)
    rollout[Postprocessing.ADVANTAGES] = discount_cumsum(delta_t, gamma * lambda_)
    rollout[Postprocessing.VALUE_TARGETS] = (rollout[Postprocessing.ADVANTAGES] + rollout[SampleBatch.VF_PREDS]).astype(np.float32) # (32,)
    rollout[Postprocessing.ADVANTAGES] = rollout[Postprocessing.ADVANTAGES].astype(np.float32)
    # pdb.set_trace()
    return rollout


def compute_gae_for_sample_batch(
        policy: Policy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
        episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:
    """Adds GAE (generalized advantage estimations) to a trajectory.
    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.
    Args:
        policy (Policy): The Policy used to generate the trajectory
            (`sample_batch`)
        sample_batch (SampleBatch): The SampleBatch to postprocess.
        other_agent_batches (Optional[Dict[PolicyID, SampleBatch]]): Optional
            dict of AgentIDs mapping to other agents' trajectory data (from the
            same episode). NOTE: The other agents use the same policy.
        episode (Optional[MultiAgentEpisode]): Optional multi-agent episode
            object in which the agents operated.
    Returns:
        SampleBatch: The postprocessed, modified SampleBatch (or a new one).
    """

    # the trajectory view API will pass populate the info dict with a np.zeros((n,))
    # array in the first call, in that case the dtype will be float32 and we
    # have to ignore it. For regular calls, we extract the rewards from the info
    # dict into the samplebatch_infos_rewards dict, which now holds the rewards
    # for all agents as dict.
    # pdb.set_trace()
    infos = sample_batch.get(SampleBatch.INFOS)
    if (infos is not None) and (infos[0] != 0):
        # ===== Fill rewards from data =====
        sample_batch[REWARDS_ROLE] = np.array([info["rewards_role"] for info in infos])

    samplebatch_infos_rewards = {'0': sample_batch[SampleBatch.INFOS]}
    samplebatch_infos_rewards_role = {'0': sample_batch[SampleBatch.INFOS]}
    if not sample_batch[SampleBatch.INFOS].dtype == "float32":
        samplebatch_infos = SampleBatch.concat_samples([
            SampleBatch({k: [v] for k, v in s.items()})
            for s in sample_batch[SampleBatch.INFOS]
        ])
        samplebatch_infos_rewards = SampleBatch.concat_samples([
            SampleBatch({str(k): [v] for k, v in s.items()})
            for s in samplebatch_infos["rewards"]
        ])
        samplebatch_infos_rewards_role = SampleBatch.concat_samples([
            SampleBatch({str(k): [v] for k, v in s.items()})
            for s in samplebatch_infos["rewards"]
        ])

    if not isinstance(policy.action_space, gym.spaces.Dict):
        raise InvalidActionSpace("Expect tuple action space")
    
    # pdb.set_trace()
    len = sample_batch[SampleBatch.ACTIONS].shape[1]
    sample_batch[SampleBatch.ACTIONS_ROLE] = sample_batch[SampleBatch.ACTIONS][:,int(len/2):]
    sample_batch[SampleBatch.ACTIONS_PRIMITIVE] = sample_batch[SampleBatch.ACTIONS][:,:int(len/2)]

    r_shape = sample_batch[SampleBatch.VF_PREDS].shape
    # print("r_shape is {} and role_shape is {}".format(r_shape,sample_batch[SampleBatch.VF_PREDS_ROLE].shape))
    if not sample_batch[SampleBatch.VF_PREDS_ROLE].shape == r_shape:
        aa = []
        for i in range(r_shape[0]):
            aa.append(sample_batch[SampleBatch.VF_PREDS_ROLE][i].tolist())
        aa_ = np.array(aa).reshape(r_shape[0],r_shape[1])
    
        sample_batch[SampleBatch.VF_PREDS_ROLE] = aa_
    # samplebatches for each agents
    batches = []
    for key, action_space, action_space_role in zip(samplebatch_infos_rewards.keys(), policy.action_space['primitive'],policy.action_space['role']):
        i = int(key)
        sample_batch_agent = sample_batch.copy()
        sample_batch_agent[SampleBatch.REWARDS] = (samplebatch_infos_rewards[key])
        sample_batch_agent[SampleBatch.REWARDS_ROLE] = (samplebatch_infos_rewards_role[key])
        # sample_batch_agent[SampleBatch.REWARDS_ROLE] = (samplebatch_infos_rewards_role[key])
        if isinstance(action_space, gym.spaces.box.Box):
            assert len(action_space.shape) == 1
            a_w = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.discrete.Discrete):
            a_w = 1
        else:
            raise InvalidActionSpace("Expect gym.spaces.box or gym.spaces.discrete action space")
        sample_batch_agent[SampleBatch.ACTIONS_ROLE] = sample_batch[SampleBatch.ACTIONS_ROLE][:, a_w * i : a_w * (i + 1)]
        sample_batch_agent[SampleBatch.ACTIONS_PRIMITIVE] = sample_batch[SampleBatch.ACTIONS_PRIMITIVE][:, a_w * i : a_w * (i + 1)]
        sample_batch_agent[SampleBatch.VF_PREDS] = sample_batch[SampleBatch.VF_PREDS][:, i]
        sample_batch_agent[SampleBatch.VF_PREDS_ROLE] = sample_batch[SampleBatch.VF_PREDS_ROLE][:, i]

        # Trajectory is actually complete -> last r=0.0.
        if sample_batch[SampleBatch.DONES][-1]:
            last_r = last_r_role = 0.0
        # Trajectory has been truncated -> last r=VF estimate of last obs.
        else:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.
            # Create an input dict according to the Model's requirements.
            input_dict = policy.model.get_input_dict(
                sample_batch, index="last")
            all_values = policy._value(**input_dict, seq_lens=input_dict.seq_lens)
            last_r = all_values['primitive'][0][i].item()
            last_r_role = all_values['role'][0][i].item()
        
        # Adds the policy logits, VF preds, and advantages to the batch,
        # using GAE ("generalized advantage estimation") or not.
        batches.append(
            compute_advantage_(
                sample_batch_agent,
                last_r,
                policy.config["gamma"],
                policy.config["lambda"],
                use_gae=policy.config["use_gae"],
                use_critic=policy.config.get("use_critic", True)
            )
        )
    # Now take original samplebatch and overwrite following elements as a concatenation of these
    for k in [
        SampleBatch.REWARDS,
        SampleBatch.REWARDS_ROLE,
        SampleBatch.VF_PREDS,
        SampleBatch.VF_PREDS_ROLE,
        Postprocessing.ADVANTAGES,
        Postprocessing.ADVANTAGES_ROLE,
        Postprocessing.VALUE_TARGETS,
        Postprocessing.VALUE_TARGETS_ROLE,
    ]:
        sample_batch[k] = np.stack([b[k] for b in batches], axis=-1)
    return sample_batch


def ppo_surrogate_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective.
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # pdb.set_trace()
    logits, state = model.from_batch(train_batch, is_training=True)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch["seq_lens"])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch["seq_lens"],
            max_seq_len,
            time_major=model.is_time_major())
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    loss_data = []
    curr_action_dist = dist_class(logits, model)
    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS],
                                  model)
    logps = curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    entropies = curr_action_dist.entropy()
    action_kl = prev_action_dist.kl(curr_action_dist)
    mean_kl_primitive = reduce_mean_valid(torch.sum(action_kl['primitive'], axis=1))
    mean_kl_role = reduce_mean_valid(torch.sum(action_kl['role'], axis=1))

    mylog = open('/home/zln/ray_results/recode.log',mode='a',encoding='utf-8')
    for i in range(len(train_batch[SampleBatch.VF_PREDS][0])):
        lens_ = train_batch[SampleBatch.VF_PREDS].shape
        # print('-*'*80)
        # print("len is {} \n i is {} \n and logps shape is {}".format(lens_, i,logps['primitive'].shape))
        # print('-*'*80)
        logp_ratio_primitive = torch.exp(
            logps['primitive'][:, i] -
            train_batch[SampleBatch.ACTION_LOGP][:, i])
        
        logp_ratio_role = torch.exp(
            logps['role'][:, i] -
            train_batch[SampleBatch.ACTION_LOGP_ROLE][:, i])

        mean_entropy_primitive = reduce_mean_valid(entropies['primitive'][:, i])
        mean_entropy_role = reduce_mean_valid(entropies['role'][:, i])

        SIL = True
        if SIL:
            SIL_ADVANTAGE = torch.where(train_batch[Postprocessing.ADVANTAGES][..., i]>0,5*train_batch[Postprocessing.ADVANTAGES][..., i]>0,train_batch[Postprocessing.ADVANTAGES][..., i]>0)
            SIL_ADVANTAGE_ROLE = torch.where(train_batch[Postprocessing.ADVANTAGES_ROLE][..., i]>0,5*train_batch[Postprocessing.ADVANTAGES_ROLE][..., i]>0,train_batch[Postprocessing.ADVANTAGES_ROLE][..., i]>0)
            surrogate_loss_primitive = torch.min(
                SIL_ADVANTAGE * logp_ratio_primitive,
                SIL_ADVANTAGE * torch.clamp(
                    logp_ratio_primitive, 1 - policy.config["clip_param"],
                    1 + policy.config["clip_param"]))
            
            surrogate_loss_role = torch.min(
                SIL_ADVANTAGE_ROLE * logp_ratio_role,
                SIL_ADVANTAGE_ROLE * torch.clamp(
                    logp_ratio_role, 1 - policy.config["clip_param"],
                    1 + policy.config["clip_param"]))
        else:
            surrogate_loss_primitive = torch.min(
                train_batch[Postprocessing.ADVANTAGES][..., i] * logp_ratio_primitive,
                train_batch[Postprocessing.ADVANTAGES][..., i] * torch.clamp(
                    logp_ratio_primitive, 1 - policy.config["clip_param"],
                    1 + policy.config["clip_param"]))
            
            surrogate_loss_role = torch.min(
                train_batch[Postprocessing.ADVANTAGES_ROLE][..., i] * logp_ratio_role,
                train_batch[Postprocessing.ADVANTAGES_ROLE][..., i] * torch.clamp(
                    logp_ratio_role, 1 - policy.config["clip_param"],
                    1 + policy.config["clip_param"]))
        
        mean_policy_loss_primitive = reduce_mean_valid(-surrogate_loss_primitive)

        mean_policy_loss_role = reduce_mean_valid(-surrogate_loss_role)

        if policy.config["use_gae"]:
            ################### primitive ########################
            # pdb.set_trace()
            prev_value_fn_out_primitive = train_batch[SampleBatch.VF_PREDS][..., i]

            value_fn_out_primitive = model.value_function()['primitive'][..., i]
            
            valid_value_data = True
            if valid_value_data:
                batch_value = value_fn_out_primitive.shape[0]
                TD_error = value_fn_out_primitive - train_batch[Postprocessing.VALUE_TARGETS][..., i]
                yuzhi = 0.05 * train_batch[Postprocessing.VALUE_TARGETS][..., i]
                valid = TD_error < yuzhi
                print('There are {} valid data in batch {} of {}'.format((valid==True).sum().item(),batch_value,i),file=mylog)       

            vf_loss1_primitive = torch.pow(value_fn_out_primitive - train_batch[Postprocessing.VALUE_TARGETS][..., i], 2.0)

            vf_clipped_primitive = prev_value_fn_out_primitive + torch.clamp(value_fn_out_primitive - prev_value_fn_out_primitive, -policy.config["vf_clip_param"],policy.config["vf_clip_param"])

            vf_loss2_primitive = torch.pow(vf_clipped_primitive - train_batch[Postprocessing.VALUE_TARGETS][..., i], 2.0)

            vf_loss_primitive = torch.max(vf_loss1_primitive, vf_loss2_primitive)

            mean_vf_loss_primitive = reduce_mean_valid(vf_loss_primitive)
            total_loss_primitive = reduce_mean_valid(
                -surrogate_loss_primitive + policy.kl_coeff * action_kl['primitive'][:, i] +
                policy.config["vf_loss_coeff"] * vf_loss_primitive -
                policy.entropy_coeff * entropies['primitive'][:, i])
            ################### primitive ########################

            ################### role ########################
            prev_value_fn_out_role = train_batch[SampleBatch.VF_PREDS_ROLE][..., i]
            value_fn_out_role = model.value_function()['role'][..., i]
            vf_loss1_role = torch.pow(
                value_fn_out_role - train_batch[Postprocessing.VALUE_TARGETS_ROLE][..., i], 2.0)
            vf_clipped_role = prev_value_fn_out_role + torch.clamp(
                value_fn_out_role - prev_value_fn_out_role, -policy.config["vf_clip_param"],
                policy.config["vf_clip_param"])
            vf_loss2_role = torch.pow(
                vf_clipped_role - train_batch[Postprocessing.VALUE_TARGETS_ROLE][..., i], 2.0)
            vf_loss_role = torch.max(vf_loss1_role, vf_loss2_role)
            mean_vf_loss_role = reduce_mean_valid(vf_loss_role)
            total_loss_role = reduce_mean_valid(
                -surrogate_loss_role + policy.kl_coeff * action_kl['role'][:, i] +
                policy.config["vf_loss_coeff"] * vf_loss_role -
                policy.entropy_coeff * entropies['role'][:, i])
            ################### role ########################
        else:
            mean_vf_loss = 0.0
            total_loss = reduce_mean_valid(-surrogate_loss_primitive +
                                           policy.kl_coeff * action_kl[:, i] -
                                           policy.entropy_coeff * entropies[:, i])

        # Store stats in policy for stats_fn.
        # pdb.set_trace()
        total_loss = total_loss_primitive + total_loss_role
        mean_policy_loss = mean_policy_loss_primitive + mean_policy_loss_role
        mean_vf_loss = mean_vf_loss_primitive + mean_vf_loss_role
        mean_entropy = mean_entropy_primitive + mean_entropy_role
        loss_data.append(
            {
                "total_loss": total_loss,
                "mean_policy_loss": mean_policy_loss,
                "mean_vf_loss": mean_vf_loss,
                "mean_entropy": mean_entropy,
            }
        )
    mylog.close()
    policy._total_loss = (torch.sum(torch.stack([o["total_loss"] for o in loss_data])),)
    policy._mean_policy_loss = torch.mean(
        torch.stack([o["mean_policy_loss"] for o in loss_data])
    )
    policy._mean_vf_loss = torch.mean(
        torch.stack([o["mean_vf_loss"] for o in loss_data])
    )
    policy._mean_entropy = torch.mean(
        torch.stack([o["mean_entropy"] for o in loss_data])
    )
    policy._vf_explained_var = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS][:,0].reshape(-1,1),
        policy.model.value_function()['primitive'])
    policy._mean_kl = mean_kl_primitive + mean_kl_role

    return policy._total_loss

def postprocess_fn(self, sample_batch, other_agent_batches=None, episode=None):
    
    return sample_batch


class ValueNetworkMixin:
    """This is exactly the same mixin class as in ppo_torch_policy,
    but that one calls .item() on self.model.value_function()[0],
    which will not work for us since our value function returns
    multiple values. Instead, we call .item() in
    compute_gae_for_sample_batch above.
    """

    def __init__(self, obs_space, action_space, config):
        if config["use_gae"]:

            def value(**input_dict):
                # pdb.set_trace()
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                value_ = self.model.value_function()
                return value_

        else:

            def value(*args, **kwargs):
                return 0.0,0.0
        self._value = value



def setup_mixins_override(policy: Policy, obs_space: gym.spaces.Space,
                          action_space: gym.spaces.Space,
                          config: TrainerConfigDict) -> None:
    """Have to initialize the custom ValueNetworkMixin
    """
    setup_mixins(policy, obs_space, action_space, config)
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)

# def make_model(policy: Policy, obs_space: gym.spaces.Space,
#                           action_space: gym.spaces.Space,
#                           config: TrainerConfigDict):
    
#     # dist_class, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])

#     # assert config["model"]["custom_model"]

#     policy.model =  RoleModel(obs_space, action_space['role'], None, config['model'], 'role_model')
#     policy.primitive_model = PrimitiveModel(obs_space, action_space['primitive'], None, config['model'], 'primitive_model')

#     return policy.model

# Build a child class of `TorchPolicy`, given the custom functions defined
# above.
MultiPPOTorchPolicy = build_policy_class(
    name="MultiPPOTorchPolicy",
    framework="torch",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=compute_gae_for_sample_batch,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    before_loss_init=setup_mixins_override,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ],
)

def get_policy_class(config):
    return MultiPPOTorchPolicy

MultiPPOTrainer = build_trainer(
    name="MultiPPO",
    default_config=ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    validate_config=ray.rllib.agents.ppo.ppo.validate_config,
    default_policy=MultiPPOTorchPolicy,
    get_policy_class=get_policy_class,
    execution_plan=ray.rllib.agents.ppo.ppo.execution_plan
)
