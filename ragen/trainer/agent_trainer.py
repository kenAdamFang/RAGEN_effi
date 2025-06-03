"""
FSDP PPO Trainer with Ray-based single controller.
Adapted from the excellently written verl implementation.
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from ragen.trainer import core_algos
# from ragen.trainer.core_algos import agg_loss # Not used in this file directly
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]


from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager, compute_response_mask, _timer, apply_kl_penalty, AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as VerlRayPPOTrainer


from ragen.llm_agent.agent_proxy import LLMAgentProxy
from ragen.utils import GenerationsLogger

from ragen.trainer.replay_buffer import ReplayBuffer
import time


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, bi_level_gae=False, high_level_gamma=1.0):
    if not hasattr(data, 'batch') or data.batch is None: data.batch = {} # Ensure data.batch exists

    if "response_mask" not in data.batch or data.batch["response_mask"] is None:
        data.batch["response_mask"] = compute_response_mask(data)

    # Check for essential keys; PPO pipeline should ensure they are populated by this stage.
    # If any are missing, it indicates a deeper issue in the pipeline order.
    # For example: data.batch.get("token_level_rewards") would be None if missing.
    # Downstream core_algos will fail if these are None.

    if adv_estimator == AdvantageEstimator.GAE:
        if bi_level_gae:
            advantages, returns = core_algos.compute_bi_level_gae_advantage_return(
                token_level_rewards=data.batch.get("token_level_rewards"), values=data.batch.get("values"),
                loss_mask=data.batch.get("response_mask"), gamma=gamma, lam=lam, high_level_gamma=high_level_gamma,
            )
        else:
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=data.batch.get("token_level_rewards"), values=data.batch.get("values"),
                response_mask=data.batch.get("response_mask"), gamma=gamma, lam=lam,
            )
        data.batch["advantages"] = advantages; data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data.batch.get("response_mask")
        if multi_turn:
            # Ensure loss_mask exists if multi_turn is True
            loss_mask = data.batch.get("loss_mask")
            if loss_mask is None: raise ValueError("loss_mask is required for GRPO multi-turn advantage calculation.")
            response_length = grpo_calculation_mask.size(1); grpo_calculation_mask = loss_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch.get("token_level_rewards"), response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch.get("uid"), norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages; data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch.get("token_level_rewards"), response_mask=data.batch.get("response_mask"),
            index=data.non_tensor_batch.get("uid"),
        )
        data.batch["advantages"] = advantages; data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch.get("token_level_rewards"), response_mask=data.batch.get("response_mask"), gamma=gamma,
        )
        data.batch["advantages"] = advantages; data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch.get("token_level_rewards"), reward_baselines=data.batch.get("reward_baselines"),
            response_mask=data.batch.get("response_mask"),
        )
        data.batch["advantages"] = advantages; data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch.get("token_level_rewards"), response_mask=data.batch.get("response_mask"),
            index=data.non_tensor_batch.get("uid"),
        )
        data.batch["advantages"] = advantages; data.batch["returns"] = returns
    else: raise NotImplementedError(f"Unknown advantage estimator: {adv_estimator}")
    return data

def _filter_rollout(batch, config_obj):
    rollout_filter_ratio = config_obj.actor_rollout_ref.rollout.rollout_filter_ratio
    num_groups = config_obj.es_manager.train.env_groups
    group_size = config_obj.es_manager.train.group_size
    dummy_metrics = {"rollout/in_group_std": 0.0, "rollout/in_group_max": 0.0, "rollout/in_group_mean": 0.0,
                     "rollout/chosen_in_group_std": 0.0, "rollout/chosen_in_group_max": 0.0, "rollout/chosen_in_group_mean": 0.0}

    # Ensure batch and relevant keys exist, return a deepcopy if checks fail
    if not hasattr(batch, 'batch') or batch.batch is None or \
       'original_rm_scores' not in batch.batch or batch.batch['original_rm_scores'] is None:
        return deepcopy(batch), dummy_metrics

    expected_elements = num_groups * group_size
    original_scores_tensor = batch.batch['original_rm_scores']
    current_elements = original_scores_tensor.shape[0] if original_scores_tensor is not None else 0

    if original_scores_tensor.numel() == 0 or current_elements != expected_elements:
        return deepcopy(batch), dummy_metrics

    rm_scores = original_scores_tensor.sum(dim=-1).view(num_groups, group_size)
    in_group_std = rm_scores.std(dim=-1); in_group_max = rm_scores.max(dim=-1).values; in_group_mean = rm_scores.mean(dim=-1)

    metrics = {"rollout/in_group_std": in_group_std.mean().item(),
               "rollout/in_group_max": in_group_max.mean().item(),
               "rollout/in_group_mean": in_group_mean.mean().item()}

    if rollout_filter_ratio == 1.0:
        metrics.update({"rollout/chosen_in_group_std": metrics["rollout/in_group_std"],
                        "rollout/chosen_in_group_max": metrics["rollout/in_group_max"],
                        "rollout/chosen_in_group_mean": metrics["rollout/in_group_mean"]})
        return deepcopy(batch), metrics # Return a deepcopy

    k_topk = max(1, int(rollout_filter_ratio * num_groups))
    if k_topk > num_groups : k_topk = num_groups

    if config_obj.actor_rollout_ref.rollout.rollout_filter_type == "std_rev":
        top_groups = (-in_group_std).topk(k_topk).indices
    elif config_obj.actor_rollout_ref.rollout.rollout_filter_type == "std":
        top_groups = in_group_std.topk(k_topk).indices
    else: raise ValueError(f"Invalid rollout filter type: {config_obj.actor_rollout_ref.rollout.rollout_filter_type}")

    mask = torch.zeros(num_groups, dtype=torch.bool, device=rm_scores.device); mask[top_groups] = True
    mask = mask.unsqueeze(1).expand(-1, group_size).flatten()

    filtered_batch_data = {}
    if hasattr(batch, 'batch') and batch.batch is not None:
        for key_b, value_b in batch.batch.items():
            if isinstance(value_b, torch.Tensor) and value_b.shape[0] == expected_elements:
                filtered_batch_data[key_b] = value_b[mask]
            else: # Preserve items that are not tensors or don't match expected_elements
                filtered_batch_data[key_b] = value_b

    filtered_non_tensor_data = {}
    if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch is not None:
        cpu_mask = None # Initialize cpu_mask
        # Check if mask is on GPU and needs to be moved for numpy
        if mask.device.type != 'cpu':
             cpu_mask = mask.cpu().numpy()
        else:
             cpu_mask = mask.numpy()

        for key_ntb, value_ntb in batch.non_tensor_batch.items():
            try:
                if isinstance(value_ntb, np.ndarray) and value_ntb.shape[0] == expected_elements:
                    filtered_non_tensor_data[key_ntb] = value_ntb[cpu_mask]
                elif isinstance(value_ntb, list) and len(value_ntb) == expected_elements:
                    filtered_non_tensor_data[key_ntb] = [v for v, m_val in zip(value_ntb, mask.tolist()) if m_val]
                else: # Preserve items that are not np.ndarray/list or don't match expected_elements
                    filtered_non_tensor_data[key_ntb] = value_ntb
            except Exception as e:
                print(f"Error filtering non_tensor_batch key {key_ntb} in _filter_rollout: {e}")
                filtered_non_tensor_data[key_ntb] = value_ntb # Preserve original on error

    copied_meta_info = deepcopy(batch.meta_info) if hasattr(batch, 'meta_info') and batch.meta_info is not None else {}

    metrics.update({"rollout/chosen_in_group_std": in_group_std[top_groups].mean().item(),
                    "rollout/chosen_in_group_max": in_group_max[top_groups].mean().item(),
                    "rollout/chosen_in_group_mean": in_group_mean[top_groups].mean().item()})

    final_td_batch = None
    if filtered_batch_data:
        # Determine common_batch_size from the filtered tensors
        common_batch_size = None
        first_tensor_val = None
        for tensor_val in filtered_batch_data.values():
            if isinstance(tensor_val, torch.Tensor):
                first_tensor_val = tensor_val # Store the first tensor encountered
                break

        if first_tensor_val is not None:
            common_batch_size = first_tensor_val.shape[:1] # e.g., torch.Size([N_filtered])
            try:
                # Filter out non-tensor items that might have slipped through if not filtered by shape earlier
                # Or ensure all items intended for TensorDict are indeed tensors.
                # For safety, create a new dict with only tensors for TensorDict source
                tensor_source_dict = {k: v for k, v in filtered_batch_data.items() if isinstance(v, torch.Tensor)}
                if not tensor_source_dict: # If, after filtering, no tensors remain
                    final_td_batch = None
                else:
                    # Validate that all tensors in tensor_source_dict share the common_batch_size
                    # This is implicitly done by TensorDict constructor if batch_size is passed,
                    # but good to be aware. If a tensor doesn't match, it will raise an error.
                    final_td_batch = TensorDict(source=tensor_source_dict, batch_size=common_batch_size)
            except Exception as e:
                print(f"Error creating TensorDict in _filter_rollout: {e}. Filtered_batch_data keys: {list(filtered_batch_data.keys())}, Common batch size: {common_batch_size}")
                # Fallback: final_td_batch remains None
        # else: no tensors found in filtered_batch_data, final_td_batch remains None
    # If filtered_batch_data was empty from the start, final_td_batch also remains None

    return DataProto(batch=final_td_batch, # This is now a TensorDict or None
                     non_tensor_batch=filtered_non_tensor_data or None,
                     meta_info=copied_meta_info), metrics

class RayAgentTrainer(VerlRayPPOTrainer):
    def __init__(self, config, tokenizer, role_worker_mapping: dict[Role, WorkerType], resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup, processor=None, reward_fn=None, val_reward_fn=None):
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn)
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0
        self.generations_logger = GenerationsLogger()
        if self.config.get('replay_buffer') and self.config.replay_buffer.enable:
            self.replay_buffer = ReplayBuffer(capacity=self.config.replay_buffer.capacity)
            print(f"Replay buffer enabled: capacity={self.config.replay_buffer.capacity}, sampling_batch_size={self.config.replay_buffer.sampling_batch_size}")
        else:
            self.replay_buffer = None; print("Replay buffer disabled.")
        
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        assert self.config.trainer.total_training_steps is not None, "must determine total training steps"
        total_training_steps = self.config.trainer.total_training_steps; self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"): self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"): self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e: print(f"Warning: Could not set total_training_steps in config. Error: {e}")

    def init_agent_proxy(self): self.agent_proxy = LLMAgentProxy(config=self.config, actor_rollout_wg=self.actor_rollout_wg, tokenizer=self.tokenizer)

    def _maybe_log_generations(self, inputs, outputs, scores, _type="val"):
        generations_to_log = self.config.trainer.generations_to_log_to_wandb.get(_type, 0) # Use .get for safety
        if generations_to_log == 0: return
        samples = list(zip(inputs, outputs, scores)); samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42); rng.shuffle(samples)
        samples = samples[:generations_to_log]
        self.generations_logger.log(self.config.trainer.logger, samples, self.global_steps, _type)

    def _validate(self):
        data_source_lst, reward_extra_infos_dict, sample_inputs, sample_outputs, sample_scores, env_metric_dict = [], defaultdict(list), [], [], [], {}
        for _ in range(self.config.trainer.validation_steps):
            sample_inputs.extend([""] * (self.config.es_manager.val.env_groups * self.config.es_manager.val.group_size))
            meta_info = {"eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id, "recompute_log_prob": False,
                         "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample, "validate": True}
            test_gen_batch = DataProto(batch=None, non_tensor_batch=None, meta_info=meta_info)
            val_st = time.time(); test_batch = self.agent_proxy.rollout(test_gen_batch, val=True); print(f"Val gen time: {time.time() - val_st:.2f}s")

            if hasattr(test_batch, 'meta_info') and test_batch.meta_info is not None and "metrics" in test_batch.meta_info:
                for k, v in test_batch.meta_info["metrics"].items(): env_metric_dict.setdefault("val-env/" + k, []).append(v)

            current_batch_responses = []
            if hasattr(test_batch, 'batch') and test_batch.batch is not None and 'responses' in test_batch.batch and test_batch.batch['responses'] is not None:
                current_batch_responses = test_batch.batch["responses"]
                sample_outputs.extend([self.tokenizer.decode(ids, skip_special_tokens=True) for ids in current_batch_responses])

            result = self.val_reward_fn(test_batch, return_dict=True); scores_val = result["reward_tensor"].sum(-1).cpu().tolist(); sample_scores.extend(scores_val)
            reward_extra_infos_dict["reward"].extend(scores_val)
            if "reward_extra_info" in result:
                for k, lst in result["reward_extra_info"].items(): reward_extra_infos_dict[k].extend(lst)

            num_responses_for_ds = len(current_batch_responses) if current_batch_responses is not None else len(scores_val)
            if hasattr(test_batch, 'non_tensor_batch') and test_batch.non_tensor_batch:
                 data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * num_responses_for_ds))
            elif num_responses_for_ds > 0 : # If non_tensor_batch is missing but we have responses
                 data_source_lst.append(["unknown"] * num_responses_for_ds)


        self._maybe_log_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, _type="val")
        validation_data_dir = self.config.trainer.get("validation_data_dir") # .get is safer
        if validation_data_dir: self._dump_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, reward_extra_infos_dict=reward_extra_infos_dict, dump_path=validation_data_dir)

        for k_info, lst_info in reward_extra_infos_dict.items(): assert not lst_info or len(lst_info) == len(sample_scores), f"{k_info}: {len(lst_info)=}, {len(sample_scores)=}"

        ds_val = np.concatenate(data_source_lst) if data_source_lst and any(isinstance(el, (list, np.ndarray)) for el in data_source_lst) else np.array([]) # Ensure non-empty before concat
        data_src2var2metric2val = process_validation_metrics(ds_val, sample_inputs, reward_extra_infos_dict) # Ensure ds_val is correctly shaped
        metric_dict_val = reduce_metrics(env_metric_dict)

        for ds, v2m2v in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in v2m2v else "reward"
            for var_n, m2v in v2m2v.items():
                n_max = 1
                if m2v:
                    n_max_k = [str(n).split("@")[-1].split("/")[0] for n in m2v.keys() if "@" in str(n)] # Ensure n is str
                    n_max = max((int(k) for k in n_max_k if k.isdigit()), default=1)
                for met_n, met_v in m2v.items():
                    met_s = "val-core" if (var_n == core_var and any(str(met_n).startswith(pfx) for pfx in ["mean","maj","best"]) and (f"@{n_max}"in str(met_n))) else "val-aux" # Ensure met_n is str
                    metric_dict_val[f"{met_s}/{ds}/{var_n}/{met_n}"] = met_v
        return metric_dict_val

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool(); self.resource_pool_to_cls = {p: {} for p in self.resource_pool_manager.resource_pool_dict.values()}
        if self.hybrid_engine: rp_ar = self.resource_pool_manager.get_resource_pool(Role.ActorRollout); self.resource_pool_to_cls[rp_ar]["actor_rollout"] = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.actor_rollout_ref, role="actor_rollout")
        else: raise NotImplementedError(f"Unsupported engine: hybrid_engine={self.hybrid_engine}")
        if self.use_critic: rp_c = self.resource_pool_manager.get_resource_pool(Role.Critic); self.resource_pool_to_cls[rp_c]["critic"] = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
        if self.use_reference_policy and not self.ref_in_actor: rp_ref = self.resource_pool_manager.get_resource_pool(Role.RefPolicy); self.resource_pool_to_cls[rp_ref]["ref"] = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
        if self.use_rm: rp_rm = self.resource_pool_manager.get_resource_pool(Role.RewardModel); self.resource_pool_to_cls[rp_rm]["rm"] = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
        all_wg = {}; self.wg_dicts = []; wg_kwargs = {}; timeout = OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout", default=None);
        if timeout is not None: wg_kwargs["ray_wait_register_center_timeout"] = timeout
        for rp, cd in self.resource_pool_to_cls.items(): wdc = create_colocated_worker_cls(class_dict=cd); wg_d = self.ray_worker_group_cls(resource_pool=rp, ray_cls_with_init=wdc, **wg_kwargs); spawn_wg = wg_d.spawn(prefix_set=cd.keys()); all_wg.update(spawn_wg); self.wg_dicts.append(wg_d)
        if self.use_critic: self.critic_wg = all_wg["critic"]; self.critic_wg.init_model()
        if self.use_reference_policy and not self.ref_in_actor: self.ref_policy_wg = all_wg["ref"]; self.ref_policy_wg.init_model()
        if self.use_rm: self.rm_wg = all_wg["rm"]; self.rm_wg.init_model()
        self.actor_rollout_wg = all_wg["actor_rollout"]; self.actor_rollout_wg.init_model()
        self.async_rollout_mode = (self.config.actor_rollout_ref.rollout.mode == "async")
        if self.async_rollout_mode: self.async_rollout_manager = AsyncLLMServerManager(config=self.config.actor_rollout_ref, worker_group=self.actor_rollout_wg)

    def _save_checkpoint(self):
        local_step_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        actor_path = os.path.join(local_step_dir, "actor")
        actor_remote = os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor") if self.config.trainer.default_hdfs_dir else None
        remove_prev = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        max_actor_k = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_prev else 1
        max_critic_k = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_prev else 1
        self.actor_rollout_wg.save_checkpoint(actor_path, actor_remote, self.global_steps, max_ckpt_to_keep=max_actor_k)
        if self.use_critic:
            critic_path = os.path.join(local_step_dir, "critic")
            critic_remote = os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic") if self.config.trainer.default_hdfs_dir else None
            self.critic_wg.save_checkpoint(critic_path, critic_remote, self.global_steps, max_ckpt_to_keep=max_critic_k)
        with open(os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"), "w") as f: f.write(str(self.global_steps))

    def _combine_data_protos(self, data_proto_list: list[DataProto]) -> DataProto:
        if not data_proto_list: return DataProto(batch=None, non_tensor_batch=None, meta_info={})
        if len(data_proto_list) == 1: return deepcopy(data_proto_list[0])

        combined_batch, combined_ntb = {}, {}
        # Meta info: Start with a deepcopy of the first item's meta_info, or an empty dict.
        base_meta = data_proto_list[0].meta_info if hasattr(data_proto_list[0], 'meta_info') and data_proto_list[0].meta_info is not None else {}
        combined_meta = deepcopy(base_meta)

        # Batch tensors
        # Check if any DataProto object has a .batch attribute that is not None and not empty
        if any(hasattr(dp,'batch') and dp.batch is not None for dp in data_proto_list):
            # Get keys from the first DataProto object that has a non-empty .batch
            first_dp_b = next((dp for dp in data_proto_list if hasattr(dp,'batch') and dp.batch is not None), None)
            if first_dp_b: # Ensure we found one
                for k in first_dp_b.batch.keys():
                    t_cat = [dp.batch[k] for dp in data_proto_list if hasattr(dp,'batch') and dp.batch is not None and k in dp.batch and dp.batch[k] is not None]
                    if t_cat:
                        try: combined_batch[k] = torch.cat(t_cat, dim=0)
                        except Exception as e: print(f"Error concatenating tensor '{k}': {e}") # Log error

        # Non-tensor batch
        if any(hasattr(dp,'non_tensor_batch') and dp.non_tensor_batch is not None for dp in data_proto_list):
            first_dp_ntb = next((dp for dp in data_proto_list if hasattr(dp,'non_tensor_batch') and dp.non_tensor_batch is not None), None)
            if first_dp_ntb: # Ensure we found one
                for k in first_dp_ntb.non_tensor_batch.keys():
                    lst_ext, all_np, dtype_np, first_chk_done = [], True, None, False
                    for dp_idx, dp in enumerate(data_proto_list): # Added dp_idx for debugging if needed
                        if hasattr(dp,'non_tensor_batch') and dp.non_tensor_batch and k in dp.non_tensor_batch:
                            item = dp.non_tensor_batch[k]
                            if not first_chk_done:
                                dtype_np = item.dtype if isinstance(item,np.ndarray) else None
                                all_np = isinstance(item,np.ndarray)
                                first_chk_done=True
                            elif not isinstance(item,np.ndarray): all_np = False # If any item is not np.ndarray

                            if isinstance(item,list): lst_ext.extend(item)
                            elif isinstance(item,np.ndarray): lst_ext.extend(item.tolist()) # Convert to list before extend
                            else: lst_ext.append(item) # Append single items
                    if lst_ext:
                        if all_np and dtype_np is not None:
                            try: combined_ntb[k] = np.array(lst_ext,dtype=dtype_np)
                            except Exception as e_np: combined_ntb[k] = lst_ext; print(f"NP array conversion failed for {k}: {e_np}")
                        else: combined_ntb[k] = lst_ext

        final_td_combined_batch = None
        if combined_batch:
            common_batch_size_comb = None
            first_tensor_val_comb = None
            for tensor_val_comb in combined_batch.values():
                if isinstance(tensor_val_comb, torch.Tensor):
                    first_tensor_val_comb = tensor_val_comb
                    break

            if first_tensor_val_comb is not None:
                common_batch_size_comb = first_tensor_val_comb.shape[:1]
                try:
                    # Ensure all items passed to TensorDict are tensors
                    tensor_source_dict_comb = {k: v for k, v in combined_batch.items() if isinstance(v, torch.Tensor)}
                    if not tensor_source_dict_comb:
                        final_td_combined_batch = None
                    else:
                        final_td_combined_batch = TensorDict(source=tensor_source_dict_comb, batch_size=common_batch_size_comb)
                except Exception as e:
                    print(f"Error creating TensorDict in _combine_data_protos: {e}. Combined_batch keys: {list(combined_batch.keys())}, Common batch size: {common_batch_size_comb}")
                    # Fallback: final_td_combined_batch remains None
            # else: no tensors found, final_td_combined_batch remains None
        # If combined_batch was empty, final_td_combined_batch also remains None

        return DataProto(batch=final_td_combined_batch, # TensorDict or None
                         non_tensor_batch=combined_ntb or None,
                         meta_info=combined_meta)

    def fit(self):
        from verl.utils.tracking import Tracking # Specific logger for VERL
        logger = Tracking(project_name=self.config.trainer.project_name, experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger, config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0; self._load_checkpoint() # Sets self.global_steps if checkpoint loaded

        if self.val_reward_fn and self.config.trainer.get("val_before_train", True):
            val_metrics_initial = self._validate(); pprint(f"Initial validation metrics: {val_metrics_initial}")
            logger.log(data=val_metrics_initial, step=max(0, self.global_steps))
            if self.config.trainer.get("val_only", False): return

        if self.global_steps == 0: self.global_steps = 1 # Start PPO iterations from 1 if not loaded

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress", dynamic_ncols=True)
        last_val_metrics_info = None; self.start_time = time.time() # Renamed last_val_metrics

        while self.global_steps <= self.total_training_steps:
            metrics, timing_raw = {}, {}
            is_last_iteration = (self.global_steps == self.total_training_steps)

            with _timer("step", timing_raw):
                rollout_meta_info_dict = {"eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id,
                                "recompute_log_prob": False, "do_sample": self.config.actor_rollout_ref.rollout.do_sample, "validate": False}
                current_rollout_dp = DataProto(meta_info=deepcopy(rollout_meta_info_dict))

                with _timer("gen", timing_raw):
                    on_policy_dp = self.agent_proxy.rollout(current_rollout_dp, val=False)
                    on_policy_dp, filter_metrics_info = _filter_rollout(on_policy_dp, self.config); metrics.update(filter_metrics_info) # Renamed
                    if hasattr(on_policy_dp,'meta_info') and on_policy_dp.meta_info is not None and "metrics" in on_policy_dp.meta_info:
                        metrics.update({"train/on_policy/" + k: v for k, v in on_policy_dp.meta_info["metrics"].items()})
                    if self.replay_buffer: self.replay_buffer.add(deepcopy(on_policy_dp))

                # Initialize numerical metrics for batch source
                metrics["train/source_is_replay"] = 0.0
                metrics["train/source_is_on_policy_empty_sample_fallback"] = 0.0
                metrics["train/source_is_on_policy_buffer_disabled"] = 0.0
                metrics["train/source_is_on_policy_sampling_disabled"] = 0.0
                metrics["train/source_is_on_policy_buffer_not_ready"] = 0.0
                # metrics["train/replay_buffer_size"] is already numerical and handled below

                if self.replay_buffer and len(self.replay_buffer) >= self.config.replay_buffer.sampling_batch_size and self.config.replay_buffer.sampling_batch_size > 0:
                    sampled_experiences_list = self.replay_buffer.sample(self.config.replay_buffer.sampling_batch_size) # Renamed
                    if sampled_experiences_list:
                        batch = deepcopy(sampled_experiences_list[0]) if self.config.replay_buffer.sampling_batch_size == 1 else self._combine_data_protos(sampled_experiences_list)
                        metrics["train/source_is_replay"] = 1.0
                        # Ensure meta_info for the batch is set up for training
                        if not hasattr(batch,'meta_info') or batch.meta_info is None: batch.meta_info = {}
                        current_training_meta_info_dict = deepcopy(rollout_meta_info_dict) # Base training meta
                        current_training_meta_info_dict.update(batch.meta_info) # Overlay specific meta from (combined) sample
                        batch.meta_info = current_training_meta_info_dict
                    else:
                        batch = deepcopy(on_policy_dp)
                        metrics["train/source_is_on_policy_empty_sample_fallback"] = 1.0
                else:
                    batch = deepcopy(on_policy_dp)
                    if not self.replay_buffer:
                        metrics["train/source_is_on_policy_buffer_disabled"] = 1.0
                    elif self.config.replay_buffer.sampling_batch_size <= 0:
                        metrics["train/source_is_on_policy_sampling_disabled"] = 1.0
                    else:
                        metrics["train/source_is_on_policy_buffer_not_ready"] = 1.0
                        metrics["train/replay_buffer_size"] = len(self.replay_buffer) if self.replay_buffer else 0.0
                # Removed: metrics["train/source"] = batch_source_info

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    from verl.utils.common_utils import get_rich_logger # Import here to avoid top-level if not used
                    # Assuming self.logger or a new logger instance
                    rich_logger_instance = get_rich_logger("REMAX_Warning") # Use a specific name
                    rich_logger_instance.error("[NotImplemented] REMAX. Exiting."); exit()

                num_seqs_in_batch = 0
                if hasattr(batch,'batch') and batch.batch is not None and 'input_ids' in batch.batch and batch.batch['input_ids'] is not None:
                    num_seqs_in_batch = batch.batch['input_ids'].shape[0]

                if not hasattr(batch,'non_tensor_batch') or batch.non_tensor_batch is None: batch.non_tensor_batch = {}
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(num_seqs_in_batch)], dtype=object)

                if hasattr(batch,'batch') and batch.batch is not None : # Ensure batch.batch exists
                    if 'loss_mask' in batch.batch and batch.batch['loss_mask'] is not None: batch.batch["response_mask"] = batch.batch["loss_mask"]
                    else: batch.batch["response_mask"] = compute_response_mask(batch) # compute_response_mask handles internal checks

                if self.config.trainer.balance_batch: self._balance_batch(batch, metrics=metrics)

                if not hasattr(batch,'meta_info') or batch.meta_info is None: batch.meta_info = {}
                if hasattr(batch,'batch') and batch.batch is not None and 'attention_mask' in batch.batch and batch.batch['attention_mask'] is not None:
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                else: batch.meta_info["global_token_num"] = []

                if self.use_rm:
                    with _timer("reward_model",timing_raw): batch=batch.union(self.rm_wg.compute_rm_score(batch))

                token_level_scores_from_reward, reward_extra_dict = None, {}
                future_reward_object = None # Initialized
                if self.config.reward_model.launch_reward_fn_async: future_reward_object = compute_reward_async.remote(batch,self.config,self.tokenizer)
                else: token_level_scores_from_reward, reward_extra_dict = compute_reward(batch,self.reward_fn)

                with _timer("old_log_prob",timing_raw):
                    batch=batch.union(self.actor_rollout_wg.compute_log_prob(batch))
                    # Ensure keys exist before masked_mean
                    if hasattr(batch,'batch') and batch.batch is not None and 'old_log_probs' in batch.batch and 'response_mask' in batch.batch and \
                       batch.batch['old_log_probs'] is not None and batch.batch['response_mask'] is not None:
                        metrics["rollout/old_log_prob"]=masked_mean(batch.batch["old_log_probs"],batch.batch["response_mask"]).item()

                if self.use_reference_policy:
                    with _timer("ref",timing_raw):
                        ref_dp = self.actor_rollout_wg.compute_ref_log_prob(batch) if self.ref_in_actor else self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch=batch.union(ref_dp)
                    if hasattr(batch,'batch') and batch.batch is not None and 'ref_log_prob' in batch.batch and 'response_mask' in batch.batch and \
                       batch.batch['ref_log_prob'] is not None and batch.batch['response_mask'] is not None:
                        metrics["rollout/ref_log_prob"]=masked_mean(batch.batch["ref_log_prob"],batch.batch["response_mask"]).item()

                if self.use_critic:
                    with _timer("values",timing_raw): batch=batch.union(self.critic_wg.compute_values(batch))

                with _timer("adv_reward_finalize",timing_raw):
                    if future_reward_object: token_level_scores_from_reward, reward_extra_dict = ray.get(future_reward_object)

                    if not hasattr(batch,'batch') or batch.batch is None: batch.batch = {} # Ensure batch.batch exists
                    if token_level_scores_from_reward is not None: batch.batch["token_level_scores"] = token_level_scores_from_reward

                    if not hasattr(batch,'non_tensor_batch') or batch.non_tensor_batch is None: batch.non_tensor_batch = {}
                    if reward_extra_dict: batch.non_tensor_batch.update({k:(np.array(v) if isinstance(v,list) else v) for k,v in reward_extra_dict.items()})

                    if self.config.algorithm.use_kl_in_reward:
                        batch,kl_metrics_info=apply_kl_penalty(batch,self.kl_ctrl_in_reward,self.config.algorithm.kl_penalty,True);metrics.update(kl_metrics_info)
                    if hasattr(batch,'batch') and batch.batch is not None and 'token_level_scores' in batch.batch and batch.batch['token_level_scores'] is not None :
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                    # else: token_level_rewards might be missing if scores are not there and KL not used.

                with _timer("adv",timing_raw):
                    batch=compute_advantage(batch,self.config.algorithm.adv_estimator,self.config.algorithm.gamma,self.config.algorithm.lam,
                                            self.config.actor_rollout_ref.rollout.n, True, # multi_turn=True
                                            self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                                            self.config.algorithm.bi_level_gae, self.config.algorithm.high_level_gamma)

                if self.config.algorithm.adv_estimator==AdvantageEstimator.GRPO and self.config.grpo_advantage_length_weight:
                    # Ensure keys exist before use
                    if hasattr(batch, 'batch') and batch.batch is not None and \
                       'response_mask' in batch.batch and batch.batch['response_mask'] is not None and \
                       'advantages' in batch.batch and batch.batch['advantages'] is not None:
                        rmask,advs=(batch.batch["response_mask"],batch.batch["advantages"])
                        mean_len = torch.sum(rmask,dim=-1).float().mean()
                        rlens=(torch.sum(rmask,dim=-1)+1e-6)/(mean_len if mean_len > 0 else 1.0 + 1e-6) # Avoid div by zero if mean_len is 0
                        batch.batch["advantages"]=advs/rlens.unsqueeze(-1)

                if self.use_critic:
                    with _timer("update_critic",timing_raw): critic_output_dp=self.critic_wg.update_critic(batch)
                    if hasattr(critic_output_dp,'meta_info') and critic_output_dp.meta_info is not None and "metrics" in critic_output_dp.meta_info:
                        metrics.update(reduce_metrics(critic_output_dp.meta_info["metrics"]))

                if self.config.trainer.critic_warmup <= self.global_steps:
                    with _timer("update_actor",timing_raw):
                        if not hasattr(batch,'meta_info') or batch.meta_info is None: batch.meta_info={}
                        batch.meta_info["multi_turn"]=True; actor_output_dp=self.actor_rollout_wg.update_actor(batch)
                    if hasattr(actor_output_dp,'meta_info') and actor_output_dp.meta_info is not None and "metrics" in actor_output_dp.meta_info:
                        metrics.update(reduce_metrics(actor_output_dp.meta_info["metrics"]))

                rollout_data_dir_path = self.config.trainer.get("rollout_data_dir") # Renamed
                if rollout_data_dir_path and hasattr(batch,'batch') and batch.batch is not None:
                    with _timer("dump_rollout_gens",timing_raw):
                        current_ntb = batch.non_tensor_batch if hasattr(batch,'non_tensor_batch') and batch.non_tensor_batch is not None else {}
                        # Ensure .get returns empty list if key missing, not None, for batch_decode
                        prompts_list = batch.batch.get("prompts", torch.empty(0))
                        responses_list = batch.batch.get("responses", torch.empty(0))
                        tls_list = batch.batch.get("token_level_scores", torch.empty(0))

                        self._dump_generations(inputs=self.tokenizer.batch_decode(prompts_list,skip_special_tokens=True) if prompts_list.numel() > 0 else [],
                                               outputs=self.tokenizer.batch_decode(responses_list,skip_special_tokens=True) if responses_list.numel() > 0 else [],
                                               scores=tls_list.sum(-1).cpu().tolist() if tls_list.numel() > 0 else [],
                                               reward_extra_infos_dict={k:v for k,v in current_ntb.items() if k!="uid"},
                                               dump_path=rollout_data_dir_path)

                if self.val_reward_fn and self.config.trainer.test_freq > 0 and (is_last_iteration or self.global_steps % self.config.trainer.test_freq==0):
                    with _timer("testing",timing_raw):
                        val_metrics_current_step=self._validate(); metrics.update(val_metrics_current_step) # Renamed
                    if is_last_iteration: last_val_metrics_info = val_metrics_current_step

                if self.config.trainer.save_freq > 0 and (is_last_iteration or self.global_steps % self.config.trainer.save_freq==0):
                    with _timer("save_ckpt",timing_raw): self._save_checkpoint()
            # End _timer("step", ...)

            metrics.update(compute_data_metrics(batch=batch,use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch,timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch,timing_raw=timing_raw,n_gpus=self.resource_pool_manager.get_n_gpus()))
            metrics.update({"timing_s/fit_total_time": time.time() - self.start_time})
            logger.log(data=metrics, step=self.global_steps)

            if is_last_iteration:
                if last_val_metrics_info: pprint(f"Final validation @ step {self.global_steps}: {last_val_metrics_info}")
                else: pprint(f"Training finished @ step {self.global_steps}.")
                if progress_bar: progress_bar.close(); progress_bar = None
                break

            if progress_bar: progress_bar.update(1)
            self.global_steps += 1

        if progress_bar and not progress_bar.disable: progress_bar.close()
