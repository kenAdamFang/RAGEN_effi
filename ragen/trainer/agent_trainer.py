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

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from ragen.trainer import core_algos
from ragen.trainer.core_algos import agg_loss # agg_loss may not be used, check original
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
from verl.utils.tracking import ValidationGenerationsLogger # This is from VERL, not RAGEN's GenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]


from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager, compute_response_mask, _timer, apply_kl_penalty, AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as VerlRayPPOTrainer

# import torch # Redundant import
# from verl.utils.torch_functional import masked_mean # Redundant import

from ragen.llm_agent.agent_proxy import LLMAgentProxy
from ragen.utils import GenerationsLogger # This is RAGEN's logger

from ragen.trainer.replay_buffer import ReplayBuffer # Added for Replay Buffer
import time # Added for self.start_time
# import uuid # Already available via global import
# import numpy as np # Already available via global import


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, bi_level_gae=False, high_level_gamma=1.0): # Unchanged from original
    if "response_mask" not in data.batch: # Make sure data.batch exists
        if not hasattr(data, 'batch') or data.batch is None: data.batch = {}
        data.batch["response_mask"] = compute_response_mask(data) 
    
    # Ensure all required keys exist in data.batch before using them
    required_keys = ["token_level_rewards", "values", "response_mask", "loss_mask"]
    for key in required_keys:
        if key not in data.batch:
            # print(f"Warning: Key '{key}' not found in data.batch for advantage computation. Setting to zeros or appropriate default.")
            # Set to a default value, e.g., zeros tensor of appropriate shape if possible, or handle error
            # This depends on downstream expectations. For now, assume it might lead to errors if not present.
            # A simple placeholder: data.batch[key] = torch.zeros_like(data.batch["response_mask"]) if "response_mask" in data.batch and data.batch["response_mask"] is not None else torch.empty(0)
            pass # Let it fail if keys are missing, to highlight issues

    if adv_estimator == AdvantageEstimator.GAE:
        if bi_level_gae:
            advantages, returns = core_algos.compute_bi_level_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"], values=data.batch["values"],
                loss_mask=data.batch["response_mask"], gamma=gamma, lam=lam, high_level_gamma=high_level_gamma,
            )
        else:
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"], values=data.batch["values"],
                response_mask=data.batch["response_mask"], gamma=gamma, lam=lam,
            )
        data.batch["advantages"] = advantages; data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1); grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"], response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"], norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages; data.batch["returns"] = returns
    # ... (other estimators remain the same) ...
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"], response_mask=data.batch["response_mask"], index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages; data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"], response_mask=data.batch["response_mask"], gamma=gamma,
        )
        data.batch["advantages"] = advantages; data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX: 
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"], reward_baselines=data.batch["reward_baselines"], response_mask=data.batch["response_mask"],
        )
        data.batch["advantages"] = advantages; data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"], response_mask=data.batch["response_mask"], index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages; data.batch["returns"] = returns
    else: raise NotImplementedError
    return data

def _filter_rollout(batch, config_obj): # Modified to accept config_obj
    rollout_filter_ratio = config_obj.actor_rollout_ref.rollout.rollout_filter_ratio
    num_groups = config_obj.es_manager.train.env_groups
    group_size = config_obj.es_manager.train.group_size
    dummy_metrics = {"rollout/in_group_std": 0.0, "rollout/in_group_max": 0.0, "rollout/in_group_mean": 0.0, 
                     "rollout/chosen_in_group_std": 0.0, "rollout/chosen_in_group_max": 0.0, "rollout/chosen_in_group_mean": 0.0}

    if not hasattr(batch, 'batch') or not batch.batch or 'original_rm_scores' not in batch.batch or batch.batch['original_rm_scores'] is None:
        return batch, dummy_metrics
    
    expected_elements = num_groups * group_size
    current_elements = batch.batch["original_rm_scores"].shape[0] if batch.batch["original_rm_scores"] is not None else 0

    if batch.batch["original_rm_scores"].numel() == 0 or current_elements != expected_elements:
        # print(f"Warning: 'original_rm_scores' shape mismatch or empty. Expected {expected_elements}, got {current_elements}. Skipping filtering.")
        return batch, dummy_metrics

    rm_scores = batch.batch["original_rm_scores"].sum(dim=-1).view(num_groups, group_size)
    in_group_std = rm_scores.std(dim=-1); in_group_max = rm_scores.max(dim=-1).values; in_group_mean = rm_scores.mean(dim=-1)
    
    metrics = {"rollout/in_group_std": in_group_std.mean().item(), "rollout/in_group_max": in_group_max.mean().item(), "rollout/in_group_mean": in_group_mean.mean().item()}

    if rollout_filter_ratio == 1.0: 
        metrics.update({"rollout/chosen_in_group_std": metrics["rollout/in_group_std"], 
                        "rollout/chosen_in_group_max": metrics["rollout/in_group_max"], 
                        "rollout/chosen_in_group_mean": metrics["rollout/in_group_mean"]})
        return batch, metrics

    k_topk = max(1, int(rollout_filter_ratio * num_groups))
    if k_topk > num_groups : k_topk = num_groups # Cannot request more groups than available

    if config_obj.actor_rollout_ref.rollout.rollout_filter_type == "std_rev":
        top_groups = (-in_group_std).topk(k_topk).indices 
    elif config_obj.actor_rollout_ref.rollout.rollout_filter_type == "std":
        top_groups = in_group_std.topk(k_topk).indices
    else: raise ValueError(f"Invalid rollout filter type: {config_obj.actor_rollout_ref.rollout.rollout_filter_type}")

    mask = torch.zeros(num_groups, dtype=torch.bool, device=rm_scores.device); mask[top_groups] = True
    mask = mask.unsqueeze(1).expand(-1, group_size).flatten()

    if hasattr(batch, 'batch') and batch.batch is not None:
        for key_b, value_b in batch.batch.items():
            if isinstance(value_b, torch.Tensor) and value_b.shape[0] == expected_elements: batch.batch[key_b] = value_b[mask]
    if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch is not None:
        for key_ntb, value_ntb in batch.non_tensor_batch.items():
            try:
                if isinstance(value_ntb, np.ndarray) and value_ntb.shape[0] == expected_elements: batch.non_tensor_batch[key_ntb] = value_ntb[mask.cpu().numpy()]
                elif isinstance(value_ntb, list) and len(value_ntb) == expected_elements: batch.non_tensor_batch[key_ntb] = [v for v, m in zip(value_ntb, mask.tolist()) if m]
            except Exception as e: print(f"Error filtering non_tensor_batch key {key_ntb} in _filter_rollout: {e}")
    
    metrics.update({"rollout/chosen_in_group_std": in_group_std[top_groups].mean().item(), 
                    "rollout/chosen_in_group_max": in_group_max[top_groups].mean().item(), 
                    "rollout/chosen_in_group_mean": in_group_mean[top_groups].mean().item()})
    return batch, metrics

class RayAgentTrainer(VerlRayPPOTrainer):
    def __init__(self, config, tokenizer, role_worker_mapping: dict[Role, WorkerType], resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup, processor=None, reward_fn=None, val_reward_fn=None):
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn)
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0
        self.generations_logger = GenerationsLogger() 
        if self.config.get('replay_buffer') and self.config.replay_buffer.enable:
            self.replay_buffer = ReplayBuffer(capacity=self.config.replay_buffer.capacity) # sampling_batch_size is taken from config in .sample()
            print(f"Replay buffer enabled: capacity={self.config.replay_buffer.capacity}, sampling_batch_size={self.config.replay_buffer.sampling_batch_size}")
        else:
            self.replay_buffer = None; print("Replay buffer disabled.")
        
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler): # Unchanged
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

    def _maybe_log_generations(self, inputs, outputs, scores, _type="val"): # Unchanged
        generations_to_log = self.config.trainer.generations_to_log_to_wandb[_type]
        if generations_to_log == 0: return
        samples = list(zip(inputs, outputs, scores)); samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42); rng.shuffle(samples)
        samples = samples[:generations_to_log]
        self.generations_logger.log(self.config.trainer.logger, samples, self.global_steps, _type)

    def _validate(self): # Mostly unchanged, minor fixes for robustness
        data_source_lst, reward_extra_infos_dict, sample_inputs, sample_outputs, sample_scores, env_metric_dict = [], defaultdict(list), [], [], [], {}
        for _ in range(self.config.trainer.validation_steps):
            sample_inputs.extend([""] * (self.config.es_manager.val.env_groups * self.config.es_manager.val.group_size))
            meta_info = {"eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id, "recompute_log_prob": False, 
                         "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample, "validate": True}
            test_gen_batch = DataProto(batch=None, non_tensor_batch=None, meta_info=meta_info)
            val_st = time.time(); test_batch = self.agent_proxy.rollout(test_gen_batch, val=True); print(f"Val gen time: {time.time() - val_st}s")
            if hasattr(test_batch, 'meta_info') and test_batch.meta_info and "metrics" in test_batch.meta_info:
                for k, v in test_batch.meta_info["metrics"].items(): env_metric_dict.setdefault("val-env/" + k, []).append(v)
            if hasattr(test_batch, 'batch') and test_batch.batch and 'responses' in test_batch.batch and test_batch.batch['responses'] is not None:
                sample_outputs.extend([self.tokenizer.decode(ids, skip_special_tokens=True) for ids in test_batch.batch["responses"]])
            result = self.val_reward_fn(test_batch, return_dict=True); scores_val = result["reward_tensor"].sum(-1).cpu().tolist(); sample_scores.extend(scores_val)
            reward_extra_infos_dict["reward"].extend(scores_val)
            if "reward_extra_info" in result:
                for k, lst in result["reward_extra_info"].items(): reward_extra_infos_dict[k].extend(lst)
            if hasattr(test_batch, 'non_tensor_batch') and test_batch.non_tensor_batch:
                 data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(scores_val))) # Use len(scores_val) for shape
        self._maybe_log_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, _type="val")
        if self.config.trainer.get("validation_data_dir"): self._dump_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, reward_extra_infos_dict=reward_extra_infos_dict, dump_path=self.config.trainer.validation_data_dir)
        for k_info, lst_info in reward_extra_infos_dict.items(): assert not lst_info or len(lst_info) == len(sample_scores), f"{k_info}: {len(lst_info)=}, {len(sample_scores)=}"
        ds_val = np.concatenate(data_source_lst) if data_source_lst else np.array([]); data_src2var2metric2val = process_validation_metrics(ds_val, sample_inputs, reward_extra_infos_dict)
        metric_dict_val = reduce_metrics(env_metric_dict)
        for ds, v2m2v in data_src2var2metric2val.items(): # ds=data_source, v2m2v=var2metric2val
            core_var = "acc" if "acc" in v2m2v else "reward"
            for var_n, m2v in v2m2v.items():
                n_max = 1; 
                if m2v: n_max_k = [n.split("@")[-1].split("/")[0] for n in m2v.keys() if "@" in n]; n_max = max(int(k) for k in n_max_k if k.isdigit()) if any(k.isdigit() for k in n_max_k) else 1
                for met_n, met_v in m2v.items():
                    met_s = "val-core" if (var_n == core_var and any(met_n.startswith(pfx) for pfx in ["mean","maj","best"]) and (f"@{n_max}"in met_n)) else "val-aux"
                    metric_dict_val[f"{met_s}/{ds}/{var_n}/{met_n}"] = met_v
        return metric_dict_val

    def init_workers(self): # Unchanged
        self.resource_pool_manager.create_resource_pool(); self.resource_pool_to_cls = {p: {} for p in self.resource_pool_manager.resource_pool_dict.values()}
        if self.hybrid_engine: rp_ar = self.resource_pool_manager.get_resource_pool(Role.ActorRollout); self.resource_pool_to_cls[rp_ar]["actor_rollout"] = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.actor_rollout_ref, role="actor_rollout")
        else: raise NotImplementedError
        if self.use_critic: rp_c = self.resource_pool_manager.get_resource_pool(Role.Critic); self.resource_pool_to_cls[rp_c]["critic"] = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
        if self.use_reference_policy and not self.ref_in_actor: rp_ref = self.resource_pool_manager.get_resource_pool(Role.RefPolicy); self.resource_pool_to_cls[rp_ref]["ref"] = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
        if self.use_rm: rp_rm = self.resource_pool_manager.get_resource_pool(Role.RewardModel); self.resource_pool_to_cls[rp_rm]["rm"] = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
        all_wg = {}; self.wg_dicts = []; wg_kwargs = {}; timeout = OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout"); 
        if timeout is not None: wg_kwargs["ray_wait_register_center_timeout"] = timeout
        for rp, cd in self.resource_pool_to_cls.items(): wdc = create_colocated_worker_cls(class_dict=cd); wg_d = self.ray_worker_group_cls(resource_pool=rp, ray_cls_with_init=wdc, **wg_kwargs); spawn_wg = wg_d.spawn(prefix_set=cd.keys()); all_wg.update(spawn_wg); self.wg_dicts.append(wg_d)
        if self.use_critic: self.critic_wg = all_wg["critic"]; self.critic_wg.init_model()
        if self.use_reference_policy and not self.ref_in_actor: self.ref_policy_wg = all_wg["ref"]; self.ref_policy_wg.init_model()
        if self.use_rm: self.rm_wg = all_wg["rm"]; self.rm_wg.init_model()
        self.actor_rollout_wg = all_wg["actor_rollout"]; self.actor_rollout_wg.init_model()
        self.async_rollout_mode = (self.config.actor_rollout_ref.rollout.mode == "async")
        if self.async_rollout_mode: self.async_rollout_manager = AsyncLLMServerManager(config=self.config.actor_rollout_ref, worker_group=self.actor_rollout_wg)

    def _save_checkpoint(self): # Unchanged
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

    def _combine_data_protos(self, data_proto_list: list[DataProto]) -> DataProto: # New Method
        if not data_proto_list: return DataProto(batch=None, non_tensor_batch=None, meta_info={})
        if len(data_proto_list) == 1: return deepcopy(data_proto_list[0])
        combined_batch, combined_ntb, base_meta = {}, {}, {}
        if data_proto_list[0] and hasattr(data_proto_list[0], 'meta_info') and data_proto_list[0].meta_info: base_meta = data_proto_list[0].meta_info
        combined_meta = deepcopy(base_meta)
        # Batch tensors
        if any(hasattr(dp,'batch') and dp.batch for dp in data_proto_list):
            first_b = next((dp for dp in data_proto_list if hasattr(dp,'batch') and dp.batch), None)
            if first_b:
                for k in first_b.batch.keys():
                    t_cat = [dp.batch[k] for dp in data_proto_list if hasattr(dp,'batch') and dp.batch and k in dp.batch and dp.batch[k] is not None]
                    if t_cat: try: combined_batch[k] = torch.cat(t_cat, dim=0) 
                              except Exception as e: print(f"Err cat tensor {k}: {e}")
        # Non-tensor batch
        if any(hasattr(dp,'non_tensor_batch') and dp.non_tensor_batch for dp in data_proto_list):
            first_ntb = next((dp for dp in data_proto_list if hasattr(dp,'non_tensor_batch') and dp.non_tensor_batch), None)
            if first_ntb:
                for k in first_ntb.non_tensor_batch.keys():
                    lst_ext, all_np, dtype_np, first_chk = [], True, None, False
                    for dp in data_proto_list:
                        if hasattr(dp,'non_tensor_batch') and dp.non_tensor_batch and k in dp.non_tensor_batch:
                            item = dp.non_tensor_batch[k]
                            if not first_chk: dtype_np = item.dtype if isinstance(item,np.ndarray) else None; all_np = isinstance(item,np.ndarray); first_chk=True
                            elif not isinstance(item,np.ndarray): all_np = False
                            if isinstance(item,list): lst_ext.extend(item)
                            elif isinstance(item,np.ndarray): lst_ext.extend(item.tolist())
                            else: lst_ext.append(item)
                    if lst_ext: combined_ntb[k] = np.array(lst_ext,dtype=dtype_np) if all_np and dtype_np else lst_ext
        return DataProto(batch=combined_batch or None, non_tensor_batch=combined_ntb or None, meta_info=combined_meta)

    def fit(self): # Heavily Modified
        from verl.utils.tracking import Tracking 
        logger = Tracking(project_name=self.config.trainer.project_name, experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger, config=OmegaConf.to_container(self.config, resolve=True))
        self.global_steps = 0; self._load_checkpoint()
        if self.val_reward_fn and self.config.trainer.get("val_before_train", True):
            val_mets = self._validate(); pprint(f"Initial validation metrics: {val_mets}")
            logger.log(data=val_mets, step=max(0, self.global_steps)) # Log at 0 if fresh start
            if self.config.trainer.get("val_only", False): return
        if self.global_steps == 0: self.global_steps = 1 
        
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress", dynamic_ncols=True)
        last_val_mets = None; self.start_time = time.time()

        while self.global_steps <= self.total_training_steps:
            metrics, timing_raw, is_last = {}, {}, (self.global_steps == self.total_training_steps)
            with _timer("step", timing_raw):
                rollout_meta = {"eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id, 
                                "recompute_log_prob": False, "do_sample": self.config.actor_rollout_ref.rollout.do_sample, "validate": False}
                current_rollout_dp = DataProto(meta_info=deepcopy(rollout_meta))
                with _timer("gen", timing_raw):
                    on_policy_dp = self.agent_proxy.rollout(current_rollout_dp, val=False)
                    on_policy_dp, filter_mets = _filter_rollout(on_policy_dp, self.config); metrics.update(filter_mets)
                    if hasattr(on_policy_dp,'meta_info') and on_policy_dp.meta_info and "metrics" in on_policy_dp.meta_info: 
                        metrics.update({"train/on_policy/" + k: v for k, v in on_policy_dp.meta_info["metrics"].items()})
                    if self.replay_buffer: self.replay_buffer.add(deepcopy(on_policy_dp))

                batch_source_metric = "on_policy_unknown" # Default
                if self.replay_buffer and len(self.replay_buffer) >= self.config.replay_buffer.sampling_batch_size and self.config.replay_buffer.sampling_batch_size > 0:
                    sampled_exp = self.replay_buffer.sample(self.config.replay_buffer.sampling_batch_size)
                    if sampled_exp:
                        batch = deepcopy(sampled_exp[0]) if self.config.replay_buffer.sampling_batch_size == 1 else self._combine_data_protos(sampled_exp)
                        batch_source_metric = "replay_buffer"
                        if not hasattr(batch,'meta_info') or not batch.meta_info: batch.meta_info = {}
                        current_train_meta = deepcopy(rollout_meta); current_train_meta.update(batch.meta_info); batch.meta_info = current_train_meta
                    else: batch = deepcopy(on_policy_dp); batch_source_metric = "on_policy_empty_sample_fallback"
                else:
                    batch = deepcopy(on_policy_dp)
                    if not self.replay_buffer: batch_source_metric = "on_policy_buffer_disabled"
                    elif self.config.replay_buffer.sampling_batch_size <= 0: batch_source_metric = "on_policy_sampling_disabled"
                    else: batch_source_metric = "on_policy_buffer_not_ready"; metrics["train/replay_buffer_size"] = len(self.replay_buffer) if self.replay_buffer else 0
                metrics["train/source"] = batch_source_metric
                
                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX: # Original RAGEN had exit()
                    from verl.utils.common_utils import get_rich_logger; get_rich_logger(__name__).error("[NotImplemented] REMAX. Exiting."); exit()

                num_seqs_in_batch = 0
                if hasattr(batch,'batch') and batch.batch and 'input_ids' in batch.batch and batch.batch['input_ids'] is not None:
                    num_seqs_in_batch = batch.batch['input_ids'].shape[0]
                if not hasattr(batch,'non_tensor_batch') or not batch.non_tensor_batch: batch.non_tensor_batch = {} # Ensure exists
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(num_seqs_in_batch)], dtype=object)

                if hasattr(batch,'batch') and batch.batch and 'loss_mask' in batch.batch and batch.batch['loss_mask'] is not None: batch.batch["response_mask"] = batch.batch["loss_mask"]
                elif hasattr(batch,'batch') and batch.batch: batch.batch["response_mask"] = compute_response_mask(batch)
                
                if self.config.trainer.balance_batch: self._balance_batch(batch, metrics=metrics)
                
                if not hasattr(batch,'meta_info') or not batch.meta_info: batch.meta_info = {} # Ensure exists
                if hasattr(batch,'batch') and batch.batch and 'attention_mask' in batch.batch and batch.batch['attention_mask'] is not None:
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                else: batch.meta_info["global_token_num"] = []

                if self.use_rm:
                    with _timer("reward_model",timing_raw): batch=batch.union(self.rm_wg.compute_rm_score(batch))
                
                tls_tensor, reid_step = None, {} # token_level_scores, reward_extra_info_dict
                if self.config.reward_model.launch_reward_fn_async: future_rew = compute_reward_async.remote(batch,self.config,self.tokenizer)
                else: tls_tensor, reid_step = compute_reward(batch,self.reward_fn)
                
                with _timer("old_log_prob",timing_raw): batch=batch.union(self.actor_rollout_wg.compute_log_prob(batch))
                if 'old_log_probs' in batch.batch and 'response_mask' in batch.batch: metrics["rollout/old_log_prob"]=masked_mean(batch.batch["old_log_probs"],batch.batch["response_mask"]).item()
                if self.use_reference_policy:
                    with _timer("ref",timing_raw): batch=batch.union(self.actor_rollout_wg.compute_ref_log_prob(batch) if self.ref_in_actor else self.ref_policy_wg.compute_ref_log_prob(batch))
                    if 'ref_log_prob' in batch.batch and 'response_mask' in batch.batch: metrics["rollout/ref_log_prob"]=masked_mean(batch.batch["ref_log_prob"],batch.batch["response_mask"]).item()
                if self.use_critic:
                    with _timer("values",timing_raw): batch=batch.union(self.critic_wg.compute_values(batch))
                
                with _timer("adv_reward_finalize",timing_raw):
                    if 'future_rew' in locals() and future_rew: tls_tensor, reid_step = ray.get(future_rew)
                    if tls_tensor is not None: batch.batch["token_level_scores"] = tls_tensor
                    if reid_step: batch.non_tensor_batch.update({k:(np.array(v) if isinstance(v,list) else v) for k,v in reid_step.items()})
                    if self.config.algorithm.use_kl_in_reward: batch,kl_mets=apply_kl_penalty(batch,self.kl_ctrl_in_reward,self.config.algorithm.kl_penalty,True);metrics.update(kl_mets)
                    elif 'token_level_scores' in batch.batch: batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                
                with _timer("adv",timing_raw): batch=compute_advantage(batch,self.config.algorithm.adv_estimator,self.config.algorithm.gamma,self.config.algorithm.lam,self.config.actor_rollout_ref.rollout.n,self.config.algorithm.get("norm_adv_by_std_in_grpo",True),True,self.config.algorithm.high_level_gamma,self.config.algorithm.bi_level_gae) # multi_turn=True
                if self.config.algorithm.adv_estimator==AdvantageEstimator.GRPO and self.config.grpo_advantage_length_weight:
                    rmask,advs=(batch.batch["response_mask"],batch.batch["advantages"]); rlens=(torch.sum(rmask,dim=-1)+1e-6)/(torch.sum(rmask,dim=-1).float().mean()+1e-6); batch.batch["advantages"]=advs/rlens.unsqueeze(-1)
                if self.use_critic:
                    with _timer("update_critic",timing_raw): crit_out_dp=self.critic_wg.update_critic(batch)
                    if hasattr(crit_out_dp,'meta_info') and crit_out_dp.meta_info and "metrics" in crit_out_dp.meta_info: metrics.update(reduce_metrics(crit_out_dp.meta_info["metrics"]))
                if self.config.trainer.critic_warmup <= self.global_steps:
                    with _timer("update_actor",timing_raw):
                        if not hasattr(batch,'meta_info') or not batch.meta_info: batch.meta_info={}
                        batch.meta_info["multi_turn"]=True; act_out_dp=self.actor_rollout_wg.update_actor(batch)
                    if hasattr(act_out_dp,'meta_info') and act_out_dp.meta_info and "metrics" in act_out_dp.meta_info: metrics.update(reduce_metrics(act_out_dp.meta_info["metrics"]))
                
                if self.config.trainer.get("rollout_data_dir") and hasattr(batch,'batch') and batch.batch:
                    with _timer("dump_rollout_gens",timing_raw):
                        ntb = batch.non_tensor_batch if hasattr(batch,'non_tensor_batch') and batch.non_tensor_batch else {}
                        self._dump_generations(inputs=self.tokenizer.batch_decode(batch.batch.get("prompts",[]),skip_special_tokens=True),
                                               outputs=self.tokenizer.batch_decode(batch.batch.get("responses",[]),skip_special_tokens=True),
                                               scores=batch.batch.get("token_level_scores",torch.empty(0)).sum(-1).cpu().tolist(),
                                               reward_extra_infos_dict={k:v for k,v in ntb.items() if k!="uid"}, 
                                               dump_path=self.config.trainer.rollout_data_dir)
                if self.val_reward_fn and self.config.trainer.test_freq > 0 and (is_last or self.global_steps % self.config.trainer.test_freq==0):
                    with _timer("testing",timing_raw): val_mets_s=self._validate(); metrics.update(val_mets_s)
                    if is_last: last_val_mets = val_mets_s 
                if self.config.trainer.save_freq > 0 and (is_last or self.global_steps % self.config.trainer.save_freq==0):
                    with _timer("save_ckpt",timing_raw): self._save_checkpoint()
            
            metrics.update(compute_data_metrics(batch=batch,use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch,timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch,timing_raw=timing_raw,n_gpus=self.resource_pool_manager.get_n_gpus()))
            metrics.update({"timing_s/fit_total_time": time.time() - self.start_time}) # Total time for fit()
            logger.log(data=metrics, step=self.global_steps)
            if is_last:
                if last_val_mets: pprint(f"Final validation @ step {self.global_steps}: {last_val_mets}")
                else: pprint(f"Training finished @ step {self.global_steps}.")
                if progress_bar: progress_bar.close(); progress_bar = None # Avoid double close
                break 
            if progress_bar: progress_bar.update(1)
            self.global_steps += 1
        if progress_bar and not progress_bar.disable: progress_bar.close()

