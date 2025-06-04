<h1 align="center"> RAGEN: Training Agents by Reinforcing Reasoning </h1>


<p align="center"><img src="public/ragen_logo.jpeg" width="300px" alt="RAGEN icon" /></p>



<p align="center" style="font-size: 18px;">
  <strong>RAGEN</strong> (<b>R</b>easoning <b>AGEN</b>t, pronounced like "region") leverages reinforcement learning (RL) to train <br>
  <strong>LLM reasoning agents</strong> in interactive, stochastic environments.<br>
  <em>We strongly believe in the future of RL + LLM + Agents. The release is a minimally viable leap forward.</em>
</p>


<p align="center">
  <a href="https://ragen-ai.github.io/"><img src="https://img.shields.io/badge/📝_HomePage-FF5722?style=for-the-badge&logoColor=white" alt="Blog"></a>
  <a href="https://arxiv.org/abs/2504.20073"><img src="https://img.shields.io/badge/📄_Paper-EA4335?style=for-the-badge&logoColor=white" alt="Paper"></a>
  <a href="https://ragen-doc.readthedocs.io/"><img src="https://img.shields.io/badge/📚_Documentation-4285F4?style=for-the-badge&logoColor=white" alt="Documentation"></a>
  <a href="https://x.com/wzihanw/status/1915052871474712858"><img src="https://img.shields.io/badge/🔍_Post-34A853?style=for-the-badge&logoColor=white" alt="Post"></a>
  <a href="https://api.wandb.ai/links/zihanwang-ai-northwestern-university/a8er8l7b"><img src="https://img.shields.io/badge/🧪_Experiment_Log-AB47BC?style=for-the-badge&logoColor=white" alt="Experiment Log"></a>

</p>

**2025.5.8 Update:**
We now release the official [Documentation](https://ragen-doc.readthedocs.io/) for RAGEN. The documentation will be continuously updated and improved to provide a comprehensive and up-to-date guidance.

**2025.5.2 Update:**
We now release a [tracking document](https://docs.google.com/document/d/1bg7obeiKTExuHHBl5uOiSpec5uLDZ2Tgvxy6li5pHX4/edit?usp=sharing) to log minor updates in the RAGEN codebase. 


**2025.4.20 Update:**

Our RAGEN [paper](https://arxiv.org/abs/2504.20073) is out!

We've further streamlined the RAGEN codebase (v0423) to improve development.
1. Architecture: Restructured veRL as a submodule for better co-development
2. Modularity: Divided RAGEN into three components—Environment Manager, Context Manager, and Agent Proxy, making it significantly simpler to add new environments (details below), track environmental dynamics, and run multiple experiments


**2025.4.16 Update:**

We recently noticed that a [third-party website](https://ragen-ai.com) has been created using our project's name and content. While we appreciate the interest in the project, we'd like to clarify that this GitHub repository is the official and primary source for all code, updates, and documentation.
If we launch an official website in the future, it will be explicitly linked here.

Thank you for your support and understanding!


**2025.3.13 Update:**


We are recently refactoring RAGEN code to help you better develop your own idea on the codebase. Please checkout our [developing branch](https://github.com/ZihanWang314/RAGEN/tree/main-new). The first version decomposes RAGEN and veRL for better co-development, taking the latter as a submodule rather than a static directory.

**2025.3.8 Update:**

1. In previous veRL implementation, there is a [KL term issue](https://github.com/volcengine/verl/pull/179/files), which has been fixed in recent versions.
2. We find evidence from multiple sources that PPO could be more stable than GRPO training in [Open-Reasoner-Zero](https://x.com/rosstaylor90/status/1892664646890312125), [TinyZero](https://github.com/Jiayi-Pan/TinyZero), and [Zhihu](https://www.zhihu.com/search?type=content&q=%E6%97%A0%E5%81%8FGRPO). We have changed the default advantage estimator to GAE (using PPO) and aim to find more stable while efficient RL optimization methods in later versions.

**2025.1.27:**

We are thrilled to release RAGEN! Check out our post [here](https://x.com/wzihanw/status/1884092805598826609).


## Overview

<!--
Reinforcement Learning (RL) with rule-based rewards has shown promise in enhancing reasoning capabilities of large language models (LLMs). However, existing approaches have primarily focused on static, single-turn tasks like math reasoning and coding. Extending these methods to agent scenarios introduces two fundamental challenges:

1. **Multi-turn Interactions**: Agents must perform sequential decision-making and react to environment feedback
2. **Stochastic Environments**: Uncertainty where identical actions can lead to different outcomes

RAGEN addresses these challenges through:
- A Markov Decision Process (MDP) formulation for agent tasks
- State-Thinking-Actions-Reward Policy Optimization (StarPO) algorithm that optimizes entire trajectory distributions
- Progressive reward normalization strategies to handle diverse, complex environments
-->

Reinforcement Learning (RL) with rule-based rewards has shown promise in enhancing reasoning capabilities of large language models (LLMs). However, existing approaches have primarily focused on static, single-turn tasks like math reasoning and coding. Extending these methods to agent scenarios introduces two fundamental challenges:

1. **Multi-turn Interactions**: Agents must perform sequential decision-making and react to environment feedback
2. **Stochastic Environments**: Uncertainty where identical actions can lead to different outcomes

To address these challenges, we propose a general RL framework: **StarPO** (**S**tate-**T**hinking-**A**ctions-**R**eward **P**olicy **O**ptimization), a comprehensive RL framework that provides a unified approach for training multi-turn, trajectory-level agents with flexible control over reasoning processes, reward assignment mechanisms, and prompt-rollout structures. 
Building upon StarPO, we introduce **RAGEN**, a modular agent training and evaluation system that implements the complete training loop, including rollout generation, reward calculation, and trajectory optimization. RAGEN serves as a robust research infrastructure for systematically analyzing LLM agent training dynamics in multi-turn and stochastic environments.

## Algorithm

RAGEN introduces a reinforcement learning framework to train reasoning-capable LLM agents that can operate in interactive, stochastic environments. 

<p align="center"><img src="public/starpo_logo.png" width="800px" alt="StarPO Framework" /></p>
<p align="center" style="font-size: 16px; max-width: 800px; margin: 0 auto;">
The StarPO (State-Thinking-Action-Reward Policy Optimization) framework with two interleaved stages: <b>rollout stage</b> and <b>update stage</b>. LLM iteratively generates reasoning-guided actions to interact with the environment to obtain trajectory-level rewards for LLM update to jointly   optimize reasoning and action strategies.
</p>

The framework consists of two key components:

### > MDP Formulation 
We formulate agent-environment interactions as Markov Decision Processes (MDPs) where states and actions are token sequences, allowing LLMs to reason over environment dynamics. At time t, state $s_t$ transitions to the next state through action $a_t$ following a transition function. The policy generates actions given the trajectory history. The objective is to maximize expected cumulative rewards across multiple interaction turns.

### > StarPO: Reinforcing Reasoning via Trajectory-Level Optimization
StarPO is a general RL framework for optimizing entire multi-turn interaction trajectories for LLM agents.
The algorithm alternates between two phases:

#### Rollout Stage: Reasoning-Interaction Trajectories
Given an initial state, the LLM generates multiple trajectories. At each step, the model receives the trajectory history and generates a reasoning-guided action: `<think>...</think><ans> action </ans>`. The environment receives the action and returns feedback (reward and next state).

#### Update Stage: Multi-turn Trajectory Optimization 
After generating trajectories, we train LLMs to optimize expected rewards. Instead of step-by-step optimization, StarPO optimizes entire trajectories using importance sampling. This approach enables long-horizon reasoning while maintaining computational efficiency. 
StarPO supports multiple optimization strategies: 
- PPO: We estimate token-level advantages using a value function over trajectories
- GRPO: We assign normalized reward to the full trajectory

Rollout and update stages interleave in StarPO, enabling both online and offline learning.

<!--
### > Reward Normalization Strategies 
We implement three progressive normalization strategies to stabilize training: 
1. **ARPO**: Preserves raw rewards directly 
2. **BRPO**: Normalizes rewards across each training batch using batch statistics
3. **GRPO**: Normalizes within prompt groups to balance learning across varying task difficulties
-->

## Environment Setup
For detailed setup instructions, please check our [documentation](https://ragen-tutorial.readthedocs.io/). Here's a quick start guide:

```bash
# Setup environment for RAGEN
bash scripts/setup_ragen.sh
```

If this fails, you can follow the manual setup instructions in `scripts/setup_ragen.md`.

## Training Models
Here's how to train models with RAGEN:

### Export variables and train
We provide default configuration in `config/base.yaml`. This file includes symbolic links to:
- `config/ppo_trainer.yaml` 
- `config/envs.yaml`

The base configuration automatically inherits all contents from these two config files, creating a unified configuration system.

To train:

```bash
python train.py --config-name base
```


### Parameter efficient training with LoRA

### Saving compute
By default our code is runnable on A100 80GB machines. If you are using machine with lower memory (e.g. RTX 4090), please consider adapting below parameters, like follows (performance might change due to smaller batch size and shorter context length):
```bash
python train.py \
  micro_batch_size_per_gpu=1 \ 
  ppo_mini_batch_size=8 \ 
  actor_rollout_ref.rollout.max_model_len=2048 \ 
  actor_rollout_ref.rollout.response_length=128 
```

#### Parameter efficient training with LoRA
We provide a default configuration with LoRA enabled in `config/base-lora.yaml`. To customize the LoRA settings, see the the `lora` section at the top of the configuration file. The current settings are:

```yaml
lora rank: 64
lora alpha: 64
actor learning rate: 1e-5
critic learning rate: 1e-4
```

<!--
## Supervised Finetuning (Optional)
For supervised finetuning with LoRA:

1. Create supervised finetuning data:
```bash
bash sft/generate_data.sh <env_type>
```

2. Finetune the model:
```bash
bash sft/finetune_lora.sh <env_type> <num_gpus> <save_path>
```

3. Merge LoRA weights with the base model:
```bash
python sft/utils/merge_lora.py \
    --base_model_name <base_model_name> \
    --lora_model_path <lora_model_path> \
    --output_path <output_path>
```
-->

## Visualization
Please check the `val/generations` metric in your wandb dashboard to see the trajectories generated by the model throughout training. Check this [relevant issue](https://github.com/RAGEN-AI/RAGEN/issues/84) for more information. 


## Performance

We evaluate RAGEN across multiple environments. Below are results Qwen-2.5-0.5B-Instruct on Sokoban, Frozenlake, and Bandit. 
- No KL loss or KL penalty was applied during training
- We selectively retained only the top 25% of trajectories that successfully completed their respective tasks

<p align="center" style="display: flex; justify-content: center; align-items: center; flex-direction: column; gap: 20px; max-width: 500px; margin: 0 auto;">
    <img src="public/exp1.png" width="250px" alt="Bandit" />
    <img src="public/exp2.png" width="250px"  alt="Simple Sokoban" />
    <img src="public/exp3.png" width="250px"  alt="Frozen lake" />
</p>

We demonstrate RAGEN's robust generalization ability by training on simple Sokoban environments (6×6 with 1 box) and successfully evaluating performance on:
- Larger Sokoban environments (8×8 with 2 boxes)
- Simple Sokoban with alternative grid vocabulary representations
- FrozenLake environments

<p align="center" style="display: flex; justify-content: center; align-items: center; flex-direction: column; gap: 20px; max-width: 500px; margin: 0 auto;">
    <img src="public/exp4.png" width="250px" alt="Larger Sokoban" />
    <img src="public/exp5.png" width="250px"  alt="Sokoban with Different Grid Vocabulary" />
    <img src="public/exp6.png" width="250px"  alt="Frozen lake" />
</p>

Key observations:
- By using no KL and filtering out failed trajectories, we can achieve better and stable performance
- Generalization results highlight RAGEN's capacity to transfer learned policies across varying environment complexities, representations, and domains.

## Evaluation
RAGEN provides a easy way to evaluate a model:
```bash
python -m ragen.llm_agent.agent_proxy --config-name <eval_config>
```
You only need to set model and environment to evaluate in `config/<eval_config>.yaml`.


<!--
## Example Trajectories

Visualization of agent reasoning on the Sokoban task:

<p align="center" style="display: flex; justify-content: center; gap: 10px;">
    <img src="./public/step_1.png" width="200px" alt="Step 1" />
    <img src="./public/step_2.png" width="200px" alt="Step 2" />
</p>

The visualizations show how the agent reasons through sequential steps to solve the puzzle.

## Case Studies
We provide several case studies showing the model's behavior:
- [Reward hacking](https://github.com/ZihanWang314/agent-r1/blob/main/cases/reward_hacking.txt)
- [Challenging moments](https://github.com/ZihanWang314/agent-r1/blob/main/cases/suck_moment.txt)

More case studies will be added to showcase both successful reasoning patterns and failure modes.
-->

## Modular System Design of RAGEN

We implement RAGEN as a modular system: there are three main modules: **Environment State Manager** (`ragen/llm_agent/es_manager.py`), **Context Manager** (`ragen/llm_agent/ctx_manager.py`), and **Agent Proxy** (`ragen/llm_agent/agent_proxy.py`).

- Environment State Manager (**es_manager**):
  - Supports multiple environments (different environments, same environment different seeds, same environment same seed)
  - Records states of each environment during rollout
  - Processes actions from **ctx_manager**, executes step, and returns action results (observations) to **ctx_manager** in a batch-wise manner
- Context Manager (**ctx_manager**):
  - Parses raw agent tokens into structured actions for the **es_manager**
  - Formats observation from **es_manager**, parses and formulates them for following rollout of agent.
  - Gathers final rollout trajectories and compiles them into tokens, attention masks, reward scores, and loss masks for llm updating.
- Agent Proxy (**agent_proxy**): Serves as the interface for executing single or multi-round rollouts

## Adding Custom Environments

To add a new environment to our framework:

1. Implement an OpenAI Gym-compatible environment in `ragen/env/new_env/env.py` with these required methods:
   - `step(action)`: Process actions and return next state
   - `reset(seed)`: Initialize environment with new seed
   - `render()`: Return current state observation
   - `close()`: Clean up resources

2. Define environment configuration in `ragen/env/new_env/config.py`

3. Register your environment in `config/envs.yaml`:
   ```yaml
   custom_envs:
     - NewEnvironment # Tag
       - env_type: new_env  # Must match environment class name
       - max_actions_per_traj: 50  # Example value
       - env_instruction: "Your environment instructions here"
       - env_config: {}  # Configuration options from config.py
   ```

4. Add the environment tag to the `es_manager` section in `config/base.yaml`

## Using RAGEN with dstack

[dstackai/dstack](https://github.com/dstackai/dstack) is an open-source container orchestrator that simplifies distributed training across cloud providers and on-premises environments
without the need to use K8S or Slurm.

### 1. Create fleet

Before submitting distributed training jobs, create a `dstack` [fleet](https://dstack.ai/docs/concepts/fleets).

### 2. Run a Ray cluster task

Once the fleet is created, define and apply a Ray cluster task:

```shell
$ dstack apply -f examples/distributed-training/ray-ragen/.dstack.yml
```

You can find the task configuration example at [`examples/distributed-training/ray-ragen/.dstack.yml`](https://github.com/dstackai/dstack/blob/master/examples/distributed-training/ray-ragen/.dstack.yml).

The `dstack apply` command will provision the Ray cluster with all dependencies and forward the Ray dashboard port to `localhost:8265`.


### 3. Submit a training job

Now you can submit a training job locally to the Ray cluster:

```shell
$ RAY_ADDRESS=http://localhost:8265
$ ray job submit \
    ...
```

See the full [RAGEN+Ray example](https://dstack.ai/examples/distributed-training/ray-ragen/).

For more details on how `dstack` can be used for distributed training, check out the [Clusters](https://dstack.ai/docs/guides/clusters/) guide.

## Feedback
We welcome all forms of feedback! Please raise an issue for bugs, questions, or suggestions. This helps our team address common problems efficiently and builds a more productive community.

## Awesome work powered or inspired by RAGEN
 - [VAGEN](https://github.com/RAGEN-AI/VAGEN): Training Visual Agents with multi-turn reinforcement learning
 - [Search-R1](https://github.com/PeterGriffinJin/Search-R1): Train your LLMs to reason and call a search engine with reinforcement learning
 - [ZeroSearch](https://github.com/Alibaba-nlp/ZeroSearch): Incentivize the Search Capability of LLMs without Searching
 - [Agent-R1](https://github.com/0russwest0/Agent-R1): Training Powerful LLM Agents with End-to-End Reinforcement Learning
 - [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): A live stream development of RL tunning for LLM agents
 - [MetaSpatial](https://github.com/PzySeere/MetaSpatial): Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse
 - [s3](https://github.com/pat-jj/s3): Efficient Yet Effective Search Agent Training via Reinforcement Learning


## Contributors

[**Zihan Wang**\*](https://zihanwang314.github.io/), [**Kangrui Wang**\*](https://jameskrw.github.io/), [**Qineng Wang**\*](https://qinengwang-aiden.github.io/), [**Pingyue Zhang**\*](https://williamzhangsjtu.github.io/), [**Linjie Li**\*](https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en), [**Zhengyuan Yang**](https://zyang-ur.github.io/), [**Xing Jin**](https://openreview.net/profile?id=~Xing_Jin3), [**Kefan Yu**](https://www.linkedin.com/in/kefan-yu-22723a25b/en/), [**Minh Nhat Nguyen**](https://www.linkedin.com/in/menhguin/?originalSubdomain=sg), [**Licheng Liu**](https://x.com/liulicheng10), [**Eli Gottlieb**](https://www.linkedin.com/in/eli-gottlieb1/), [**Yiping Lu**](https://2prime.github.io), [**Kyunghyun Cho**](https://kyunghyuncho.me/), [**Jiajun Wu**](https://jiajunwu.com/), [**Li Fei-Fei**](https://profiles.stanford.edu/fei-fei-li), [**Lijuan Wang**](https://www.microsoft.com/en-us/research/people/lijuanw/), [**Yejin Choi**](https://homes.cs.washington.edu/~yejin/), [**Manling Li**](https://limanling.github.io/)

*:Equal Contribution.

## Acknowledgements
We thank the [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1) team for providing the DeepSeek-R1 model and early conceptual inspirations. We are grateful to the [veRL](https://github.com/volcengine/verl) team for their infrastructure support. We thank the [TinyZero](https://github.com/Jiayi-Pan/TinyZero) team for their discoveries that informed our initial exploration. We would like to appreciate insightful discussions with Han Liu, Xinyu Xing, Li Erran Li, John Schulman, Akari Asai, Eiso Kant, Lu Lu, Runxin Xu, Huajian Xin, Zijun Liu, Weiyi Liu, Weimin Wu, Yibo Wen, Jiarui Liu, Lorenzo Xiao, Ishan Mukherjee, Anabella Isaro, Haosen Sun, How-Yeh Wan, Lester Xue, Matthew Khoriaty, Haoxiang Sun, Jiajun Liu.

## Star History

<p align="center" style="display: flex; justify-content: center; align-items: center; flex-direction: column; gap: 20px; max-width: 500px; margin: 0 auto;">
    <img src="public/star-history-202556.png" alt="" />
</p>

## Citation
If you find RAGEN useful, we would appreciate it if you consider citing our work:
```md
@misc{ragen,
      title={RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning}, 
      author={Zihan Wang and Kangrui Wang and Qineng Wang and Pingyue Zhang and Linjie Li and Zhengyuan Yang and Xing Jin and Kefan Yu and Minh Nhat Nguyen and Licheng Liu and Eli Gottlieb and Yiping Lu and Kyunghyun Cho and Jiajun Wu and Li Fei-Fei and Lijuan Wang and Yejin Choi and Manling Li},
      year={2025},
      eprint={2504.20073},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.20073}, 
}
```

Markdown Document: RAGEN Replay Buffer Implementation Explained

# RAGEN Framework: Replay Buffer Implementation Details

This document explains the implementation of the replay buffer functionality within the RAGEN framework, focusing on how experiences are collected, stored, sampled, and used for training the PPO agent. The primary file for these changes is `ragen/trainer/agent_trainer.py`.

## 1. Core Idea

The goal is to allow the PPO agent to train on experiences sampled from a buffer, rather than solely on the most recently generated (on-policy) trajectory. This involves:
    1.  Generating on-policy trajectories.
    2.  Storing these trajectories in a replay buffer.
    3.  When a PPO update is due, sampling a batch of trajectories from this buffer.
    4.  Using this sampled batch for calculating advantages and performing actor/critic updates.
    5.  Ensuring that all PPO-specific calculations (log_probs, values, advantages) on the sampled batch are performed using the *current* state of the actor and critic models.

## 2. Configuration (`config/base.yaml`)

The replay buffer's behavior is controlled by parameters in `config/base.yaml` under the `replay_buffer` section:

```yaml
replay_buffer:
  enable: true               # bool:   Enable (true) or disable (false) the replay buffer.
  capacity: 10000            # int:    Maximum number of trajectories (DataProto objects) the buffer can hold.
  sampling_batch_size: 128   # int:    Number of trajectories to sample from the buffer for one PPO training iteration.
                               #         If 0, on-policy data is used even if the buffer is enabled.
3. ReplayBuffer Class (ragen/trainer/replay_buffer.py)
This class is responsible for storing and providing samples of trajectories.

__init__(self, capacity: int, sampling_batch_size: int = None):
Initializes a collections.deque with maxlen=capacity. This deque stores DataProto objects (each representing a trajectory or a segment of experience).
The sampling_batch_size argument is optional here and not used internally by the buffer itself for its core logic, as the actual batch size for sampling is passed to the sample method.
add(self, experience: DataProto):
Appends a new experience (a DataProto object) to the internal deque. If the deque is full (i.e., at capacity), the oldest experience is automatically discarded (FIFO behavior).
sample(self, current_batch_size: int = None) -> list[DataProto]:
Requires current_batch_size to be provided (raises ValueError if None).
Randomly samples min(current_batch_size, len(self._buffer)) experiences from the deque.
Returns a list of DataProto objects.
__len__(self): Returns the current number of experiences in the buffer.
4. Integration into RayAgentTrainer (ragen/trainer/agent_trainer.py)
This is where the main logic for using the replay buffer resides, primarily within the __init__ and fit methods.

4.1. Initialization (RayAgentTrainer.__init__)
Based on self.config.replay_buffer.enable and self.config.replay_buffer.capacity, an instance of ReplayBuffer is created and assigned to self.replay_buffer.
# In RayAgentTrainer.__init__
if self.config.get('replay_buffer') and self.config.replay_buffer.enable:
    self.replay_buffer = ReplayBuffer(capacity=self.config.replay_buffer.capacity)
    print(f"Replay buffer enabled: capacity={self.config.replay_buffer.capacity}, sampling_batch_size={self.config.replay_buffer.sampling_batch_size}")
else:
    self.replay_buffer = None; print("Replay buffer disabled.")
4.2. Training Loop (RayAgentTrainer.fit)
This is the core of the interaction. Here's a step-by-step breakdown of the relevant logic within the while self.global_steps <= self.total_training_steps: loop:

Step 1: On-Policy Experience Generation

A new on-policy trajectory (or set of trajectories from parallel environments) is generated.
# In RayAgentTrainer.fit()
rollout_meta_info_dict = {"eos_token_id": self.tokenizer.eos_token_id, ... , "validate": False}
current_rollout_dp = DataProto(meta_info=deepcopy(rollout_meta_info_dict))

with _timer("gen", timing_raw):
    on_policy_dp = self.agent_proxy.rollout(current_rollout_dp, val=False)
    # This on_policy_dp is a DataProto object containing the newly generated experience.
This on_policy_dp is then filtered using _filter_rollout. The refactored _filter_rollout now returns a new DataProto object containing the filtered data.
# In RayAgentTrainer.fit()
on_policy_dp, filter_metrics_info = _filter_rollout(on_policy_dp, self.config)
metrics.update(filter_metrics_info)
# Log metrics for this on-policy generation
if hasattr(on_policy_dp,'meta_info') and on_policy_dp.meta_info and "metrics" in on_policy_dp.meta_info:
    metrics.update({"train/on_policy/" + k: v for k, v in on_policy_dp.meta_info["metrics"].items()})
Step 2: Adding to Replay Buffer

If the replay buffer is enabled, a deepcopy of the (potentially filtered) on_policy_dp is added to the buffer.
# In RayAgentTrainer.fit()
if self.replay_buffer: # Check if replay_buffer object exists
    self.replay_buffer.add(deepcopy(on_policy_dp))
Step 3: Batch Selection for PPO Update

This is where the decision is made whether to use data from the replay buffer or the just-generated on-policy data.
First, numerical metrics for logging the source are initialized:
# In RayAgentTrainer.fit()
metrics["train/source_is_replay"] = 0.0
metrics["train/source_is_on_policy_empty_sample_fallback"] = 0.0
metrics["train/source_is_on_policy_buffer_disabled"] = 0.0
metrics["train/source_is_on_policy_sampling_disabled"] = 0.0
metrics["train/source_is_on_policy_buffer_not_ready"] = 0.0
The selection logic:
# In RayAgentTrainer.fit()
# batch_source_info was removed, direct metric setting is used.
if self.replay_buffer and        len(self.replay_buffer) >= self.config.replay_buffer.sampling_batch_size and        self.config.replay_buffer.sampling_batch_size > 0:

    sampled_experiences_list = self.replay_buffer.sample(self.config.replay_buffer.sampling_batch_size)

    if sampled_experiences_list:
        batch = deepcopy(sampled_experiences_list[0]) if self.config.replay_buffer.sampling_batch_size == 1 else self._combine_data_protos(sampled_experiences_list)
        metrics["train/source_is_replay"] = 1.0

        if not hasattr(batch,'meta_info') or batch.meta_info is None: batch.meta_info = {}
        current_training_meta_info_dict = deepcopy(rollout_meta_info_dict)
        current_training_meta_info_dict.update(batch.meta_info)
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
        if self.replay_buffer is not None : metrics["train/replay_buffer_size"] = float(len(self.replay_buffer))
The batch variable now holds the DataProto object for this PPO training iteration.
Step 4: _combine_data_protos and _filter_rollout - TensorDict Handling

Both _filter_rollout (when filtering is applied) and _combine_data_protos (when combining multiple DataProtos) are now responsible for ensuring that the batch attribute of the DataProto objects they create is a proper TensorDict instance (or None).
They achieve this by:
Collecting filtered/combined tensors into a standard Python dictionary (filtered_batch_data or combined_batch).
Determining the common batch size of these tensors.
Creating a TensorDict from this dictionary and common batch size: final_td_batch = TensorDict(source=python_dict_of_tensors, batch_size=common_batch_size)
Passing this final_td_batch as the batch argument to the DataProto constructor.
# Example from _filter_rollout (similar logic in _combine_data_protos)
# ... after filtered_batch_data (python dict) is populated ...
final_td_batch = None
if filtered_batch_data:
    tensor_source_dict = {k: v for k, v in filtered_batch_data.items() if isinstance(v, torch.Tensor)}
    if tensor_source_dict:
        common_batch_size = next(iter(tensor_source_dict.values())).shape[:1]
        try:
            final_td_batch = TensorDict(source=tensor_source_dict, batch_size=common_batch_size)
        except Exception as e:
            print(f"Error creating TensorDict in _filter_rollout: {e}")
            final_td_batch = None
# ...
return DataProto(batch=final_td_batch, non_tensor_batch=filtered_non_tensor_data, meta_info=copied_meta_info), metrics
This was the fix for the AttributeError: 'dict' object has no attribute 'batch_size'.
Step 5: PPO Pipeline Processing

The batch (now correctly sourced, combined if necessary, and with its .batch attribute being a proper TensorDict or None) proceeds through the standard PPO calculations as detailed in previous explanations: UID generation, masking, balancing, reward computation, re-computation of log_probs and values using current models, advantage calculation, and finally actor/critic updates.
5. Verification Points for You
RayAgentTrainer.__init__:
Confirm self.replay_buffer initialization and the console printout.
RayAgentTrainer.fit() - Data Generation & Buffering:
Trace on_policy_dp after _filter_rollout().
Confirm self.replay_buffer.add() is called.
RayAgentTrainer.fit() - Batch Sampling:
Use breakpoints or print statements to observe:
len(self.replay_buffer).
The number of sampled_experiences_list.
The structure of batch after potential combination by _combine_data_protos. Specifically, check type(batch.batch). It should be <class 'tensordict.tensordict.TensorDict'> or None.
Monitor logged metrics:
metrics["train/replay_buffer_size"].
The metrics["train/source_is_*"] set should have one value as 1.0 and others as 0.0, correctly indicating the data source.
_filter_rollout() and _combine_data_protos() Correctness:
The absence of the AttributeError: 'dict' object has no attribute 'batch_size' when DataProto(...) is called within these functions is the primary indicator that they are now correctly producing TensorDict objects for the batch attribute.
PPO Pipeline:
Confirm the batch variable is passed through all subsequent PPO processing stages.
By observing the logged metrics for train/source_is_* and train/replay_buffer_size, you can directly verify if and when the replay buffer is being used and how it's filling up. Small, targeted experiments with specific capacity and sampling_batch_size values can help make this behavior clear.

This detailed walkthrough should help you trace the logic in the code.
