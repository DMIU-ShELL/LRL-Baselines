#!/usr/bin/env python3
"""
Evaluate the final mask (last task) checkpoint across all tasks in the CT-graph curriculum.

Loads the last task checkpoint from a task_stats directory, sets that mask for
all evaluations, and reports per-task performance. Saves summary and full stats.
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from eval_ctgraph import _eval
from deep_rl import *
from deep_rl.utils.normalizer import ImageNormalizer


def load_config_and_tasks(env_config_path, log_dir, num_workers=1, seed=0, new_task_mask='linear_comb'):
    config = Config()
    config.env_name = Config.ENV_METACTGRAPH
    config.env_config_path = env_config_path
    config.lr = 0.00025
    config.cl_preservation = 'supermask'
    config.seed = seed
    random_seed(config.seed)
    config.tag = 'eval_final_mask'
    config.log_dir = log_dir
    config.num_workers = num_workers

    # get num_tasks from env_config
    import json
    with open(env_config_path, 'r') as f:
        env_config_ = json.load(f)
    num_tasks = env_config_['num_tasks']
    del env_config_

    task_fn = lambda log_dir: MetaCTgraphFlatObs(config.env_name, env_config_path, log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: MetaCTgraphFlatObs(config.env_name, env_config_path, log_dir)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(
        state_dim, action_dim, label_dim,
        phi_body=FCBody_SS(state_dim, task_label_dim=label_dim, hidden_units=(200, 200, 200), num_tasks=num_tasks, new_task_mask=new_task_mask),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks,
        new_task_mask=new_task_mask)
    config.policy_fn = None
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.01
    config.rollout_length = 512
    config.optimization_epochs = 8
    config.num_mini_batches = 64
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = None
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir, file_name='eval_final_mask')
    config.cl_requires_task_label = True
    config.eval_interval = 10
    config.task_ids = np.arange(num_tasks).tolist()

    agent = LLAgent(config)
    tasks = agent.config.cl_tasks_info
    return config, agent, tasks


def load_final_checkpoint(agent, task_stats_dir: Path):
    # pick highest task index checkpoint matching the model pattern
    ckpts = sorted(task_stats_dir.glob('*-model-*-task-*.bin'))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {task_stats_dir}")
    # filenames end with task-<n>.bin
    def _task_id(p: Path):
        try:
            return int(p.stem.split('task-')[-1])
        except Exception:
            return -1
    ckpts.sort(key=_task_id)
    final_ckpt = ckpts[-1]
    agent.load(final_ckpt)
    final_task_idx = _task_id(final_ckpt) - 1  # zero-based
    return final_ckpt, final_task_idx


def main():
    parser = argparse.ArgumentParser(description="Evaluate final task mask across all tasks.")
    parser.add_argument('algo', help='algorithm', choices=['ll_supermask'])
    parser.add_argument('path', help='path to experiment root (contains task_stats)')
    parser.add_argument('--env_config_path', default='./env_configs/ct28/seed1/meta_ctgraph_ct14_md.json')
    parser.add_argument('--new_task_mask', default='linear_comb')
    parser.add_argument('--seed', type=int, default=86)
    parser.add_argument('--mask-index', type=int, default=None, help='Optional mask index to use (0-based). Defaults to last task index in task_stats.')
    args = parser.parse_args()

    base_path = Path(args.path).resolve()
    task_stats_dir = base_path / 'task_stats'
    log_dir = base_path / 'eval_final_mask'
    log_dir.mkdir(parents=True, exist_ok=True)

    set_one_thread()
    select_device(0)

    config, agent, tasks = load_config_and_tasks(args.env_config_path, str(log_dir), num_workers=1, seed=args.seed, new_task_mask=args.new_task_mask)

    final_ckpt, final_task_idx = load_final_checkpoint(agent, task_stats_dir)
    if args.mask_index is not None:
        final_task_idx = args.mask_index
    config.logger.info(f"Loaded final checkpoint: {final_ckpt}; using mask index {final_task_idx}")

    # ensure seen_tasks mapping so task_eval_start can find indices
    agent.seen_tasks = {i: t['task_label'] for i, t in enumerate(tasks)}
    # override task_eval_start to always use the final mask index
    from deep_rl.mask_modules.mmn.mask_nets import set_model_task
    def _task_eval_start_override(task_label):
        set_model_task(agent.network, final_task_idx)
        agent.curr_eval_task_label = task_label
    agent.task_eval_start = _task_eval_start_override

    eval_data, ret = _eval(agent, tasks)

    with open(log_dir / 'eval_summary_final_mask.bin', 'wb') as f:
        pickle.dump(eval_data, f)
    with open(log_dir / 'eval_full_stats_final_mask.bin', 'wb') as f:
        pickle.dump(ret, f)
    config.logger.info('Evaluation with final mask across all tasks:')
    config.logger.info(eval_data)

    agent.close()


if __name__ == '__main__':
    main()
