#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import json
import shutil
from deep_rl import *
import argparse
import numpy as np


def ppo_baseline_continualworld(name, args):
    env_config_path = args.env_config_path

    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 5e-4
    config.cl_preservation = 'baseline'
    config.seed = args.seed
    random_seed(config.seed)
    exp_id = '-{0}'.format(config.seed)
    log_name = name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 1

    with open(env_config_path, 'r') as f:
        env_config_ = json.load(f)
    num_tasks = len(env_config_['tasks'])

    task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: GaussianActorCriticNet_CL(
        state_dim, action_dim, label_dim,
        phi_body=DummyBody_CL(state_dim, task_label_dim=label_dim),
        actor_body=FCBody_CL(state_dim + label_dim, hidden_units=(128, 128), gate=torch.tanh),
        critic_body=FCBody_CL(state_dim + label_dim, hidden_units=(128, 128), gate=torch.tanh))
    config.policy_fn = SamplePolicy
    config.state_normalizer = RunningStatsNormalizer()
    config.reward_normalizer = RewardRunningStatsNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.97
    config.entropy_weight = 5e-3
    config.rollout_length = 512 * 10
    config.optimization_epochs = 16
    config.num_mini_batches = 160
    config.ppo_ratio_clip = 0.2
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = args.max_steps
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir, file_name='train-log')
    config.cl_requires_task_label = True

    config.eval_interval = 200
    config.task_ids = np.arange(num_tasks).tolist()

    agent = BaselineAgent(config)
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    config.cl_num_learn_blocks = 1
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_w_oracle(agent, tasks)
    with open('{0}/tasks_info_after_train.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)


def ppo_ll_continualworld_si(name, args):
    env_config_path = args.env_config_path

    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 5e-4
    config.cl_preservation = 'si_multi_head' if args.multi_head is True else 'si_single_head'
    config.seed = args.seed
    random_seed(config.seed)
    exp_id = '-{0}'.format(config.seed)
    log_name = name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 1

    # SI hyperparameters
    config.si_epsilon = 1e-3
    config.cl_loss_coeff = 1e4

    with open(env_config_path, 'r') as f:
        env_config_ = json.load(f)
    num_tasks = len(env_config_['tasks'])

    task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)
    if args.multi_head is True:
        config.network_fn = lambda state_dim, action_dim, label_dim: GaussianActorCriticNet_CL_MultiHead(
            state_dim, action_dim, num_tasks,
            task_label_dim=label_dim,
            phi_body=FCBody_CL(state_dim, task_label_dim=label_dim, hidden_units=(200, 200, 200)),
            actor_body=DummyBody_CL(200),
            critic_body=DummyBody_CL(200))
    else:
        config.network_fn = lambda state_dim, action_dim, label_dim: GaussianActorCriticNet_CL(
            state_dim, action_dim, label_dim,
            phi_body=FCBody_CL(state_dim, task_label_dim=label_dim, hidden_units=(200, 200, 200)),
            actor_body=DummyBody_CL(200),
            critic_body=DummyBody_CL(200))
    config.policy_fn = SamplePolicy
    config.state_normalizer = RunningStatsNormalizer()
    config.reward_normalizer = RewardRunningStatsNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.97
    config.entropy_weight = 5e-3
    config.rollout_length = 512 * 10
    config.optimization_epochs = 16
    config.num_mini_batches = 160
    config.ppo_ratio_clip = 0.2
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = args.max_steps
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir, file_name='train-log')
    config.cl_requires_task_label = True

    config.eval_interval = 200
    config.task_ids = np.arange(num_tasks).tolist()

    agent = LLAgentSI(config)
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    config.cl_num_learn_blocks = 1
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_w_oracle(agent, tasks)
    with open('{0}/tasks_info_after_train.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)


if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    select_device(-1)

    parser = argparse.ArgumentParser()
    parser.add_argument('algo', help='algorithm to run')
    parser.add_argument('--env_name', help='evaluation environment', default='continualworld')
    parser.add_argument('--env_config_path', help='path to environment config', default='./env_configs/continualworld_10.json')
    parser.add_argument('--max_steps', help='maximum steps per task', default=10_240_000, type=int)
    parser.add_argument('--seed', help='seed for the experiment', default=8379, type=int)
    parser.add_argument('--multi_head', help='use multi-head network', default=False, action='store_true')
    args = parser.parse_args()

    if args.env_name == 'continualworld':
        name = Config.ENV_CONTINUALWORLD
        if args.algo == 'baseline':
            ppo_baseline_continualworld(name, args)
        elif args.algo == 'si':
            ppo_ll_continualworld_si(name, args)
        else:
            raise ValueError('algo {0} not implemented'.format(args.algo))
    else:
        raise ValueError('--env_name {0} not implemented'.format(args.env_name))

