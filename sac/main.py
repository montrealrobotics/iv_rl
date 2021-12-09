import argparse
import wandb
from gym.envs.mujoco import *

import rlkit.torch.pytorch_util as ptu

from rlkit.data_management.env_replay_buffer import EnsembleEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger_custom, set_seed
from rlkit.samplers.data_collector import EnsembleMdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.networks import FlattenMlp, FlattenTwoHeadMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer

import torch 


def select_network(model_name):
    if model_name == "IV_SAC" or "Var" in model_name:
        return FlattenTwoHeadMlp
    else:
        return FlattenMlp


def get_env(env_name, seed):

    if env_name in ['gym_walker2d', 'gym_hopper',
                    'gym_cheetah', 'gym_ant']:
        from mbbl.env.gym_env.walker import env
    env = env(env_name=env_name, rand_seed=seed, misc_info={'reset_type': 'gym'})
    return env

def ensemble_experiment(model, variant):
    expl_env = NormalizedBoxEnv(get_env(variant['env'], variant['env_seed']))
    eval_env = NormalizedBoxEnv(get_env(variant['env'], variant['env_seed']))
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    
    M = variant['layer_size']
    num_layer = variant['num_layer']
    network_structure = [M] * num_layer
    
    NUM_ENSEMBLE = variant['num_ensemble']
    L_qf1, L_qf2, L_target_qf1, L_target_qf2, L_policy, L_eval_policy = [], [], [], [], [], []
    network = select_network(variant["model"])
    for _ in range(NUM_ENSEMBLE):
    
        qf1 = network(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=network_structure,
        )
        qf2 = network(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=network_structure,
        )
        target_qf1 = network(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=network_structure,
        )
        target_qf2 = network(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=network_structure,
        )
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=network_structure,
        )
        eval_policy = MakeDeterministic(policy)
        
        L_qf1.append(qf1)
        L_qf2.append(qf2)
        L_target_qf1.append(target_qf1)
        L_target_qf2.append(target_qf2)
        L_policy.append(policy)
        L_eval_policy.append(eval_policy)
    
    eval_path_collector = EnsembleMdpPathCollector(
        eval_env,
        L_eval_policy,
        NUM_ENSEMBLE,
        eval_flag=True,
    )
    
    expl_path_collector = EnsembleMdpPathCollector(
        expl_env,
        L_policy,
        NUM_ENSEMBLE,
        ber_mean=variant['ber_mean'],
        eval_flag=False,
        critic1=L_qf1,
        critic2=L_qf2,
        inference_type=variant['inference_type'],
        feedback_type=1,
    )
    
    replay_buffer = EnsembleEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        NUM_ENSEMBLE,
        log_dir=variant['log_dir'],
    )
    
    trainer = model(
        args=variant['args'],
        env=eval_env,
        policy=L_policy,
        qf1=L_qf1,
        qf2=L_qf2,
        target_qf1=L_target_qf1,
        target_qf2=L_target_qf2,
        num_ensemble=NUM_ENSEMBLE,
        feedback_type=1,
        temperature=variant['temperature'],
        temperature_act=0,
        expl_gamma=0,
        log_dir=variant['log_dir'],
        xi=variant['xi'],
        dynamic_xi=variant['dynamic_xi'],
        minimal_eff_bs=variant['minimal_eff_bs'],
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    
    algorithm.to(ptu.device)
    algorithm.train()


def sac_experiment(model, variant):
    expl_env = NormalizedBoxEnv(get_env(variant['env'], variant['seed']))
    eval_env = NormalizedBoxEnv(get_env(variant['env'], variant['seed']))
    
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    network = select_network(variant["model"])
    M = variant['layer_size']
    num_layer = variant['num_layer']
    network_structure = [M] * num_layer
    
    qf1 = network(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=network_structure,
    )
    qf2 = network(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=network_structure,
    )
    target_qf1 = network(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=network_structure,
    )
    target_qf2 = network(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=network_structure,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=network_structure,
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = model(
        args=variant['args'],
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




def run_sac(model, args):
    
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=args.num_episodes,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=args.batch_size,
            #save_frequency=args.save_freq,
        ),
        trainer_kwargs=dict(
            discount=args.gamma,
            soft_target_tau=args.soft_target_tau,
            target_update_period=1,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        num_ensemble=args.num_nets,
        num_layer=args.num_layer,
        seed=args.env_seed,
        ber_mean=args.mask_prob,
        env=args.env,
        inference_type=args.ucb_lambda,
        temperature=args.sunrise_temp,
        log_dir="",
    )
    
    #wandb.init(settings=wandb.Settings(start_method='fork'), project="iv_rl", entity="montreal_robotics", config=args, name="%s_%s_%.2f_%d"%(args.env, "EnsembleSAC", args.temperature, args.seed))

    set_seed(args.net_seed)
    exp_name = args.model
    log_dir = setup_logger_custom(exp_name, variant=variant)
            
    variant['log_dir'] = log_dir
    variant["xi"] = args.xi
    variant['dynamic_xi'] = args.dynamic_xi
    variant['minimal_eff_bs'] = args.minimal_eff_bs
    variant['env_seed'] = args.env_seed
    variant['net_seed'] = args.net_seed
    variant['model'] = args.model
    variant['args'] = args
    ptu.set_gpu_mode(True)
    if "SAC" == args.model or "VarSAC" in args.model:
        sac_experiment(model, variant)
    else:
        ensemble_experiment(model, variant)

