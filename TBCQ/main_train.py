#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import math
import torch
import hydra
import traceback
import numpy as np
import pandas as pd
from numpy import *

# from .lib import util
from omegaconf import DictConfig, OmegaConf

from TBCQ.simulation.generate_env import generate_env
from TBCQ.model.generate_model import generate_model, generate_behavior_policy
from TBCQ.utils import sample_from_env
from TBCQ.common.normalize import EnvNormalizer
from TBCQ.exp import OneRoundExp, OfflineExp
from TBCQ.eval import OneRoundEvaluation, OfflineEvaluation
from TBCQ.common.replay.replay_buffer import ReplayBuffer


def set_random_seed(seed, logging):
    rand_seed = np.random.randint(0, 100000) if seed is None else seed
    logging('random seed = {}'.format(rand_seed))
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


def buffer_build(args, env, logging):
    """
    build replay_buffer
    :param policy:
    :param args:
    :param env:
    :param logging:
    :return:
    """

    behavior_policy = generate_behavior_policy(args, env)
    behavior_policy.replay_buffer = ReplayBuffer(capacity=int(args.train.buffer_size) + 1)
    if args.env.type == 'thickener':
        u_mean, u_std = [np.mean(env.u_bounds[0, :]), np.mean(env.u_bounds[1, :])], 1
        y_mean, y_std = [np.mean(env.y_bounds[0, :]), np.mean(env.y_bounds[1, :])], 1
        c_mean, c_std = [np.mean(env.c_bounds[0, :]), np.mean(env.c_bounds[1, :])], 1
        behavior_policy.normalizer = EnvNormalizer(u_mean, u_std, y_mean, y_std, c_mean, c_std)
    behavior_policy.logging = logging

    if args.env.type == 'thickener':
        env_params = f"{args.env.type}_noise_type_{args.env.noise_type}_ts_{args.env.ts}_penalty_{args.env.penalty_size}/"
    else:
        env_params = f"{args.env.type}"
    if args.train.ts:
        buffer_path = f"{os.getcwd().split('/ckpt/')[0]}/ckpt/buffers/" \
                      f"random_{args.random_seed}/" \
                      f"{env_params}" \
                      f"{args.behavior_policy.type}/buffer_size_{int(args.train.buffer_size)}_tl_{args.train.tl}"
        logging(f"buffer path:{buffer_path}")
    else:
        buffer_path = f"{os.getcwd().split('/ckpt/')[0]}/ckpt/buffers/" \
                      f"random_{args.random_seed}/" \
                      f"{env_params}" \
                      f"{args.behavior_policy.type}/buffer_size_{int(args.train.buffer_size)}"
        logging(f"buffer path:{buffer_path}")

    if os.path.exists(f"{buffer_path}") and ('replay_buffer.npy' in os.listdir(f"{buffer_path}")):
        logging(f"load buffer path: {buffer_path}")
        replay_buffer = ReplayBuffer(capacity=int(args.train.buffer_size))
        replay_buffer.load(f"{buffer_path}")
    else:
        logging(f"can not find buffer, buffer rebuilding......")
        if os.path.exists(f"{buffer_path}") is False:
            os.makedirs(buffer_path)

        exp_name = "buffer build"
        behavior_exp = OneRoundExp(controller=behavior_policy, env=env,
                          max_step=int(args.train.buffer_size),
                          act_period=int(args.train.act_period), exp_name=exp_name)
        logging('behavior exp start')
        res_list = [behavior_exp.run()]

        eval_res = OneRoundEvaluation(res_list=res_list, save_root=buffer_path)
        eval_res.plot_all(show=False)
        logging('behavior_policy evaluation images plt over')

        behavior_policy.replay_buffer.save(f"{buffer_path}")
        logging(f"buffer saved path: {buffer_path}")

        replay_buffer = behavior_policy.replay_buffer

    return replay_buffer


def policy_rebuild(args, env, policy, logging):
    if args.train.target_buffer is not True:
        replay_buffer = buffer_build(args, env, logging)
    elif args.train.mix_behavior_policy is not True:
        buffer_path = f"{os.getcwd().split('/ckpt/')[0]}/ckpt/{args.train.buffer_path_1}"
        replay_buffer = ReplayBuffer(capacity=int(args.train.buffer_size))
        replay_buffer.load(f"{buffer_path}")
        logging(f"buffer path: {buffer_path}")
    else:
        buffer_1 = ReplayBuffer(capacity=2 * int(args.train.buffer_size))
        buffer_2 = ReplayBuffer(capacity=int(args.train.buffer_size))
        buffer_1.load(f"{os.getcwd().split('/ckpt/')[0]}/ckpt/{args.train.buffer_path_1}")
        buffer_2.load(f"{os.getcwd().split('/ckpt/')[0]}/ckpt/{args.train.buffer_path_2}")
        buffer_1.merge(buffer_2)
        replay_buffer = buffer_1
        logging(f"buffer path: {args.train.buffer_path_1} and {args.train.buffer_path_2}")

    y_mean, y_std, u_mean, u_std, c_mean, c_std, r_mean, r_std = replay_buffer.normalizer_params()
    policy.replay_buffer = replay_buffer
    policy.normalizer = EnvNormalizer(u_mean, u_std, y_mean, y_std, c_mean, c_std, r_mean, r_std)
    logging(f"policy rebuild over")


def online_train_policy(args, logging, policy, env, exp_name=None, save_root=None, max_step=None, exp_step=None):
    """
    Using behavior policy to generate buffers for offline RL
    """
    if exp_name is None:
        exp_name = f"{args.model.type}"
    if save_root is None:
        save_root = os.getcwd()
    if max_step is None:
        max_step = args.train.max_timesteps
    if exp_step is None:
        exp_step = args.train.eval_freq

    policy.replay_buffer = ReplayBuffer(capacity=int(max_step) + 1)
    if args.env.type == 'thickener':
        u_mean, u_std = [np.mean(env.u_bounds[0, :]), np.mean(env.u_bounds[1, :])], 1
        y_mean, y_std = [np.mean(env.y_bounds[0, :]), np.mean(env.y_bounds[1, :])], 1
        c_mean, c_std = [np.mean(env.c_bounds[0, :]), np.mean(env.c_bounds[1, :])], 1
        policy.normalizer = EnvNormalizer(u_mean, u_std, y_mean, y_std, c_mean, c_std)

    training_iters = 0
    min_avg_cost = 1e10

    logging(f"---------------Online training starting---------------")

    while training_iters < int(max_step):
        logging('\n')
        logging(f"---------------Training iterations: {training_iters}---------------")

        exp_name_iter = exp_name + f"_{training_iters}"

        exp = OneRoundExp(controller=policy, env=env,
                          max_step=int(exp_step),
                          act_period=int(args.train.act_period), exp_name=exp_name_iter)
        res_list = [exp.run()]

        if training_iters % int(args.train.plt_freq) == 0:
            eval_res = OneRoundEvaluation(res_list=res_list, save_root=save_root)
            eval_res.plot_all(show=False)
            logging('policy evaluation images plt over')

        _, _, _, _, penalty_list, other_info = res_list[0]
        avg_cost = np.array(penalty_list.data[exp_name_iter]).astype('float32').mean(axis=0)
        avg_cost = avg_cost[0]

        logging(
            f"Evaluation over {int(args.train.eval_episodes)} episodes: avg_cost: {avg_cost:.3f}, "
            f"time_used: {other_info['time_used'][-1]}")

        if avg_cost < min_avg_cost:
            min_avg_cost = avg_cost
            torch.save(policy, os.path.join(os.getcwd(), 'best.pth'))
            torch.save(policy, os.path.join(os.getcwd(), 'control.pkl'))
            logging('policy saved')

        training_iters += int(args.train.eval_freq)

    # buffer save
    policy.replay_buffer.save(f"{save_root}")
    logging(f"buffer saved path: {save_root}")

    logging(f"{exp_name} exp finished, plots save path: {save_root}")

    return res_list


def offline_train_policy(args, logging, policy, env):
    """
    offline training
    """
    training_iters = 0
    min_avg_cost = 1e10

    logging(f"---------------Offline training starting---------------")

    while training_iters <= int(args.train.max_timesteps):
        logging('\n')
        logging(f"---------------Training iterations: {training_iters}---------------")
        pol_vals = policy.train(int(args.train.eval_freq))

        # if training_iters % int(args.train.eval_freq) == 0:
        logging(f"---------------Evaluating iterations: {training_iters}---------------")

        # eval_policy
        exp_name = f"{args.model.type}_{training_iters}"
        exp = OfflineExp(controller=policy, env=env, max_step=int(args.train.eval_episodes),
                         act_period=int(args.train.act_period), exp_name=exp_name)
        res_list = [exp.run()]

        if training_iters % int(args.train.plt_freq) == 0:
            eval_res = OfflineEvaluation(res_list=res_list, save_root=os.getcwd())
            eval_res.plot_all(show=False)
            logging('policy evaluation images plt over')

        _, _, _, _, penalty_list, other_info = res_list[0]
        avg_cost = np.array(penalty_list.data[exp_name]).astype('float32').mean(axis=0)
        avg_cost = avg_cost[0]

        logging(
            f"Evaluation over {int(args.train.eval_episodes)} episodes: avg_cost: {avg_cost:.3f}, "
            f"time_used: {other_info['time_used'][-1]}")

        # early stop
        if avg_cost < min_avg_cost:
            torch.save(policy, os.path.join(os.getcwd(), 'noise-2_tbcq-pid_buffer-5000_random-0_traj-1.pkl'))
            logging('best cost policy saved')
            min_avg_cost = avg_cost

        training_iters += int(args.train.eval_freq)

    # save model file
    torch.save(policy, os.path.join(os.getcwd(), 'final.pkl'))
    logging('final policy saved')


def main_train(args, logging):

    # set random seed
    set_random_seed(args.random_seed, logging)
    use_cuda = args.use_cuda and torch.cuda.is_available()

    # generate env with args
    env = generate_env(args)
    logging(f"env.W:{env.penalty_calculator.weight_matrix} \n"
            f"env.S:{env.penalty_calculator.S}")

    # generate model with args
    policy = generate_model(args, env)
    policy.logging = logging

    # Model load
    if use_cuda and args.model.type != "pid":
        policy = policy.cuda()

    logging('save dir = {}'.format(os.getcwd()))
    logging(policy)

    # 构建replay_buffer, 若为hdp类算法，则构建数据集训练预测模型
    env.reset()

    if args.model.exp_type == "offline":                        # offline 类model
        policy_rebuild(args, env, policy, logging)
        offline_train_policy(args, logging, policy, env)
    elif args.model.exp_type == "online":                           # online 类model
        online_train_policy(args, logging, policy, env)
    else:
        raise NotImplementedError

    # train_policy(args, logging, policy, env)


@hydra.main(config_path='config', config_name="config.yaml")
def main_app(args: DictConfig) -> None:
    from TBCQ.common.common import SimpleLogger

    logging = SimpleLogger('./log.out')

    # Save args for running model_test.py individually
    # util.write_DictConfig('./', 'exp.yaml', args)

    logging(OmegaConf.to_yaml(args))

    # Model Training
    try:
        main_train(args, logging)

    except Exception as e:
        var = traceback.format_exc()
        logging(var, '\n'
                "error: ", e)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_app()
