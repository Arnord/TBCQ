#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import math
import time
import torch
import hydra
import traceback
import numpy as np
import pandas as pd
from numpy import *
from omegaconf import DictConfig, OmegaConf

from TBCQ.simulation.generate_env import generate_env


def set_random_seed(seed, logging):
    rand_seed = np.random.randint(0, 100000) if seed is None else seed
    logging('random seed = {}'.format(rand_seed))
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def eval_random(args, env, policy, seed):
    state = env.reset(seed)
    policy.step_reset()
    policy.env = env
    controller_step = int(args.test.single_step)
    data = [state]
    cost_list = []
    with torch.no_grad():
        for step in range(controller_step - 1):
            action = policy.act(state)
            next_state, cost, done, _ = env.step(action)
            cost_list.append(cost)
            data.append(next_state)
            state = next_state
    return cost_list

def test_random(args, logging, policy, env, exp_name=None, save_root=None, max_step=None, exp_step=None):

    if max_step is None:
        max_step = args.test.max_timesteps

    time_begin_all = time.time()

    cost_mult = []
    testing_iters = 0
    test_random_seed = 0

    logging(f"---------------test starting---------------")

    while testing_iters < int(max_step):
        logging('\n')
        logging(f"---------------Testing iterations: {testing_iters}---------------")
        time_begin = time.time()
        cost_list = eval_random(args, env, policy, test_random_seed)
        cost_mult.append(cost_list)

        avg_cost = np.clip(np.array(cost_list).mean(), 0, 500)

        time_end = time.time()
        time_used = time_end - time_begin

        logging(
            f"Random {test_random_seed} evaluation over {int(test_random_seed)} episodes: avg_cost: {avg_cost:.3f}, "
            f"time_used: {time_used}")

        testing_iters += 1
        test_random_seed += 1

    cost_array = np.vstack(np.clip(np.array(cost_mult), 0, 500))

    save_path = f"{args.model}-1e5_noise-{args.env.noise_type}_buffer-{args.test.buffer_size}.npy"
    np.save(os.path.join(os.getcwd(), save_path), cost_array, allow_pickle=True)

    cost_mean = cost_array.mean(axis=1).mean()
    cost_std = cost_array.mean(axis=1).std()
    cost_p95 = np.percentile(cost_array.mean(axis=1), 95)

    time_end_all = time.time()
    time_used_all = time_end_all - time_begin_all

    logging(
        f"Test over cost_mean: {cost_mean:.3f}, "
        f"Test over cost_std: {cost_std:.3f}, "
        f"Test over cost_p95: {cost_p95:.3f}, "
        f"time_used_all: {time_used_all}")

    return cost_array


def main_test(args, logging):

    set_random_seed(0, logging)
    use_cuda = args.use_cuda and torch.cuda.is_available()

    env = generate_env(args)
    logging(f"env.W:{env.penalty_calculator.weight_matrix} \n"
            f"env.S:{env.penalty_calculator.S}")

    model_random = args.random_seed
    root_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))

    model_path = f"{root_path}/model_file/noise-{args.env.noise_type}_{args.model.type}-pid_buffer-{int(args.test.buffer_size)}_random-{model_random}_traj-1.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = torch.load(model_path).to(device)

    logging('save dir = {}'.format(os.getcwd()))
    logging(policy)

    # Model Testing
    test_random(args, logging, policy, env)


@hydra.main(config_path='config', config_name="config.yaml")
def main_app(args: DictConfig) -> None:
    from TBCQ.common.common import SimpleLogger

    logging = SimpleLogger('./log.out')

    logging(OmegaConf.to_yaml(args))

    try:
        main_test(args, logging)

    except Exception as e:
        var = traceback.format_exc()
        logging(var, '\n'
                "error: ", e)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_app()
