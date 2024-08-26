#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import os
import json
import numpy as np
import random
from torch.utils.data import Dataset


def set_random_seed(seed):
    rand_seed = 0 if seed is None else seed
    print('random seed = {}'.format(rand_seed))
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def get_project_absoulte_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_sys_path():
    import sys
    project_path = (os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    sys.path.append(project_path)

class ChangePath:
    def __init__(self, new_path) -> None:
        self.new_path = new_path if new_path else os.getcwd()
        os.makedirs(self.new_path, exist_ok=True)

    def __enter__(self):
        self.origin_path = os.getcwd()
        os.chdir(self.new_path)
        return os.getcwd()

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.origin_path)


# save dict of np array to npz
def save_npz(dict, path):
    np.savez(path, **dict)


def load_npz(path):
    return np.load(path, allow_pickle=True)


# current time string
def get_current_time_str():
    import time
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def get_exp_path(exp_name, add_time=False):
    """
    :param exp_name: 实验名
    :return:
    """
    result_path = os.path.join(get_project_absoulte_path(), 'results', exp_name)
    if add_time:
        result_path = os.path.join(result_path, get_current_time_str())
    os.makedirs(result_path, exist_ok=True)
    return result_path


def sample_from_env(env, policy, rounds, input_choose_func=None, ob_choose_func=None, cache_dir=None,
                    use_cache=True, cache_name_addition='None', length=100, step=1, dilation=1, dtype=np.float32):
    """

    The default organization format for input and output is
    Input : [u]
    Observation : [y,c]

    :param env:
    :param policy:
    :param rounds:

    :return: Dataset
    """

    if cache_dir is None:
        cache_dir = os.path.join(get_project_absoulte_path(), 'results', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, f'{env.name}-{str(rounds)}')
    if cache_name_addition != 'None':
        npz_path += f'-{cache_name_addition}'
    npz_path += '.npz'

    if os.path.exists(npz_path) and use_cache:
        data = load_npz(npz_path)
        # extract the value of the dict whose key starts with 'x'
        x_all = [np.array(data[key], dtype=dtype) for key in data.keys() if key.startswith('x')]
        y_all = [np.array(data[key], dtype=dtype) for key in data.keys() if key.startswith('y')]
        return listarry2dataset(x_all, y_all, length, step, dilation)

    input_choose_func = lambda ys, y, u, c: u if input_choose_func is None else input_choose_func
    ob_choose_func = lambda ys, y, u, c: np.concatenate([y, c], axis=-1) if input_choose_func is None else ob_choose_func

    x_all = []
    y_all = []
    x_tmp = []
    y_tmp = []

    # Generate training data
    for _ in range(rounds):
        if _ % 100 == 0:
            print(f'Generating dataset in round {_+1}')
        ys, y, u, c = env.observation_split()
        y_tmp.append(np.concatenate([y, c], axis=-1)[np.newaxis, :])
        act = policy(env.observation())
        x_tmp.append(act[np.newaxis, :])
        env.step(act)

        if random.random() < 0.0005:
            env.reset()
            x_all.append(np.concatenate(x_tmp, axis=0))
            y_all.append(np.concatenate(y_tmp, axis=0))
            x_tmp = []
            y_tmp = []
    x_all.append(np.concatenate(x_tmp, axis=0))
    y_all.append(np.concatenate(y_tmp, axis=0))

    x_all = [np.array(x, dtype=dtype) for x in x_all]
    y_all = [np.array(y, dtype=dtype) for y in y_all]

    dx = {f'x{i}': x_all[i] for i in range(len(x_all))}
    dy = {f'y{i}': y_all[i] for i in range(len(y_all))}
    save_npz({**dx, **dy}, npz_path)

    return listarry2dataset(x_all, y_all, length, step, dilation)


def listarry2dataset(x, y, length, step, dilation):
    class IODataset(Dataset):
        def __init__(self, X, Y, length, step=5, dilation=1):

            self.dilation = dilation
            self.step = step
            self.length = length

            def iter(X):
                for x in X:
                    assert isinstance(x, np.ndarray)
                    for i in range(0, len(x) - length * dilation + 1, step):
                        indices = np.arange(i, i + (length - 1) * dilation + 1, dilation, dtype=np.int)
                        yield x[indices]

            slice_x = np.stack([x for x in iter(X)], axis=0)
            slice_y = np.stack([y for y in iter(Y)], axis=0)

            self.slice_x, self.x_mean, self.x_std = self.normalize(slice_x)
            self.slice_y, self.y_mean, self.y_std = self.normalize(slice_y)

            self.L = slice_x.shape[0]

        def normalize(self, data):
            mean = np.mean(data, axis=(0, 1))
            std = np.std(data, axis=(0, 1))
            return (data - mean) / std, mean, std

        def __len__(self):
            return self.L

        def __getitem__(self, item):
            x = self.slice_x[item]
            y = self.slice_y[item]
            return x, y

    return IODataset(x, y, length, step, dilation)

def dict2instance(dict, class_):
    return class_(**dict)

def instance2dict(instance):
    return instance.__dict__

