#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import time
import json
import pandas as pd
import torch
import logging
from torch import nn
from numpy import *

from TBCQ.exp import OneRoundExp

# log class
class SimpleLogger(object):
    def __init__(self, f, header='#logger output'):
        dir = os.path.dirname(f)
        self.dir = dir
        self.begin_time_sec = time.time()
        # print('test dir', dir, 'from', f)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(f, 'w') as fID:
            fID.write('%s\n' % header)
        self.f = f

    def __call__(self, *args):
        # standard output
        print(*args)
        # log to file
        try:
            with open(self.f, 'a') as fID:
                fID.write('Time_sec = {:.1f} '.format(time.time() - self.begin_time_sec))
                fID.write(' '.join(str(a) for a in args) + '\n')
        except:
            print('Warning: could not log to', self.f)


def exp_select(exp_para):
    exp_type = exp_para.pop("type")
    if exp_type == "one_round_exp":
        exp = OneRoundExp(**exp_para)
    elif exp_type == "offline_exp":
        exp = None
    else:
        raise ValueError("Not identified exp type")

    return exp


def loss_visualization(outputs, text, fig_path, training_iters):
    from matplotlib import pyplot as plt
    import os

    fig_path = fig_path
    plt_label = text
    y_vae_loss = outputs['vae_loss_seq']
    y_critic_loss = outputs['critic_loss_seq']
    y_actor_loss = outputs['actor_loss_seq']
    x_data_step = [x for x in range(len(y_vae_loss))]


    # vae_loss
    plt.figure()
    plt.plot(x_data_step, y_vae_loss, color='red', label="vae_loss")
    # plt.axvline(best_epoch, color='r', linestyle='--', label='best epoch')
    plt.xlabel('Step')
    plt.ylabel('loss')
    plt.title(f'{plt_label}_vae_loss_{training_iters}')
    plt.legend()
    plt.savefig(os.path.join(fig_path, f'{plt_label}_vae_loss_{training_iters}.png'))
    plt.close()

    # critic_loss
    plt.figure()
    plt.plot(x_data_step, y_critic_loss, color='blue', label="critic_loss")
    # plt.axvline(best_epoch, color='r', linestyle='--', label='best epoch')
    plt.xlabel('Step')
    plt.ylabel('loss')
    plt.title(f'{plt_label}_critic_loss_{training_iters}')
    plt.legend()
    plt.savefig(os.path.join(fig_path, f'{plt_label}_critic_loss_{training_iters}.png'))
    plt.close()

    # actor_loss
    plt.figure()
    plt.plot(x_data_step, y_actor_loss, color='black', label="actor_loss")
    # plt.axvline(best_epoch, color='r', linestyle='--', label='best epoch')
    plt.xlabel('Step')
    plt.ylabel('loss')
    plt.title(f'{plt_label}_actor_loss_{training_iters}')
    plt.legend()
    plt.savefig(os.path.join(fig_path, f'{plt_label}_actor_loss_{training_iters}.png'))
    plt.close()

    # all
    plt.figure()
    plt.plot(x_data_step, y_vae_loss, color='red', label="vae_loss")
    plt.plot(x_data_step, y_critic_loss, color='blue', label="critic_loss")
    plt.plot(x_data_step, y_actor_loss, color='black', label="actor_loss")
    # plt.axvline(best_epoch, color='r', linestyle='--', label='best epoch')
    plt.xlabel('Step')
    plt.ylabel('loss')
    plt.title(f'{plt_label}_loss_{training_iters}')
    plt.legend()
    plt.savefig(os.path.join(fig_path, f'{plt_label}_loss_{training_iters}.png'))
    plt.close()






