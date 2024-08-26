#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import torch
from TBCQ.model import TBCQ, PID
from TBCQ.control.base_offline import OfflineBase

def generate_model(args, env):
    if args.model.type == 'tbcq':
        model = TBCQ(
            y_size=env.size_yudc[0],
            u_size=env.size_yudc[1],
            c_size=env.size_yudc[3],
            u_bounds=env.u_bounds,

            max_u=args.model.max_u,
            discount=args.model.discount,
            tau=args.model.tau,
            lmbda=args.model.lmbda,
            phi=args.model.phi,
            lr=args.model.lr,

            ts=args.train.ts,
            tl=args.train.tl,

            batch_size=int(args.train.batch_size),
        )
    elif args.model.type == 'pid':
        model = PID(
            y_size=env.size_yudc[0],
            u_size=env.size_yudc[1],
            c_size=env.size_yudc[3],
            u_bounds=env.u_bounds,

            KP=args.model.KP,
            KI=args.model.KI,
            KD=args.model.KD,
            BTC=args.model.BTC,
            forget_gamma=args.model.forget_gamma,
            root_value=args.model.root_value,

            ts=args.train.ts,
            tl=args.train.tl,

            penalty_calculator=env.penalty_calculator
        )
    elif args.model.type == 'random':
        model = OfflineBase(
            y_size=env.size_yudc[0],
            u_size=env.size_yudc[1],
            c_size=env.size_yudc[3],
            u_bounds=env.u_bounds,

            ts=args.train.ts,
            tl=args.train.tl,
        )
    else:
        raise NotImplementedError
    return model



def generate_behavior_policy(args, env):
    """
    :param args:
    :param env:
    :return:
    """

    if args.behavior_policy.type == 'random':
        behavior_policy = OfflineBase(
            y_size=env.size_yudc[0],
            u_size=env.size_yudc[1],
            c_size=env.size_yudc[3],
            u_bounds=env.u_bounds,
            ts=args.train.ts,
            tl=args.train.tl,
        )
    elif args.behavior_policy.type == 'pid':
        behavior_policy = PID(
            y_size=env.size_yudc[0],
            u_size=env.size_yudc[1],
            c_size=env.size_yudc[3],
            u_bounds=env.u_bounds,

            KP=args.behavior_policy.KP,
            KI=args.behavior_policy.KI,
            KD=args.behavior_policy.KD,
            BTC=args.behavior_policy.BTC,
            forget_gamma=args.behavior_policy.forget_gamma,
            root_value=args.behavior_policy.root_value,

            ts=args.train.ts,
            tl=args.train.tl,

            penalty_calculator=env.penalty_calculator
        )
    else:
        raise NotImplementedError
    return behavior_policy

