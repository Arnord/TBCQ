#!/usr/bin/python
# -*- coding:utf8 -*-
import os

from TBCQ.simulation.thickener import Thickener
from TBCQ.common.penaltys import Quadratic

def select_penalty(args, env):
    if args.env.penalty_type == 'quadratic':
        if args.env.penalty_size == 'high':
            weight_matrix = [0, 1]
            S = [0.5, 0.5]
        elif args.env.penalty_size == 'low':
            weight_matrix = [0, 0.1]
            S = [0.1, 0.1]
        else:
            raise NotImplementedError

        if args.env.cost_y is True:
            S = [0, 0]

        penalty_para = {
            "y_size": env.y_size,
            "u_size": env.u_size,
            "weight_matrix": weight_matrix,
            "S": S,
            'u_bounds': env.u_bounds
        }
        penalty_calculator = Quadratic(**penalty_para)
    else:
        raise NotImplementedError
    return penalty_calculator


def generate_env(args):
    if args.env.type == 'thickener':
        env = Thickener(
            dt=args.env.dt,
            noise_in=args.env.noise_in,
            noise_p=args.env.noise_p,
            noise_type=args.env.noise_type,
            time_length=args.env.time_length,

            y_start=args.env.y_start,
            y_star=args.env.y_star,

            ts=args.env.ts,
            tl=args.env.tl,
            eta=args.env.eta,

            random_seed=args.random_seed,
        )
        penalty_calculator = select_penalty(args, env)
        env.penalty_calculator = penalty_calculator
    else:
        raise NotImplementedError
    return env

