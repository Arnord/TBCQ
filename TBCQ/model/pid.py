#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
from TBCQ.control.base_offline import OfflineBase


class PID(OfflineBase):

    def __init__(self,
                 y_size,
                 u_size,
                 c_size,
                 u_bounds,

                 KP=2,
                 KI=0.025,
                 KD=0.001,
                 BTC=3,
                 forget_gamma=0.7,
                 root_value=None,

                 ts=False,
                 tl=None,

                 penalty_calculator=None,
                 replay_buffer=None,
                 logging=None):
        """
        The target variables are two-dimensional, namely the height of the mud layer and the concentration of the bottom flow
        :param KP:
        :param KI:
        :param KD:
        :param Base:
        :param BTC:
        :param max:
        :param min:
        :param initial_output:
        :param gamma:
        :param root_value:
        """
        super(PID, self).__init__(y_size=y_size, u_size=u_size, c_size=c_size, u_bounds=u_bounds,
                                  ts=ts, tl=tl)

        self.gamma = float(forget_gamma)
        self.KP = float(KP)
        self.KI = float(KI)
        self.KD = float(KD)
        self.BTC = float(BTC)
        self.root_value = root_value

        self.penalty_calculator = penalty_calculator
        self.replay_buffer = replay_buffer

        self.sum_err = [0, 0]
        self.last_err = [0, 0]
        self.last_output = root_value

        self.logging = logging


    def forget_sum_error(self, sum_error, dt):
        """

        :param sum_error:
        :param dt: seconds
        :return:
        """
        return sum_error * math.exp(-dt/self.gamma)

    def _act(self, state):

        y = state[self.indice_y]
        y_star = state[self.indice_y_star]

        dt = 1

        pv_out_list = []
        for i in range(len(y)):
            exp_val_c = float(y_star[i])
            error = exp_val_c - y[i]
            p_out = self.KP * error

            self.sum_err[i] = self.forget_sum_error(self.sum_err[i], dt) + error * dt
            i_out = self.KI * self.sum_err[i]

            derivative = (error - self.last_err[i]) / dt
            d_out = self.KD * derivative

            # output
            pv_out_list.append(p_out + i_out + d_out)
            self.last_err[i] = error

        # The target quantity is 2D, therefore the weighted average is used as the control output
        pv_out = ((1 - self.gamma) * pv_out_list[0] + self.gamma * pv_out_list[1]) / 2

        # The flocculant pump speed is directly proportional to the bottom flow concentration.
        # If delta_c=y_star - y is positive, then increase the flocculant pump speed and decrease the bottom flow pump speed
        # The speed of the bottom flow pump is inversely proportional to the concentration of the bottom flow
        output = [self.root_value[0] - self.BTC * pv_out, self.root_value[1] + self.BTC * pv_out]

        # clip output to u_bounds
        last_output_bounds = [[x - 1 for x in self.last_output], [x + 1 for x in self.last_output]]
        output = np.clip(output, last_output_bounds[0], last_output_bounds[1])
        act = np.clip(output, self.u_bounds[:, 0], self.u_bounds[:, 1])
        act = act + np.random.normal(0, 10 * 0.1, size=2)

        self.last_output = act

        return act


