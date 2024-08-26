# -*- coding:utf8 -*-
import pprint

import numpy as np

from TBCQ.exp.data_package import DataPackage
from TBCQ.exp.one_round_exp import OneRoundExp
from TBCQ.control.base_offline import OfflineBase

# 实验类：用于调度env和controller进行交互，并每隔一定的训练轮次(rounds)，对模型进行一次评估
class OfflineExp:
    def __init__(self, env=None, controller=None,
                 max_step=10000,
                 eval_freq=1000,
                 act_period=6,

                 exp_name=None,
                 logging=None
                 ):
        """

        :param env:
        :param controller:
        :param max_step: 最大训练次数
        :param eval_freq：模型验证频率（环境交互频率）
        :param eval_episodes: 每轮验证模型交互次数
        :param act_period: 同一动作与环境交互次数（高时延）

        :param exp_name:
        """

        self.env = env
        self.controller = controller

        self.max_step = max_step
        self.eval_freq = eval_freq
        self.act_period = act_period

        if exp_name is None:
            exp_name = "None"
        self.exp_name = exp_name
        self.logging = logging

    def run(self):

        state = self.env.reset()
        self.controller.step_reset()
        self.controller.env = self.env
        # 训练eval_cycle个round之后，进行一次模型评估

        y_data = DataPackage(exp_name=self.exp_name, value_name=self.env.y_name)
        u_data = DataPackage(exp_name=self.exp_name, value_name=self.env.u_name)
        d_data = DataPackage(exp_name=self.exp_name, value_name=self.env.d_name)
        c_data = DataPackage(exp_name=self.exp_name, value_name=self.env.c_name)
        penalty_data = DataPackage(exp_name=self.exp_name, value_name=['cost'])

        y_data.push(self.env.y_star[np.newaxis, :], 'set point')
        y_data.push(self.env.y[np.newaxis, :])
        controller_step = int(self.max_step/self.act_period)

        for step in range(controller_step):
            action = self.controller.act(state)

            # 考虑一个action在仿真环境中被连续执行多次，本质上修改了MDP。
            for _ in range(self.act_period):
                next_state, r, done, _ = self.env.step(action)
                penalty_data.push(r)
                # 记录目标值
                y_data.push(self.env.y_star[np.newaxis, :], 'set point')
                y_data.push(self.env.y)
                # 记录控制结果
                u_data.push(self.env.u)
                c_data.push(self.env.c)
                d_data.push(self.env.d)

            state = next_state

        other_info = {}
        other_info['time_used'] = self.controller.time_used, self.controller.train_time_used, self.controller.act_time_used
        other_info['exp_name'] = self.exp_name
        res = y_data, u_data, c_data, d_data, penalty_data, other_info
        # print('Experiment : %s finished!' % self.exp_name)

        return res
