# -*- coding:utf8 -*-
import pprint

import numpy as np

from TBCQ.exp.data_package import DataPackage

class OneRoundExp:
    def __init__(self, env=None, controller=None,
                 max_step=1000,
                 exp_name=None,
                 act_period=6,
                 ):
        """

        :param env:
        :param controller:
        :param max_step:
        :param exp_name:
        """

        self.env = env
        self.controller = controller
        self.max_step = max_step
        self.render_mode = False
        self.log = {}
        if exp_name is None:
            exp_name = "None"
        self.exp_name = exp_name
        self.act_period = act_period


    def add_log(self, key, value):
        self.log[key] = value

    def render(self):

        print('************Exp**************')
        pprint.pprint(self.log)
        print('************Exp**************')
        print()

    def run(self):

        state = self.env.reset()
        self.controller.step_reset()
        self.controller.env = self.env

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
            first_r = None
            final_next_state = None
            done = False
            # Consider an action being executed multiple times in a simulation environment
            for _ in range(self.act_period):
                next_state, r, done, _ = self.env.step(action)
                penalty_data.push(r)

                y_data.push(self.env.y_star[np.newaxis, :], 'set point')
                y_data.push(self.env.y)

                u_data.push(self.env.u)
                c_data.push(self.env.c)
                d_data.push(self.env.d)

                if first_r is None:
                    first_r = r
                final_next_state = next_state

            self.controller.train(state, action, final_next_state, first_r, done)
            state = final_next_state

        other_info = {}
        other_info['time_used'] = self.controller.time_used, self.controller.train_time_used, self.controller.act_time_used
        other_info['exp_name'] = self.exp_name
        res = y_data, u_data, c_data, d_data, penalty_data, other_info
        print('Experiment : %s finished!' % self.exp_name)

        return res
