#!/usr/bin/python

import random
import sys

# -*- coding:utf8 -*-
import numpy as np
from scipy.integrate import odeint

from TBCQ.simulation.base_env import BaseEnv


class Thickener(BaseEnv):
    def __init__(self, dt=1, penalty_calculator=None, noise_in=False, noise_p=0.01,
                 size_yudc=None, u_low=None,
                 u_high=None,
                 c_low=None,
                 c_high=None,
                 y_low=None,
                 y_high=None,
                 normalize=False,
                 time_length=120,
                 one_step_length=0.0005,
                 y_name=None,
                 u_name=None,
                 c_name=None,
                 c_start=None,
                 y_start=None,
                 y_star=None,
                 mean_c=None,
                 cov_c=None,
                 random_seed=None,
                 noise_type=None,

                 logging=None,
                 ts=False,
                 tl=None,
                 eta=None,

                 const_ff=False,
                 ):
        """

        :param dt:
        :param penalty_calculator:
        :param noise_in:
        :param size_yudc:
        :param u_low:
        :param u_high:
        :param normalize:
        :param time_length:
        :param one_step_length:
        :param y_name:
        :param c_name:
        :param y_star:
        :param noise_type: 1:mutation at 400 2:Random，3：mutation at 1600
        """

        if size_yudc is None:
            size_yudc = [2, 2, 0, 2]
        if y_name is None:
            y_name = ["Height", "Concentration(out)"]
        if c_name is None:
            c_name = ["pulp speed(Feed In)", "Concentration(In)"]
        if u_name is None:
            u_name = ["pulp speed(Feed out)", "pulp speed(Flocculant)"]

        if u_low is None:
            u_low = [40, 30]
        if u_high is None:
            u_high = [120, 50]

        if c_low is None:
            c_low = [34, 63]
        if c_high is None:
            c_high = [46, 83]

        if y_low is None:
            y_low = [0.75, 280]
        if y_high is None:
            y_high = [2.5, 1200]
        super(Thickener, self).__init__(
            dt=dt,
            penalty_calculator=penalty_calculator,
            size_yudc=size_yudc,
            u_low=u_low,
            u_high=u_high,
            c_low=c_low,
            c_high=c_high,
            y_low=y_low,
            y_high=y_high,
            normalize=normalize,
            time_length=time_length,
            one_step_length=one_step_length,
            y_name=y_name,
            u_name=u_name,
            c_name=c_name,
            random_seed=random_seed
        )

        self.name = 'thickener'

        # Random seed setting
        np.random.seed(self.random_seed)

        # Initial noise value
        if c_start is None:
            c_start = np.array([40, 73], dtype=float)
        self.c_start = np.array(c_start)

        # Initial value
        if y_start is None:
            y_start = np.array([1.5, 660], dtype=float)
        self.y_start = np.array(y_start)

        # target value
        if y_star is None:
            y_star = np.array([1.48, 680], dtype=float)
        self.y_star = np.array(y_star)

        # Mean and covariance of noise terms
        if mean_c is None:
            mean_c = np.copy(c_start)
        if cov_c is None:
            cov_c = np.array([[10, 0], [0, 18]])
        self.mean_c = np.array(mean_c)
        self.cov_c = np.array(cov_c)

        self.noise_in = noise_in
        self.noise_p = noise_p
        self.noise_type = noise_type

        self.param = {}

        self.param['rho_s'] = 4150  # Sand density in original paper

        self.param['rho_e'] = 1803  # Apparent density of the medium
        self.param['rho_l'] = 1000
        self.param['mu_e'] = 2      # Apparent viscosity of the medium
        self.param['d0'] = 0.00008  # Particle diameter at the inlet
        self.param['p'] = 0.5  # Average concentration coefficient

        self.param['A'] = 300.5   # Bottom area of thickener

        self.param['ks'] = 0.157  # Coefficient of action of flocculant

        self.param['ki'] = 0.0005 * 3600  # The concentration coefficient of the compression layer will affect the concentration at the height of the mud layer interface
        self.param['Ki'] = 50.0 / 3600  # The ratio of Feed flow rate(m^3/ s) and Feed pump speed(Hz，~40)
        self.param['Ku'] = 2.0 / 3600   # The ratio of Discharge flow rate(m^3/ s) and Discharge pump speed(Hz，~80)
        self.param['Kf'] = 0.75 / 3600  # The ratio of Flocculant flow rate(m^3/ s) and Pump speed of flocculant pump(Hz，~40)
        # Todo
        self.param['theta'] = 85
        self.param['theta'] = 3000
        self.param['theta'] = 2300
        self.param['g'] = 9.8

        self.logging = logging
        self.ts = ts
        self.tl = tl
        self.u_list = []
        self.last_u = None
        self.eta = eta

        self.const_ff = const_ff
        self.index = 0

    def reset_u(self):
        return np.array([80, 38], dtype=float)

    def reset_y(self):
        return np.copy(self.y_start)

    def reset_c(self):
        return np.copy(self.c_start)

    def reset_y_star(self):
        return np.copy(self.y_star)

    def observation(self):
        return np.hstack([self.y_star, self.y, self.u, self.c])

    def f(self, y, u, c, d):

        # Determination of Long Delay
        if self.ts is True:
            if self.last_u is None:
                pass
            else:
                u = 0.7 * u + (1 - 0.7) * self.last_u
            self.last_u = u

        # altitude, flow concentration
        ht, cu = tuple(y.tolist())
        # flow pump frequency, flocculant frequency
        fu, ff = tuple(u.tolist())
        # Feed pump speed, feed concentration
        fi, ci = tuple(c.tolist())

        # region ODE tool
        t_array = np.linspace(0, self.time_length, int(self.time_length / self.one_step_length))
        y_begin = y.tolist()

        # Simulate the physical settling process of the thickener
        y_new = odeint(self.cal_grad_inter, y_begin, t_array, args=(fu, ff, fi, ci, self.param,), )
        y = np.copy(y_new[-1, :])

        c = self.update_c(c)

        return y, u, c, d

    def update_c(self, c):

        if self.noise_in is False:
            return c

        if self.noise_type == 0:
            # Randomly change once every period of time
            if np.random.uniform(0, 1) < self.noise_p:
                c = np.random.multivariate_normal(mean=self.mean_c, cov=self.cov_c)
                c = self.bound_detect(c, self.c_bounds)[2]
        # Mutation at 400 time_step
        elif self.noise_type == 1:
            if self.time_step == 400:
                c = np.array([35, 65])

        # Random: stochastic process with bounds limitations
        elif self.noise_type == 2:
            # np.random.seed(self.random_seed)
            det_c_mean = (self.c_bounds[:, 0] + self.c_bounds[:, 1])/2 - c
            c = c + np.random.multivariate_normal(mean=0.001*det_c_mean, cov=[[0.8, 0], [0, 0.8]])
            # self.random_seed = np.random.randint(0, int(1e9))   # 暂时先固定噪音
            c = self.bound_detect(c, self.c_bounds)[2]

        # Random Gaussian
        elif self.noise_type == 3:
            c_list = [[46, 83], [40, 73], [34, 63]]
            if self.time_step % 100 == 0:
                c = c_list[self.index]
                self.index += 1
                self.index = self.index % 3
            c = c + np.random.multivariate_normal(mean=[0, 0], cov=[[0.1, 0], [0, 0.1]])
            c = self.bound_detect(c, self.c_bounds)[2]
        return c

    # 用那个常微分工具，不能直接在args中写**dict,建立一个中转的静态方法
    @staticmethod
    def cal_grad_inter(y, t, fu, ff, fi, ci, para):
        para['y'] = y
        para['fu'] = fu
        para['ff'] = ff
        para['fi'] = fi
        para['ci'] = ci
        return Thickener.cal_grad(**para)

    @staticmethod
    def cal_grad(
            y,
            fu, ff,
            fi, ci,
            rho_s,
            rho_e,
            rho_l,
            mu_e,
            d0,
            p,
            A,
            ks,
            ki,
            Ki,
            Ku,
            Kf,
            theta,
            g,

    ):
        ht, cu = y
        qi = Ki * fi  # 进料流量(m^3 / s)
        qu = Ku * fu  # 出料流量(m^3 / s)
        qf = Kf * ff  # 絮凝剂流量(m^3 / s)
        dt = ks * qf + d0  # 被絮凝了的粒子大小
        ut = dt * dt * (rho_s - rho_e) * g / (18 * mu_e)  # 修正后的粒子自由沉降速度
        ur = qu / A  # 由于底流导致的总体下行速度，大概是ut的10分之一左右
        cl = ki * qi * ci  # 泥层界面高度处的密度(kg/m^3)
        ca = p * (cl + cu)  # 泥层内平均密度
        wt = ci * qi  # 单位时间进入浓密机的固体质量
        # wt_out = cu * qu  # 单位时间浓密机排出的固体质量
        r = rho_l / rho_s * (rho_s - ca) / ca

        # print('ca = %f, r = %f' % (ca, r))

        # 定义中间计算变量， 具体含义参看文档

        a = ca
        b = ht
        c = cl * (ut + ur) - cu * ur
        # d = wt * theta / A / (ca * ca)
        d = wt * theta / A / rho_s * (-rho_s) / ca**2

        # 这个assert保证底流泵泵速增大时，底流浓度降低，泥层高度增加
        # 如果这里不满足条件说明控制器把浓密机控制坏了
        if not b > a * d:
            raise ValueError()

        assert b > a * d

        # y = c / (b - a * d)
        y = c / (b + a * d)
        x = d * y

        grad_ca = y
        grad_ht = x
        grad_cu = grad_ca / p

        return [grad_ht, grad_cu]





