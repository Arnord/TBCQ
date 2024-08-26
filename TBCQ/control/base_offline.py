
# -*- coding:utf8 -*-
from TBCQ.control.base_control import ControlBase
from TBCQ.common.normalize import BaseNormalizer
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import copy

class OfflineBase(ControlBase, nn.Module):
    def __init__(self,
                 y_size,
                 u_size,
                 c_size,
                 u_bounds=None,
                 gpu_id=0,
                 replay_buffer=None,
                 ts=False,
                 tl=None
                 ):
        """
        所有ac控制算法的父类
        :param gpu_id: 用于训练的gpuid
        :param u_bounds: 控制上下限
        :param exploration: 用于在输出的控制动作中添加探索
        :param replay_buffer: 用于存储经验
        :param ts: 是否使用时间序列
        :param tl: 时间序列长度
        """
        super(OfflineBase, self).__init__(y_size, u_size, c_size, u_bounds)
        nn.Module.__init__(self)
        self.device = None
        self.cuda_device(gpu_id)
        self.replay_buffer = replay_buffer
        # self.exploration = exploration

        # # 用于设定模型是否处于训练状态，该状态影响模型是否进行探索
        # self._train_mode = True
        # self.__normalizer = None

        self.ts = ts
        self.tl = tl

        self.s_list = []
        self.u_list = []
        self.ns_list = []
        self.r_list = []
        self.done_list = []


    # @property
    # def train_mode(self):
    #     return self._train_mode

    # @train_mode.setter
    # def train_mode(self,new_state):
    #     print('train_mode is changed to %s!' % "True" if new_state else "False")
    #     self._train_mode = new_state

    def cuda_device(self, cuda_id):
        use_cuda = torch.cuda.is_available() if cuda_id >= 0 else False
        cuda = 'cuda:'+str(cuda_id)
        self.device = torch.device(cuda if use_cuda else "cpu")

    def _train(self, s, u, ns, r, done):
        # 制作replay_buffer
        # 判定是否使用时间序列
        if self.replay_buffer is None:
            return None
        if not self.ts:
            self.replay_buffer.push(s, u, r, ns, done)
        else:
            self.s_list.append(s)
            self.u_list.append(u)
            self.ns_list.append(ns)
            self.r_list.append(r)
            self.done_list.append(done)
            if len(self.s_list) == self.tl:
                self.replay_buffer.push(copy.deepcopy(self.s_list), copy.deepcopy(self.u_list),
                                        copy.deepcopy(self.r_list), copy.deepcopy(self.ns_list),
                                        copy.deepcopy(self.done_list))
                self.s_list.pop(0)
                self.u_list.pop(0)
                self.ns_list.pop(0)
                self.r_list.pop(0)
                self.done_list.pop(0)

        # self.logging(f"buffer size {len(self.replay_buffer)}, step: {self.step} ")
        # self.update_model(replay_buffer)

    def _act(self, state):

        # 随机策略
        act_u = lambda x, al=self.u_bounds[:, 0], ah=self.u_bounds[:, 1]: np.random.uniform(al, ah)
        action = act_u(state)

        return action

    @property
    def normalizer(self):
        return self.__normalizer

    @normalizer.setter
    def normalizer(self, normalizer):
        assert isinstance(normalizer, BaseNormalizer)
        self.__normalizer = normalizer

    # TODO save models list
    def save_model(self, model_list=None, name_list=None):
        if model_list is None:
            pass

    # TODO load models list
    def load_model(self):
        pass


