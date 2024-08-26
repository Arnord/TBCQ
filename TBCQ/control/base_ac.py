
# -*- coding:utf8 -*-
from TBCQ.control.base_control import ControlBase
from TBCQ.common.normalize import BaseNormalizer
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np


class ACBase(ControlBase, nn.Module):
    def __init__(self,
                 y_size,
                 u_size,
                 c_size,
                 gpu_id=0,
                 replay_buffer = None,
                 u_bounds = None,
                 exploration = None,
                 ):
        """
        所有ac控制算法的父类
        :param gpu_id: 用于训练的gpuid
        :param replay_buffer: 经验回放池
        :param u_bounds: 控制上下限
        :param exploration: 用于在输出的控制动作中添加探索
        """
        super(ACBase, self).__init__(y_size, u_size, c_size, u_bounds)
        nn.Module.__init__(self)
        self.device = None
        self.cuda_device(gpu_id)
        self.replay_buffer = replay_buffer
        self.exploration = exploration

        # 用于设定模型是否处于训练状态，该状态影响模型是否进行探索
        self._train_mode = True
        self.__normalizer = None


    @property
    def train_mode(self):
        return self._train_mode

    @train_mode.setter
    def train_mode(self,new_state):
        print('train_mode is changed to %s!' % "True" if new_state else "False")
        self._train_mode = new_state

    def cuda_device(self, cuda_id):
        use_cuda = torch.cuda.is_available() if cuda_id >= 0 else False
        cuda = 'cuda:'+str(cuda_id)
        self.device = torch.device(cuda if use_cuda else "cpu")

    def _train(self, s, u, ns, r, done):
        raise NotImplementedError

    def policy_act(self, state):
        raise NotImplementedError

    def _act(self, state):

        # 计算增加噪音之前的策略
        action_before_noise = self.policy_act(state)
        if self.train_mode:

            # 训练过程中加入噪音，用于探索
            action_after_noise = self.exploration.add_noise(action_before_noise, self.step)
        else:
            action_after_noise = action_before_noise
        self.add_log("Action (After adding noise)", action_after_noise)
        self.add_log("Action (Before adding noise)", action_before_noise)
        return action_after_noise

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


