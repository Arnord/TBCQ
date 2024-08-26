
# -*- coding:utf8 -*-
import numpy as np

class BasePenalty:
    def __init__(self, weight_matrix, S=None, y_size=None, u_size=None):
        """
        Norm of weight_matrix should be bigger than S
        :param weight_matrix:
        :param S:
        """

        if weight_matrix is None:
            weight_matrix = [1, 1]

        if S is None:
            S = [1, 1]
        self.state_buffer = []
        self.position = -1
        self.capacity = 100
        self.weight_matrix = weight_matrix
        self.S = S
        self.y_size = y_size if y_size is not None else len(weight_matrix)
        self.u_size = u_size if u_size is not None else len(S)
        self.create()

    def change_list2diag(self, size, W=None):
        if W is None:
            W = np.diag(np.ones(size))
        else:
            W = np.array(W)
            if W.shape[0] != size:
                raise ValueError("size mismatch")
            W = np.diag(W)
        return W

    def create(self):

        self.weight_matrix = self.change_list2diag(self.y_size, self.weight_matrix)
        self.S = self.change_list2diag(self.u_size, self.S)

    def push(self, x):
        if len(self.state_buffer) < self.capacity:
            self.state_buffer.append(None)
        self.position = (self.position+1) % self.capacity
        self.state_buffer[self.position] = x

    def get_last_state(self):
        return self.state_buffer[self.position]

    def cal_penalty(self,y_star, y, u, c, d):
        raise NotImplementedError

    def cal(self, y_star, y, u, c, d):
        penalty = self.cal_penalty(y_star, y, u, c, d)
        self.push((y_star, y, u, c, d))
        return penalty




