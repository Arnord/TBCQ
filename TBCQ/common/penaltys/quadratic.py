
# -*- coding:utf8 -*-
import numpy as np

from TBCQ.common.penaltys.base_penalty_cal import BasePenalty
import torch

class Quadratic(BasePenalty):

    def __init__(self, weight_matrix, S, y_size=None, u_size=None, u_bounds=None):
        super().__init__(weight_matrix, S, y_size, u_size)
        self.u_bounds = u_bounds
        self.u_mid = np.mean(self.u_bounds, axis=1)

    def cal_penalty(self, y_star, y, u, c, d):
        weight_matrix = self.weight_matrix
        y_size = np.prod(y_star.shape)

        u_mid = np.mean(self.u_bounds, axis=1)

        tmp = (y_star-y).reshape(1, y_size)
        det_u = (u-u_mid).reshape(1,-1)

        """
        a is a row vector
        res = a * W * a.T + u * S * u.T
        """
        penalty_u = float(det_u.dot(self.S).dot(det_u.T))
        #penalty_u = penalty_u*penalty_u*penalty_u*100000
        res = float(tmp.dot(weight_matrix).dot(tmp.T)) + penalty_u
        return res

    def diff_cal_penalty(self, y_star, y, u, c):
        """
        :param y_star: the target output
        :param y: the current output
        :param u: the control input
        :param c: the disturbance
        Args:
            y_star: cql_torch.Tensor with shape (batch_size, y_size)
            y: cql_torch.Tensor with shape (batch_size, y_size)
            u: cql_torch.Tensor with shape (batch_size, u_size)
            c: cql_torch.Tensor with shape (batch_size, c_size)

        Returns: cql_torch.Tensor with shape(batch_size, 1)

        """
        device = y_star.device
        weight_matrix = torch.FloatTensor(self.weight_matrix).to(device)
        S = torch.FloatTensor(self.S).to(device)
        u_mid = torch.FloatTensor(self.u_mid).to(device)
        y_det = y_star-y
        det_u = u-u_mid
        penalty_u = (det_u @ torch.matmul(S, det_u.T)).diag().unsqueeze(dim=-1)
        penalty_y = (y_det @ torch.matmul(weight_matrix, y_det.T)).diag().unsqueeze(dim=-1)

        return penalty_u + penalty_y

    def sp_cal_penalty(self, y_star, y, u):
        """
                :param y_star: the target output
                :param y: the current output
                :param u: the control input
                :param c: the disturbance
                Args:
                    y_star: cql_torch.Tensor with shape (batch_size, y_size)
                    y: cql_torch.Tensor with shape (batch_size, y_size)
                    u: cql_torch.Tensor with shape (batch_size, u_size)
                    c: cql_torch.Tensor with shape (batch_size, c_size)

                Returns: cql_torch.Tensor with shape(batch_size, 1)

                """
        device = y_star.device
        weight_matrix = torch.FloatTensor(self.weight_matrix).to(device)
        S = torch.FloatTensor(self.S).to(device)
        if S == [0, 0]:
            S = [0.5, 0.5]
        u_mid = torch.FloatTensor(self.u_mid).to(device)
        y_det = y_star - y
        det_u = u - u_mid
        penalty_u = (det_u @ torch.matmul(S, det_u.T)).diag().unsqueeze(dim=-1)
        penalty_y = (y_det @ torch.matmul(weight_matrix, y_det.T)).diag().unsqueeze(dim=-1)

        return penalty_y, penalty_u

