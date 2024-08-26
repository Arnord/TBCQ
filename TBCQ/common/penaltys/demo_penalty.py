
# -*- coding:utf8 -*-
import numpy as np

from TBCQ.common.penaltys.base_penalty_cal import BasePenalty
class DemoPenalty(BasePenalty):

    def cal_penalty(self, y_star, y, u, c, d):
        weight_matrix = self.weight_matrix
        y_size = np.prod(y_star.shape)

        tmp = (y_star-y).reshape(1, y_size)

        """
        a is a row vector
        res = a * W * a.T
        """
        res = float(tmp.dot(weight_matrix).dot(tmp.T))
        return res
