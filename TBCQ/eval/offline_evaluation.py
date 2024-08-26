# -*- coding:utf8 -*-

from sklearn.metrics import mean_squared_error
from ..exp.data_package import cal_metric
import os

class OfflineEvaluation:
    def __init__(self, res_list, save_root):
        """

        :param res_list: result
        :param exp_name:
        :param y_name:
        :param training_rounds:
        :param y_num:
        :param penalty_plt_param:
        :param eval_plt_param:
        """

        self.y_list, self.u_list, self.c_list, self.d_list, self.penalty_list, other_info = zip(*res_list)
        self.res_list = res_list
        self.save_root = save_root
        self.img_root = os.path.join(save_root, 'images')
        os.makedirs(self.img_root, exist_ok=True)

        # 将每个实验块跑出来的结果合并。
        self.y_data = self.y_list[0].merge_list(self.y_list)
        self.u_data = self.u_list[0].merge_list(self.u_list)
        self.c_data = self.c_list[0].merge_list(self.c_list)
        self.d_data = self.d_list[0].merge_list(self.d_list)
        self.penalty_data = self.penalty_list[0].merge_list(self.penalty_list)
        self.grad_data = list()
        self.other_info = other_info

    def plot_all(self, show=False):
        self.y_data.plt(self.img_root, show=show)
        self.u_data.plt(self.img_root, show=show)
        self.c_data.plt(self.img_root, show=show)
        self.d_data.plt(self.img_root, show=show)
        self.penalty_data.plt(self.img_root, show=show)

    def evaluate(self, metric=None):
        if metric is None:
            metric = [('MSE', mean_squared_error)]
        self.plot_all()
        metrics_dict = cal_metric(self.y_data, metric)

        for item in self.res_list:
            other_info = item[-1]
            print('{}\t all time: {}, train time: {}, act time: {}'.format(
                other_info['exp_name'], *other_info['time_used'][:3]
            ))
        return metrics_dict
