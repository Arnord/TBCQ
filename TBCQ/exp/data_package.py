#!/usr/bin/python
# -*- coding:utf8 -*-
import collections

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import os


class DataPackage:
    def __init__(self, exp_name=None, value_name=None):
        """
        :param exp_name:
        :param value_name:
        :param para:
        """
        if exp_name is None:
            raise ValueError('exp_name should not be none')
        self.exp_name = exp_name
        self.value_name = value_name
        self.size = None
        self.data = collections.defaultdict(list)

    def push(self, x, name=None):
        """

        :param x: shape(1,x)
        :param name:
        :return:
        """
        value = np.array(x).reshape(-1)
        if self.size is None:
            self.size = value.shape[0]
        else:
            if self.size != value.shape[0]:
                raise ValueError("Dimensional inconsistency! of DataPackage %s", self.exp_name)
        if name is None:
            self.data[self.exp_name].append(value)
        else:
            self.data[name].append(value)

    def merge(self, dp):
        if not isinstance(dp, DataPackage):
            raise ValueError('merged object should be an instance of DataPackage')
        for (key, values) in dp.data.items():
            self.data[key] = values

    def merge_list(self, dp_list):
        for dp in dp_list:
            self.merge(dp)
        return self

    def plt(self, img_root, show=False, para=None):
        para = {} if para is None else para
        if self.value_name is None:
            self.value_name = [str(i) for i in range(self.size)]

        if self.size == 0:
            return
        if len(self.value_name) != self.size:
            raise ValueError('size of value_name and size are not match')
        for pic_id in range(self.size):
            plt.figure(**para)
            legend_name = []
            for (key, values) in self.data.items():
                values_array = np.array(values)
                line_color = 'k' if key == 'set point' else None
                legend_name.append(key)
                x_array = np.arange(0, values_array.shape[0], 1)
                plt.plot(x_array, values_array[:, pic_id], c=line_color)
                plt.legend(legend_name)

            plt.title(f"{self.exp_name}_{self.value_name[pic_id]}")
            plt.xlabel('Steps')

            try:
                plt.savefig(
                    os.path.join(img_root, str(self.exp_name)+str(self.value_name[pic_id])+'_'.join(legend_name)+'.png'),
                    dpi=110
                )
                plt.close()
            except FileNotFoundError:
                print('Not given directory for saving images')
                pass

            if show:
                plt.show()


# repeat the first dim of nd array n times

def cal_metric(data_package, metric_name_func=None):
    if not isinstance(data_package, DataPackage):
        raise ValueError(' The first param should be an instance of DataPackage')

    if metric_name_func is None:
        metric_func = [('MSE', mean_squared_error)]
    metric_dict = collections.defaultdict(list)
    if "set point" not in data_package.data.keys():
        return

    for (metric_name, metric_func) in metric_name_func:
        for (key, values) in data_package.data.items():
            if key == 'set point':
                continue
            metric_dict[key].append(('all', metric_name, metric_func(np.array(data_package.data['set point']), np.array(values))))
            for pic_id in range(data_package.size):
                set_point = np.array(data_package.data['set point'])[:, pic_id]
                values_array = np.array(values)
                metric_dict[key].append(
                    (data_package.value_name[pic_id], metric_name, metric_func(set_point, values_array[:, pic_id]))
                )
    return metric_dict


