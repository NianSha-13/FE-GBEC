# -*- coding: utf-8 -*-
"""
@Time ： 2023/5/13 12:05
@Auth ： daiminggao
@File ：granular_ball.py
@IDE ：PyCharm
@Motto:咕咕咕
"""
import numpy as np


class GranularBall:
    def __init__(self, data, gb_index,w=1):
        self.data = data
        self.center = self.data[:, :].mean(0)
        self.radius = self.get_radius()
        self.overlap = 0
        self.label = -1
        self.index=gb_index

        self.w = w  # 维度权重
        self.w_center = np.multiply(self.center, w)  # 加权中心
        self.w_center = self.w_center.flatten()
        self.w_radius = self.get_w_raduis()  # 加权半径
        #
        # self.center = self.w_center
        # self.radius = self.w_radius

    def get_radius(self):
        if len(self.data) == 1:
            return 0
        return max(((self.data[:, :] - self.center) ** 2).sum(axis=1) ** 0.5)
    

    def get_w_raduis(self):
        if len(self.data) == 1:
            return 0
        return max(((self.data[:, :] * self.w - self.w_center) ** 2).sum(axis=1) ** 0.5)
