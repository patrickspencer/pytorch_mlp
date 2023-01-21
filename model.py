#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn


class MLP(nn.Module):
    def __init__(self, D_in=1000, H=100, D_out=10):
        """
        3 layer neural network with dropout and final sigmoid activation
        """
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()

    def forward(self, x):
        """
        Simple forward function
        """
        y0 = self.dropout(self.linear1(x).clamp(min=0))
        y1 = self.sigmoid(self.linear2(y0))
        return y1
