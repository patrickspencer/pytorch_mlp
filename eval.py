#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from model import MLP

PATH = 'model_output.pt'
model = MLP()
model.load_state_dict(torch.load(PATH))
model.eval()

N = 10
D_in = 1000

x = torch.randn(N, D_in)

y_pred = model(x)
print(y_pred)
