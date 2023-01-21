#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N is batch size; D_in is input dimension;
H is hidden dimension; D_out is output dimension.
"""

import torch
from torch import nn
from model import MLP


def train():
    N = 64
    D_in = 1000
    H = 100
    D_out = 10

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    # Construct our model by instantiating the class defined above.
    model = MLP(D_in, H, D_out)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    model.train()
    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = loss_fn(y_pred, y)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    PATH = 'model_output.pt'
    torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    train()
