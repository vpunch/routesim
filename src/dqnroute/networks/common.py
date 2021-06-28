import os

import torch
import torch.nn as nn
import torch.optim as optim

from ..constants import TORCH_MODELS_DIR

__activ_clses = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid
}

__optim_clses = {
    'rmsprop': optim.RMSprop,
    'adam': optim.Adam,
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad
}


def activ_class(name):
    return __activ_clses[name]


def optim_class(name):
    return __optim_clses[name]


class FFNetwork(nn.Sequential):
    def __init__(self, in_dim, out_dim, layers, activ_name):
        super().__init__()

        activation = activ_class(activ_name)()

        prev_dim = in_dim
        for i, layer in enumerate(layers):
            if type(layer) == int:
                self.add_module(f'fc_{i+1}', nn.Linear(prev_dim, layer))
                self.add_module(f'activation_{i+1}', activation)
                prev_dim = layer
            elif layer == 'dropout':
                self.add_module(f'dropout_{i+1}', nn.Dropout())

        self.add_module(f'output', nn.Linear(prev_dim, out_dim))


class SaveableModel(nn.Module):
    def save_path(self):
        return TORCH_MODELS_DIR + '/' + self.label

    def save(self):
        os.makedirs(TORCH_MODELS_DIR, exist_ok=True)
        return torch.save(self.state_dict(), self.save_path())

    def restore(self):
        return self.load_state_dict(torch.load(self.save_path()))
