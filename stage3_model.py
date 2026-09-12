"""Neural digit controller on the real 256-node mask, with artificial interfaces."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import ConnectomeLM
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class DigitController(nn.Module):
    def __init__(self, condition='fly', seed=0, ticks=4, mode='discrete'):
        super().__init__()
        torch.manual_seed(seed)
        self.condition, self.mode, self.ticks = condition, mode, ticks
        self.disable_edges = False
        self.n = 256
        if condition == 'mlp':
            if mode != 'discrete': raise ValueError('MLP has no recurrent hidden state')
            self.mlp = nn.Sequential(nn.Linear(33,120),nn.Tanh(),nn.Linear(120,120),
                                     nn.Tanh(),nn.Linear(120,20))
        else:
            graph = dict(np.load(ROOT/'data/graph_256.npz'))
            self.core = ConnectomeLM(graph,1,'frozen' if condition=='no_edges' else condition,'rate',seed)
            del self.core.embedding
            del self.core.decoder
            self.input = nn.Linear(33,self.n)
            self.output = nn.Linear(self.n,20)
            self.gate_logit = nn.Parameter(torch.zeros(self.n))

    def matrix(self):
        if self.condition == 'mlp': return None
        self.core.disable_edges = self.disable_edges or self.condition == 'no_edges'
        return self.core.matrix()

    def forward(self, x, state=None, matrix=None):
        parts = [F.one_hot(x[:,0],3),F.one_hot(x[:,1],10),F.one_hot(x[:,2],10),
                 F.one_hot(x[:,3] if self.mode=='discrete' else torch.zeros_like(x[:,3]),10)]
        inputs = torch.cat(parts,dim=1).float()
        if self.condition == 'mlp':
            return self.mlp(inputs).reshape(-1,2,10), None
        if matrix is None: matrix = self.matrix()
        drive = self.input(inputs)
        h = torch.zeros_like(drive) if self.mode=='discrete' or state is None else state
        gate = self.gate_logit.sigmoid()
        for _ in range(self.ticks):
            h = (1-gate)*h + gate*torch.tanh(drive+F.linear(h,matrix))
        return self.output(h).reshape(-1,2,10), h


def make_model(config):
    return DigitController(config['condition'],config['seed'],config['ticks'],config['mode'])


def load_model(path):
    ckpt = torch.load(path,map_location='cpu',weights_only=True)
    model = make_model(ckpt['config'])
    model.load_state_dict(ckpt['model']); model.eval()
    return model, ckpt
