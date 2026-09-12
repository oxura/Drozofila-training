"""Action policies on the source graph, rewired graph, or parameter-matched MLP."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import ConnectomeLM

ROOT=Path(__file__).resolve().parent


class TapePolicy(nn.Module):
    def __init__(self,condition='fly',mode='reactive',seed=0,ticks=4):
        super().__init__();torch.manual_seed(seed)
        self.condition,self.mode,self.ticks=condition,mode,ticks
        self.disable_edges=False
        if condition=='gru':
            self.gru=nn.GRUCell(34,64);self.output=nn.Linear(64,14)
        elif condition=='mlp':
            if mode not in ['reactive','blind']:raise ValueError('MLP is stateless')
            self.mlp=nn.Sequential(nn.Linear(34,119),nn.Tanh(),nn.Linear(119,119),nn.Tanh(),nn.Linear(119,14))
        else:
            graph=dict(np.load(ROOT/'data/graph_256.npz'))
            self.core=ConnectomeLM(graph,1,'frozen' if condition=='no_edges' else condition,'rate',seed)
            del self.core.embedding;del self.core.decoder
            self.input=nn.Linear(34,256);self.output=nn.Linear(256,14)
            self.gate_logit=nn.Parameter(torch.zeros(256))

    def matrix(self):
        if self.condition in ['mlp','gru']:return None
        self.core.disable_edges=self.disable_edges or self.condition=='no_edges'
        return self.core.matrix()

    def forward(self,observations,state=None,matrix=None):
        op,cell,last=observations.unbind(-1)
        if self.mode in ['hidden','blind']:last=torch.full_like(last,14)
        inputs=torch.cat([F.one_hot(op,7),F.one_hot(cell,12),F.one_hot(last,15)],dim=-1).float()
        if self.condition=='mlp':return self.mlp(inputs),None
        if self.condition=='gru':
            h=inputs.new_zeros((len(inputs),64)) if state is None or self.mode in ['reactive','blind'] else state
            for _ in range(self.ticks):h=self.gru(inputs,h)
            return self.output(h),h
        if matrix is None:matrix=self.matrix()
        drive=self.input(inputs)
        h=torch.zeros_like(drive) if state is None or self.mode in ['reactive','blind'] else state
        gate=self.gate_logit.sigmoid()
        for _ in range(self.ticks):h=(1-gate)*h+gate*torch.tanh(drive+F.linear(h,matrix))
        return self.output(h),h


def make_model(config):
    return TapePolicy(config['condition'],config['mode'],config['seed'],config['ticks'])


def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=make_model(checkpoint['config']);model.load_state_dict(checkpoint['model']);model.eval()
    return model,checkpoint
