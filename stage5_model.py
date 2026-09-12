"""Episodic subset scorer or pooled-support baseline, with matched graph controls."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import ConnectomeLM
from stage5_memory import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent


class RuleLearner(nn.Module):
    def __init__(self, condition='fly', mode='memory', seed=0, ticks=3):
        super().__init__(); torch.manual_seed(seed)
        self.condition, self.mode, self.ticks = condition, mode, ticks
        self.disable_edges = False
        input_size = len(FEATURE_NAMES) if mode == 'memory' else 40
        if mode == 'pooled':
            self.support_encoder = nn.Sequential(nn.Linear(9, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        if condition == 'mlp':
            # Near-matched trainable count to the corresponding graph adapter.
            width = 96 if mode == 'memory' else 116
            self.network = nn.Sequential(nn.Linear(input_size, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        else:
            graph = dict(np.load(ROOT / 'data/graph_256.npz'))
            self.core = ConnectomeLM(graph, 1, 'frozen' if condition == 'no_edges' else condition, 'rate', seed)
            del self.core.embedding; del self.core.decoder
            self.input = nn.Linear(input_size, 256); self.output = nn.Linear(256, 1)
            self.gate_logit = nn.Parameter(torch.zeros(256))

    def score(self, features):
        if self.condition == 'mlp':
            return self.network(features).squeeze(-1)
        self.core.disable_edges = self.disable_edges or self.condition == 'no_edges'
        matrix = self.core.matrix()
        drive = self.input(features); state = torch.zeros_like(drive); gate = self.gate_logit.sigmoid()
        for _ in range(self.ticks):
            state = (1 - gate) * state + gate * torch.tanh(drive + F.linear(state, matrix))
        return self.output(state).squeeze(-1)

    def forward(self, features, values=None, support_mask=None):
        if self.mode == 'memory':
            weights = torch.softmax(self.score(features), dim=-1)
            return torch.einsum('bh,bhq->bq', weights, values)
        # features=(padded support x/y, query x), strictly eight input bits.
        support, query = features
        embedded = self.support_encoder(support)
        context = (embedded * support_mask[..., None]).sum(1) / support_mask.sum(1, keepdim=True)
        combined = torch.cat([context[:, None, :].expand(-1, query.shape[1], -1), query], dim=-1)
        return torch.sigmoid(self.score(combined))


def make_model(config):
    return RuleLearner(config['condition'], config['mode'], config['seed'], config['ticks'])


def load_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = make_model(saved['config']); model.load_state_dict(saved['model']); model.eval()
    return model, saved
