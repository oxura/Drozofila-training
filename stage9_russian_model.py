"""Stage7 weights with a larger input vocabulary, unchanged numeric output.

Only neural attention/copying operates at inference. There is no Russian
semantic parser, program interpreter, dictionary of variable values or oracle.
"""
import math
from pathlib import Path
import torch
from torch import nn
from stage7_model import MemoryReasoner
from stage6_tokens import VOCAB, BOS, SEP, PAD

ROOT = Path(__file__).resolve().parent


def source_path(condition, seed):
    name = 'history' if condition == 'mlp' else condition
    return ROOT / f'results/stage7/confirmation/{name}_seed{seed}/best.pt'


class RussianReasoner(MemoryReasoner):
    def __init__(self, condition, seed, input_size):
        super().__init__(condition=condition, seed=seed, mode='history')
        self.embedding = nn.Embedding(input_size, 96, padding_idx=0)

    def forward(self, ids, valid, prompt_mask, cache=None):
        current_valid = valid
        if cache is None:
            positions = (valid.long().cumsum(1) - 1).clamp_min(0)
            t = ids.shape[1]; causal = torch.ones(t, t, dtype=torch.bool, device=ids.device).tril()
            allowed = causal[None, None] & valid[:, None, None, :]
            allowed |= (~valid)[:, None, :, None] & torch.eye(t, dtype=torch.bool, device=ids.device)[None, None]
            past = [None] * len(self.blocks)
        else:
            positions = cache['valid'].sum(1, keepdim=True).long()
            valid = torch.cat([cache['valid'], valid], 1)
            allowed = valid[:, None, None, :]; past = cache['layers']
        h = self.embedding(ids); layers = []
        for block, old in zip(self.blocks, past):
            h, new = block(h, positions, allowed, old); layers.append(new)
        h = self.norm(h)
        representable = (ids >= 4) & (ids < len(VOCAB))
        current_mask = current_valid & representable
        current_copy_ids = ids.masked_fill(~representable, PAD)
        if cache is None:
            keys = self.copy_k(h); copy_ids = current_copy_ids; mask = current_mask
            copy_allowed = mask[:, None, :] & causal[None]
        else:
            keys = torch.cat([cache['copy_keys'], self.copy_k(h)], 1)
            copy_ids = torch.cat([cache['copy_ids'], current_copy_ids], 1)
            mask = torch.cat([cache['copy_mask'], current_mask], 1)
            copy_allowed = mask[:, None, :]
        raw = self.output(h).softmax(-1)
        scores = torch.einsum('btd,bsd->bts', self.copy_q(h), keys) / math.sqrt(32)
        weights = scores.masked_fill(~copy_allowed, -1e4).softmax(-1) * copy_allowed
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        copied = torch.zeros_like(raw).scatter_add(2, copy_ids[:, None, :].expand(-1, h.shape[1], -1), weights)
        gate = self.copy_gate(h).sigmoid()
        gate = torch.where(copy_allowed.any(-1, keepdim=True), gate, torch.ones_like(gate))
        probabilities = (gate * raw + (1 - gate) * copied).clamp_min(1e-9)
        return probabilities, dict(layers=layers, valid=valid, copy_keys=keys,
            copy_ids=copy_ids, copy_mask=mask, memory=None)


def initialize(config, vocabulary):
    model = RussianReasoner(config['condition'], config['seed'], len(vocabulary['tokens']))
    source = torch.load(source_path(config['condition'], config['seed']), map_location='cpu', weights_only=True)
    state = dict(source['model'])
    state['embedding.weight'] = torch.cat([state['embedding.weight'], model.embedding.weight.detach()[len(VOCAB):]], 0)
    model.load_state_dict(state)
    return model


def load_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = RussianReasoner(saved['config']['condition'], saved['config']['seed'], len(saved['vocabulary']['tokens']))
    model.load_state_dict(saved['model']); model.eval()
    return model, saved
