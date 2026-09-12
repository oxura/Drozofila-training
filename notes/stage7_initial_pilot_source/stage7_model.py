"""Continuation of stage6: causal history copying and optional learned slot memory.

No program parser, arithmetic semantics, true registers or teacher in this module.
Slot addressing and write/read gates are learned from token loss.
"""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from stage6_model import Reasoner
from stage6_tokens import BOS, SEP, PAD


class SlotMemory(nn.Module):
    def __init__(self, d, slots=8, width=16):
        super().__init__()
        self.write_address = nn.Linear(d, slots)
        self.write_gate = nn.Linear(d, 1)
        self.value = nn.Linear(d, width)
        self.read_address = nn.Linear(d, slots)
        self.output = nn.Linear(width, d)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.slots, self.width = slots, width

    def forward(self, h, valid, state=None):
        gate = torch.sigmoid(self.write_gate(h)) * torch.softmax(self.write_address(h), -1)
        gate = gate * valid[..., None]
        a = (1 - gate)[..., None]
        b = gate[..., None] * torch.tanh(self.value(h))[:, :, None, :]
        # Parallel prefix scan of affine updates M_t = a_t M_{t-1} + b_t.
        # Every doubling uses only earlier positions; no future-token access.
        offset = 1
        while offset < h.shape[1]:
            old_a, old_b = a, b
            a = torch.cat([old_a[:, :offset], old_a[:, offset:] * old_a[:, :-offset]], 1)
            b = torch.cat([old_b[:, :offset], old_b[:, offset:] + old_a[:, offset:] * old_b[:, :-offset]], 1)
            offset *= 2
        memory = b if state is None else b + a * state[:, None]
        read = (torch.softmax(self.read_address(h), -1)[..., None] * memory).sum(2)
        return self.output(read), memory[:, -1]


class MemoryReasoner(Reasoner):
    def __init__(self, condition='mlp', seed=0, mode='prompt', wider=False):
        super().__init__(condition=condition, copy=True, seed=seed, d=96, layers=3, heads=4)
        self.mode = mode
        self.disable_history = False; self.disable_memory = False
        if wider:
            if condition != 'mlp': raise ValueError('Widening control is defined for ordinary FF blocks.')
            for block in self.blocks:
                block.ff = nn.Sequential(nn.Linear(96, 320), nn.GELU(), nn.Linear(320, 96))
        if mode == 'slots': self.memory = SlotMemory(96)

    def forward(self, ids, valid, prompt_mask, cache=None):
        current_valid = valid
        if cache is None:
            positions = (valid.long().cumsum(1) - 1).clamp_min(0)
            t = ids.shape[1]
            causal = torch.ones(t, t, dtype=torch.bool, device=ids.device).tril()
            allowed = causal[None, None] & valid[:, None, None, :]
            allowed |= (~valid)[:, None, :, None] & torch.eye(t, dtype=torch.bool, device=ids.device)[None, None]
            past = [None] * len(self.blocks)
        else:
            positions = cache['valid'].sum(1, keepdim=True).long()
            valid = torch.cat([cache['valid'], valid], 1)
            allowed = valid[:, None, None, :]
            past = cache['layers']
        h = self.embedding(ids); layers = []
        for block, old in zip(self.blocks, past):
            h, new = block(h, positions, allowed, old); layers.append(new)
        h = self.norm(h)
        history = self.mode in ('history', 'slots') and not self.disable_history
        if cache is None:
            copy_keys = self.copy_k(h); copy_ids = ids
            copy_mask = current_valid & (ids >= 4) if history else prompt_mask
            copy_allowed = copy_mask[:, None, :] & causal[None]
        elif history:
            copy_keys = torch.cat([cache['copy_keys'], self.copy_k(h)], 1)
            copy_ids = torch.cat([cache['copy_ids'], ids], 1)
            copy_mask = torch.cat([cache['copy_mask'], current_valid & (ids >= 4)], 1)
            copy_allowed = copy_mask[:, None, :]
        else:
            copy_keys, copy_ids, copy_mask = cache['copy_keys'], cache['copy_ids'], cache['copy_mask']
            copy_allowed = copy_mask[:, None, :]
        state = None
        if self.mode == 'slots':
            delta, state = self.memory(h, current_valid, None if cache is None else cache['memory'])
            if not self.disable_memory: h = h + delta
        raw = torch.softmax(self.output(h), -1)
        scores = torch.einsum('btd,bsd->bts', self.copy_q(h), copy_keys) / math.sqrt(32)
        weights = torch.softmax(scores.masked_fill(~copy_allowed, -1e4), -1)
        # All-masked early/padding rows copy nothing, instead of a uniform future leak.
        weights = weights * copy_allowed
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        copied = torch.zeros_like(raw).scatter_add(2, copy_ids[:, None, :].expand(-1, h.shape[1], -1), weights)
        gate = torch.sigmoid(self.copy_gate(h))
        gate = torch.where(copy_allowed.any(-1, keepdim=True), gate, torch.ones_like(gate))
        probabilities = (gate * raw + (1 - gate) * copied).clamp_min(1e-9)
        return probabilities, dict(layers=layers, valid=valid, copy_keys=copy_keys,
                                   copy_ids=copy_ids, copy_mask=copy_mask, memory=state)


def make_model(config):
    return MemoryReasoner(config['condition'], config['seed'], config['mode'], config.get('wider', False))


def source_checkpoint(condition, seed):
    return Path(__file__).resolve().parent / f'results/stage6/confirmation/resolved_copy_atomic_{condition}_seed{seed}/best.pt'


def initialize(config):
    model = make_model(config)
    path = source_checkpoint(config['condition'], config['seed'])
    old = torch.load(path, map_location='cpu', weights_only=True)
    state = dict(old['model'])
    if config.get('wider', False):
        # Duplicate eight hidden units and divide outgoing weights by multiplicity.
        # GELU and the rest of each block are unchanged, preserving its function.
        g = torch.Generator().manual_seed(config['seed'] + 712)
        for i in range(3):
            indices = torch.cat([torch.arange(312), torch.randperm(312, generator=g)[:8]])
            multiplicity = torch.bincount(indices, minlength=312)
            prefix = f'blocks.{i}.ff.'
            state[prefix + '0.weight'] = state[prefix + '0.weight'][indices]
            state[prefix + '0.bias'] = state[prefix + '0.bias'][indices]
            state[prefix + '2.weight'] = state[prefix + '2.weight'][:, indices] / multiplicity[indices]
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith('memory.') for k in missing), (missing, unexpected)
    return model, path


def load_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = make_model(saved['config']); model.load_state_dict(saved['model']); model.eval()
    return model, saved
