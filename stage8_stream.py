"""Causal, bounded-state memory prototype. No teacher or task imports.

The frontend exposes literal WRITE/QUERY events and character/value features.
It does not resolve variables. Slots have learned addresses and gates; no slot
is assigned to a variable by the program. This is a deliberately easier input
interface than stage7 text generation, and is not a general language model.
"""
import re
import torch
from torch import nn

FEATURES = 2 + 2 * 27 + 10


def encode_prompt(prompt):
    if not isinstance(prompt, str) or not prompt.startswith('код '):
        raise ValueError('Expected the formal prefix: код ')
    pieces = prompt[4:].split(';')
    if len(pieces) < 2 or re.fullmatch(r'\?[a-z]{1,2}', pieces[-1]) is None:
        raise ValueError('Expected literal assignments followed by ?name.')
    events = []
    for part in pieces[:-1]:
        match = re.fullmatch(r'([a-z]{1,2})=([0-9])', part)
        if match is None: raise ValueError('Only one-digit literal writes are supported.')
        events.append((0, match[1], int(match[2])))
    events.append((1, pieces[-1][1:], None))
    encoded = torch.zeros(len(events), FEATURES)
    for i, (operation, name, value) in enumerate(events):
        encoded[i, operation] = 1
        for j in range(2):
            character = ord(name[j]) - ord('a') + 1 if j < len(name) else 0
            encoded[i, 2 + 27 * j + character] = 1
        if value is not None: encoded[i, -10 + value] = 1
    return encoded


class Slots(nn.Module):
    def __init__(self, width, cells=8, value_width=16):
        super().__init__()
        self.address_write = nn.Linear(width, cells)
        self.gate_write = nn.Linear(width, 1)
        self.value = nn.Linear(width, value_width)
        self.address_read = nn.Linear(width, cells)
        self.output = nn.Linear(value_width, width)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.cells, self.value_width = cells, value_width

    def forward(self, hidden, memory=None):
        weight = self.gate_write(hidden).sigmoid() * self.address_write(hidden).softmax(-1)
        a = (1 - weight)[..., None]
        b = weight[..., None] * self.value(hidden).tanh()[:, :, None, :]
        offset = 1
        while offset < hidden.shape[1]:
            old_a, old_b = a, b
            a = torch.cat([old_a[:, :offset], old_a[:, offset:] * old_a[:, :-offset]], 1)
            b = torch.cat([old_b[:, :offset], old_b[:, offset:] + old_a[:, offset:] * old_b[:, :-offset]], 1)
            offset *= 2
        states = b if memory is None else b + a * memory[:, None]
        read = (self.address_read(hidden).softmax(-1)[..., None] * states).sum(2)
        return self.output(read), states[:, -1]


class StreamMemory(nn.Module):
    def __init__(self, mode='gru', width=64):
        super().__init__()
        if mode not in ['gru', 'slots']: raise ValueError(mode)
        self.mode, self.width = mode, width
        self.encoder = nn.Sequential(nn.Linear(FEATURES, width), nn.Tanh())
        self.recurrent = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, 10)
        if mode == 'slots': self.memory = Slots(width)

    def forward(self, events, state=None, disable_memory=False):
        hidden, recurrent_state = self.recurrent(self.encoder(events), None if state is None else state[0])
        memory_state = None
        if self.mode == 'slots':
            delta, memory_state = self.memory(hidden, None if state is None else state[1])
            if not disable_memory: hidden = hidden + delta
        return self.head(hidden), (recurrent_state, memory_state)


def load_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = StreamMemory(saved['config']['mode'], saved['config']['width'])
    model.load_state_dict(saved['model']); model.eval()
    return model, saved


@torch.no_grad()
def answer_stream(model, prompt, disable_memory=False):
    state = None
    # Only recurrent state and optional cells survive between events.
    for event in encode_prompt(prompt):
        logits, state = model(event[None, None], state, disable_memory=disable_memory)
    return int(logits[0, -1].argmax())
