"""Trainable connectome-constrained sequence model, NOT a brain emulation.

Only existing directed edges have parameters. Signs stay fixed. Input/output
adapters are artificial, dense, and trainable. Rate and experimental LIF cells
share the same graph; neither reproduces all original Brian2 dynamics.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def rewire(pre, post, seed, swaps_per_edge=10):
    """Directed double-edge swaps preserve every in/out degree, without duplicates."""
    rng = np.random.default_rng(seed)
    pre, post = pre.copy(), post.copy()
    edges = set(zip(pre.tolist(), post.tolist()))
    accepted = 0
    target = swaps_per_edge * len(pre)
    for _ in range(target * 20):
        i, j = rng.integers(len(pre), size=2)
        a, b, c, d = int(pre[i]), int(post[i]), int(pre[j]), int(post[j])
        if a == c or b == d or a == d or c == b or (a, d) in edges or (c, b) in edges:
            continue
        edges.remove((a, b)); edges.remove((c, d))
        edges.add((a, d)); edges.add((c, b))
        post[i], post[j] = d, b
        accepted += 1
        if accepted == target:
            break
    assert accepted == target, 'Insufficient valid rewiring swaps'
    return pre, post, accepted


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage > 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad):
        voltage, = ctx.saved_tensors
        return grad / (1 + 5 * voltage.abs()).square()


class ConnectomeLM(nn.Module):
    def __init__(self, graph, vocab_size, condition='fly', cell='rate', seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n = len(graph['root_ids'])
        self.cell = cell
        self.condition = condition
        self.disable_edges = False
        pre, post = graph['pre'].copy(), graph['post'].copy()
        self.rewiring_swaps = 0
        if condition == 'rewired':
            pre, post, self.rewiring_swaps = rewire(pre, post, seed + 9000)
        raw = graph['weight'].astype(np.float32)
        magnitudes = np.log1p(np.abs(raw))
        norm = np.sqrt(np.bincount(post, weights=magnitudes ** 2, minlength=self.n))
        initial = 0.65 * magnitudes / np.maximum(norm[post], 1e-6)
        self.register_buffer('pre', torch.tensor(pre, dtype=torch.long))
        self.register_buffer('post', torch.tensor(post, dtype=torch.long))
        self.register_buffer('sign', torch.tensor(np.sign(raw)))
        self.register_buffer('initial_magnitude', torch.tensor(initial, dtype=torch.float32))
        self.edge_log_gain = nn.Parameter(torch.zeros(len(pre)), requires_grad=condition != 'frozen')
        self.embedding = nn.Embedding(vocab_size, self.n, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.20)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        self.decoder = nn.Linear(self.n, vocab_size)

    def matrix(self):
        values = self.sign * self.initial_magnitude * self.edge_log_gain.clamp(-4, 4).exp()
        if self.disable_edges:
            values = values * 0
        # Dense multiplication is faster on CPU for this small induced subgraph.
        # Only E edge gains are parameters; nonexistent edges cannot learn.
        return values.new_zeros((self.n, self.n)).index_put((self.post, self.pre), values)

    def initial_state(self, batch_size, device):
        z = torch.zeros(batch_size, self.n, device=device, dtype=self.embedding.weight.dtype)
        return z, z.clone()

    def step(self, ids, state, matrix):
        h, spikes = state
        drive = self.embedding(ids)
        if self.cell == 'rate':
            h = 0.25 * h + 0.75 * torch.tanh(drive + F.linear(h, matrix))
            readout = h
        else:
            voltage = 0.85 * h + drive + F.linear(spikes, matrix)
            spikes = SurrogateSpike.apply(voltage - 1.0)
            h = voltage - spikes.detach()
            # Membrane readout is an artificial interface, disclosed in README.
            readout = h
        return self.decoder(readout), (h, spikes)

    def forward(self, ids):
        matrix = self.matrix()
        state = self.initial_state(ids.shape[0], ids.device)
        outputs = []
        for t in range(ids.shape[1]):
            logits, state = self.step(ids[:, t], state, matrix)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def generate(self, prompt_ids, max_tokens=24):
        """Batch has equally long unpadded prompts ending in <sep>."""
        self.eval()
        matrix = self.matrix()
        state = self.initial_state(prompt_ids.shape[0], prompt_ids.device)
        for t in range(prompt_ids.shape[1]):
            logits, state = self.step(prompt_ids[:, t], state, matrix)
        generated = []
        done = torch.zeros(prompt_ids.shape[0], dtype=torch.bool, device=prompt_ids.device)
        for _ in range(max_tokens):
            # PAD/BOS/SEP/UNK are not output symbols.
            logits[:, [0, 1, 2, 4]] = -torch.inf
            token = logits.argmax(-1)
            token = torch.where(done, torch.full_like(token, 3), token)
            generated.append(token)
            done |= token == 3
            if bool(done.all()):
                break
            logits, state = self.step(token, state, matrix)
        return torch.stack(generated, dim=1)
