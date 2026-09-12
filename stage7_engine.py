"""Autoregressive text inference; optional forced prefixes are for diagnostics only."""
import torch
from stage6_engine import generate as basic_generate
from stage6_tokens import PAD, BOS, SEP, EOS, encode, decode, prompt_ids


@torch.no_grad()
def generate(model, prompts, max_new_tokens=512, batch_size=32, prefixes=None):
    if prefixes is None:
        return basic_generate(model, prompts, max_new_tokens, batch_size)
    if len(prefixes) != len(prompts): raise ValueError('One prefix per prompt required.')
    results = []
    for start in range(0, len(prompts), batch_size):
        p = prompts[start:start + batch_size]; forced = prefixes[start:start + batch_size]
        beginnings = [prompt_ids(x) for x in p]
        encoded = [a + encode(b) for a, b in zip(beginnings, forced)]
        width = max(map(len, encoded)); ids = torch.full((len(p), width), PAD, dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool); pm = valid.clone()
        for i, row in enumerate(encoded):
            offset = width - len(row); ids[i, offset:] = torch.tensor(row); valid[i, offset:] = True
            pm[i, offset + 1:offset + len(beginnings[i]) - 1] = True
        probabilities, cache = model(ids, valid, pm); probabilities = probabilities[:, -1]
        tokens = [[] for _ in p]; done = torch.zeros(len(p), dtype=torch.bool); halted = [False] * len(p)
        for _ in range(max_new_tokens):
            probabilities[:, [PAD, BOS, SEP]] = 0
            chosen = probabilities.argmax(-1)
            for i, token in enumerate(chosen.tolist()):
                if done[i]: continue
                if token == EOS: done[i] = True; halted[i] = True
                else: tokens[i].append(token)
            if bool(done.all()): break
            next_ids = torch.where(done, torch.zeros_like(chosen), chosen)[:, None]
            probabilities, cache = model(next_ids, (~done)[:, None], pm, cache)
            probabilities = probabilities[:, -1]
        results.extend(dict(text=decode(row), halted=stop, generated_tokens=len(row) + int(stop))
                       for row, stop in zip(tokens, halted))
    return results
