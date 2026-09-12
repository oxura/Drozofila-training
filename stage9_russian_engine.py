"""Free generation from text, without semantic/task/training imports."""
import torch
from stage9_russian_tokens import PAD, BOS, SEP, EOS, decode


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens=192, batch_size=32):
    model.eval(); results = []
    for start in range(0, len(prompts), batch_size):
        batch = [tokenizer.prompt(p) for p in prompts[start:start + batch_size]]
        width = max(map(len, batch)); ids = torch.full((len(batch), width), PAD, dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for i, row in enumerate(batch): ids[i, -len(row):] = torch.tensor(row); valid[i, -len(row):] = True
        pm = valid & (ids != BOS) & (ids != SEP)
        probabilities, cache = model(ids, valid, pm); probabilities = probabilities[:, -1]
        done = torch.zeros(len(batch), dtype=torch.bool); halted = [False] * len(batch); output = [[] for _ in batch]
        for _ in range(max_new_tokens):
            probabilities[:, [PAD, BOS, SEP]] = 0; chosen = probabilities.argmax(-1)
            for i, token in enumerate(chosen.tolist()):
                if done[i]: continue
                if token == EOS: done[i] = True; halted[i] = True
                else: output[i].append(token)
            if bool(done.all()): break
            next_ids = torch.where(done, torch.zeros_like(chosen), chosen)[:, None]
            probabilities, cache = model(next_ids, (~done)[:, None], pm, cache)
            probabilities = probabilities[:, -1]
        results.extend(dict(text=decode(row), halted=stop, generated_tokens=len(row) + int(stop)) for row, stop in zip(output, halted))
    return results
