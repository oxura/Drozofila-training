"""Inference consumes demonstrations and questions only; no rule/oracle import."""
import numpy as np
import torch
from stage5_memory import encode


def pooled_batch(rows):
    width = max(len(r['support_x']) for r in rows)
    sx = np.zeros((len(rows), width, 9), dtype=np.float32)
    mask = np.zeros((len(rows), width), dtype=np.float32)
    for i, row in enumerate(rows):
        n = len(row['support_x'])
        sx[i, :n, :8] = row['support_x']; sx[i, :n, 8] = row['support_y']; mask[i, :n] = 1
    qx = np.asarray([r['query_x'] for r in rows], dtype=np.float32)
    return (torch.from_numpy(sx), torch.from_numpy(qx)), torch.from_numpy(mask)


@torch.no_grad()
def predict(model, rows, batch_size=16):
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        if model.mode == 'memory':
            encodings = [encode(r['support_x'], r['support_y'], r['query_x']) for r in batch]
            features = torch.tensor(np.stack([e['features'] for e in encodings]))
            values = torch.tensor(np.stack([e['values'] for e in encodings]))
            probs = model(features, values)
        else:
            features, mask = pooled_batch(batch)
            probs = model(features, support_mask=mask)
        output.extend(probs.tolist())
    return output
