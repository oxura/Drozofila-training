"""Train/validation-only episodic learning. No task identity or oracle is input."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from stage5_tasks import ROOT, DATA, load
from stage5_memory import encode
from stage5_engine import predict, pooled_batch
from stage5_model import make_model


def metrics(rows, probabilities):
    y = np.asarray([r['query_y'] for r in rows]); p = np.asarray(probabilities)
    correct = (p >= .5) == y
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(query_accuracy=float(correct.mean()), episode_exact=float(correct.all(1).mean()),
                bce=float(-(y * np.log(clipped) + (1-y) * np.log(1-clipped)).mean()),
                count=len(rows), queries=int(y.size), label_one_fraction=float(y.mean()))


def train(config, directory, resume=None):
    torch.set_num_threads(config['threads']); torch.use_deterministic_algorithms(True)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'history.jsonl').exists() and resume is None:
        raise FileExistsError(directory)
    rows, val = load('train'), load('val')
    manifest = json.loads((DATA / 'manifest.json').read_text())
    cache = None
    if config['mode'] == 'memory':
        encoded = [encode(r['support_x'], r['support_y'], r['query_x']) for r in rows]
        cache = (torch.tensor(np.stack([e['features'] for e in encoded])),
                 torch.tensor(np.stack([e['values'] for e in encoded])))
        del encoded
    labels = torch.tensor([r['query_y'] for r in rows], dtype=torch.float32)
    model = make_model(config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['lr'], weight_decay=1e-4)
    rng = np.random.default_rng(config['seed'] + 50505); start = 1; best = (-1, float('-inf')); examples = 0
    if resume:
        old = torch.load(resume, map_location='cpu', weights_only=True)
        assert old['dataset_hashes'] == manifest['hashes']
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        rng.bit_generator.state = old['numpy_rng']; torch.set_rng_state(old['torch_rng'])
        start = old['step'] + 1; best = tuple(old['best_score']); examples = old['training_episodes']
        if start > config['steps']:
            raise ValueError('New total must exceed checkpoint step')
    losses = []; max_grad = 0; t0 = time.perf_counter()
    for step in range(start, config['steps'] + 1):
        lr = config['lr'] * (.15 + .85 * (1 + math.cos(math.pi * (step-1) / max(1, config['steps']-1))) / 2)
        for group in optimizer.param_groups: group['lr'] = lr
        ids = rng.integers(len(rows), size=config['batch_size'])
        model.train(); optimizer.zero_grad(set_to_none=True)
        if cache is not None:
            probs = model(cache[0][ids], cache[1][ids])
        else:
            features, mask = pooled_batch([rows[i] for i in ids])
            probs = model(features, support_mask=mask)
        loss = F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), labels[ids])
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
        loss.backward()
        if hasattr(model, 'core') and model.core.edge_log_gain.grad is not None:
            max_grad = max(max_grad, float(model.core.edge_log_gain.grad.abs().max()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step(); losses.append(float(loss.detach())); examples += len(ids)
        if step == 1 or step % config['eval_every'] == 0 or step == config['steps']:
            model.eval(); probabilities = predict(model, val); scores = metrics(val, probabilities)
            score = (scores['query_accuracy'], -scores['bce']); improved = score > best; best = max(best, score)
            entry = dict(step=step, loss=float(np.mean(losses)), validation=scores, seconds=time.perf_counter()-t0,
                         training_episodes=examples, learning_rate=lr)
            losses = []
            with (directory / 'history.jsonl').open('a') as f: f.write(json.dumps(entry) + '\n')
            saved = dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                         best_score=list(best), numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                         dataset_hashes=manifest['hashes'], training_episodes=examples)
            torch.save(saved, directory / 'last.pt')
            if improved: torch.save(saved, directory / 'best.pt')
            print(json.dumps(dict(run=directory.name, **entry)), flush=True)
    saved = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model']); model.eval(); probabilities = predict(model, val)
    result = dict(config=config, validation=metrics(val, probabilities), best_step=saved['step'],
                  seconds=time.perf_counter()-t0, parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  max_edge_gradient=max_grad, dataset_hashes=manifest['hashes'], final_test_read=False)
    (directory / 'validation.json').write_text(json.dumps(result, indent=2))
    (directory / 'validation_predictions.jsonl').write_text(''.join(json.dumps(dict(id=r['id'], probabilities=p))+'\n' for r,p in zip(val, probabilities)))
    print('FINISHED ' + json.dumps(dict(run=directory.name, **result)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['memory', 'pooled'], default='memory')
    p.add_argument('--conditions', default='fly'); p.add_argument('--seeds', default='0')
    p.add_argument('--steps', type=int, default=1800); p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--ticks', type=int, default=3); p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--threads', type=int, default=2); p.add_argument('--eval-every', type=int, default=300)
    p.add_argument('--output', default='results/stage5/development'); p.add_argument('--resume')
    args = p.parse_args()
    if args.resume:
        path = Path(args.resume).resolve(); old = torch.load(path, map_location='cpu', weights_only=True)
        config = dict(old['config'], steps=args.steps, threads=args.threads, eval_every=args.eval_every)
        train(config, path.parent, path); return
    for condition in args.conditions.split(','):
        if condition not in ['fly', 'rewired', 'frozen', 'no_edges', 'mlp']: raise ValueError(condition)
        for seed in map(int, args.seeds.split(',')):
            config = dict(mode=args.mode, condition=condition, seed=seed, steps=args.steps, batch_size=args.batch_size,
                          ticks=args.ticks, lr=args.lr, threads=args.threads, eval_every=args.eval_every)
            train(config, ROOT / args.output / f'{args.mode}_{condition}_seed{seed}')


if __name__ == '__main__':
    main()
