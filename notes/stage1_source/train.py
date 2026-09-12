"""Run a small, reproducible pilot. Evaluation uses no teacher-forced answers."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import ConnectomeLM
from tasks import save_tasks, encode_rows
from evaluate import evaluate

ROOT = Path(__file__).resolve().parent


def batch(encoded, indices, device):
    selected = [encoded[int(i)] for i in indices]
    length = max(len(x) for x, _ in selected)
    x = torch.zeros((len(selected), length), dtype=torch.long, device=device)
    y = torch.full_like(x, -100)
    for i, (inputs, labels) in enumerate(selected):
        x[i, :len(inputs)] = torch.tensor(inputs, device=device)
        y[i, :len(labels)] = torch.tensor(labels, device=device)
    return x, y


def run(args, records, vocab, condition, cell, seed, resume=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 42)
    graph_path = ROOT / f'data/graph_{args.nodes}.npz'
    graph = dict(np.load(graph_path))
    model = ConnectomeLM(graph, len(vocab), condition, cell, seed).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=0.0001)
    run_dir = ROOT / args.output / f'{cell}_{condition}_seed{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / 'history.jsonl'
    config = dict(nodes=args.nodes, seed=seed, cell=cell, condition=condition,
                  lr=args.lr, batch_size=args.batch_size, steps_per_epoch=args.steps_per_epoch,
                  epochs=args.epochs, device=args.device, threads=args.threads)
    best, start_epoch = -1.0, 1
    if resume is not None:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        rng.bit_generator.state = resume['numpy_rng']
        torch.set_rng_state(resume['torch_rng'])
        best, start_epoch = resume['best_val'], resume['epoch'] + 1
    elif history_path.exists():
        raise FileExistsError(f'{run_dir} already exists; use a new --output or --resume')
    encoded = encode_rows(records['train'], vocab)
    by_task = defaultdict(list)
    for i, row in enumerate(records['train']):
        by_task[row['task']].append(i)
    task_indices = list(by_task.values())
    t0 = time.perf_counter()
    initial_metrics, _ = evaluate(model, records['val'], vocab)
    max_edge_grad = 0.0
    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for step in range(args.steps_per_epoch):
            # Equal task probability; sampling with replacement is intentional.
            indices = [rng.choice(task_indices[k % len(task_indices)]) for k in range(args.batch_size)]
            rng.shuffle(indices)
            x, y = batch(encoded, indices, args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            token_loss = F.cross_entropy(logits.transpose(1, 2), y, reduction='none', ignore_index=-100)
            loss = (token_loss.sum(1) / (y != -100).sum(1)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            if model.edge_log_gain.grad is not None:
                max_edge_grad = max(max_edge_grad, float(model.edge_log_gain.grad.abs().max()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_metrics, _ = evaluate(model, records['val'], vocab)
        score = val_metrics['macro_exact']
        improved = score > best
        if improved:
            best = score
        entry = dict(epoch=epoch, training_loss=float(np.mean(losses)), val=val_metrics,
                     elapsed_seconds=time.perf_counter() - t0)
        history.append(entry)
        with history_path.open('a') as f:
            f.write(json.dumps(entry) + '\n')
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          epoch=epoch, best_val=best, config=config, vocab=vocab,
                          numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state())
        torch.save(checkpoint, run_dir / 'last.pt')
        if improved:
            torch.save(checkpoint, run_dir / 'best.pt')
        if epoch == start_epoch or epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps(dict(run=run_dir.name, **entry)), flush=True)
    # The untouched test partition is evaluated once, after validation selection.
    selected = torch.load(run_dir / 'best.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(selected['model'])
    final = dict(config=config, initial_val=initial_metrics, best_epoch=selected['epoch'],
                 train_seconds=time.perf_counter() - t0, edge_count=model.pre.numel(),
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 trainable_edge_parameters=model.edge_log_gain.numel() if model.edge_log_gain.requires_grad else 0,
                 max_observed_edge_gradient=max_edge_grad,
                 selected_mean_abs_edge_log_change=float(model.edge_log_gain.detach().abs().mean()),
                 rewiring_swaps=model.rewiring_swaps)
    for split in ['val', 'test', 'phrasing']:
        metrics, examples = evaluate(model, records[split], vocab)
        final[split] = metrics
        (run_dir / f'{split}_predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in examples))
    # Descriptive post-training ablation, not a separately trained baseline.
    model.disable_edges = True
    final['test_without_recurrent_edges'], _ = evaluate(model, records['test'], vocab)
    (run_dir / 'metrics.json').write_text(json.dumps(final, indent=2))
    print('FINISHED ' + json.dumps(final), flush=True)
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nodes', type=int, default=256)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--steps-per-epoch', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=63)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--conditions', default='fly,rewired,frozen')
    p.add_argument('--cells', default='rate')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default='results/main')
    p.add_argument('--resume')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    records, vocab = save_tasks(ROOT / 'data/tasks')
    if args.resume:
        resume = torch.load(args.resume, map_location=args.device, weights_only=True)
        cfg = resume['config']
        for key in ['nodes', 'lr', 'batch_size', 'steps_per_epoch']:
            setattr(args, key, cfg[key])
        args.output = str(Path(args.resume).resolve().parent.parent.relative_to(ROOT))
        run(args, records, vocab, cfg['condition'], cfg['cell'], cfg['seed'], resume)
    else:
        for cell in args.cells.split(','):
            if cell not in ['rate', 'lif']:
                raise ValueError(cell)
            for seed in map(int, args.seeds.split(',')):
                for condition in args.conditions.split(','):
                    if condition not in ['fly', 'rewired', 'frozen']:
                        raise ValueError(condition)
                    run(args, records, vocab, condition, cell, seed)


if __name__ == '__main__':
    main()
