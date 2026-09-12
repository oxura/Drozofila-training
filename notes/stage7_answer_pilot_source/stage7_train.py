"""Stage7 continuation; exclusively loads train/replay/validation."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from stage6_train import batch_tensors
from stage6_tokens import prompt_ids, encode, EOS, IDS
from stage7_tasks import ROOT, DATA, load, completion, digest, write_json, write_rows
from stage7_model import initialize, make_model
from stage7_engine import generate


def prepare(row):
    prefix = prompt_ids(row['prompt'])
    return prefix + encode(completion(row)) + [EOS], len(prefix)


def scores(rows, outputs):
    records = []
    for row, out in zip(rows, outputs):
        answer = out['text'].rsplit('|', 1)[-1]
        expected = completion(row)
        record = dict(id=row['id'], family=row['family'], prompt=row['prompt'], actual=row['answer'],
                      expected_completion=expected, answer=answer,
                      answer_correct=bool(out['halted'] and answer == row['answer']),
                      completion_correct=bool(out['halted'] and out['text'] == expected), **out)
        for key in ['query_last', 'query_initial', 'query_gap', 'overwrites', 'paired_id', 'relevant_change', 'difficulty']:
            if key in row: record[key] = row[key]
        records.append(record)
    def metric(items):
        return dict(count=len(items), answer_accuracy=float(np.mean([x['answer_correct'] for x in items])),
                    completion_exact=float(np.mean([x['completion_correct'] for x in items])),
                    halted_fraction=float(np.mean([x['halted'] for x in items])))
    groups = defaultdict(list)
    for r in records: groups[r['family']].append(r)
    by_family = {key: metric(value) for key, value in groups.items()}
    result = dict(**metric(records), by_family=by_family,
                  macro_answer_accuracy=float(np.mean([v['answer_accuracy'] for v in by_family.values()])),
                  macro_completion_exact=float(np.mean([v['completion_exact'] for v in by_family.values()])))
    result['query_position'] = {}
    for family in ['memory', 'code']:
        for flag in [False, True]:
            subset = [r for r in records if r['family'] == family and r.get('query_last') == flag]
            if subset: result['query_position'][family + ('_last' if flag else '_earlier')] = metric(subset)
    return result, records


@torch.no_grad()
def validate(model, rows):
    model.eval(); outputs = generate(model, [r['prompt'] for r in rows], max_new_tokens=192)
    result, records = scores(rows, outputs); total = 0.; tokens = 0
    for start in range(0, len(rows), 32):
        ids, labels, valid, pm = batch_tensors([prepare(r) for r in rows[start:start + 32]])
        probabilities, _ = model(ids, valid, pm)
        total += float(F.nll_loss(probabilities.log().transpose(1, 2), labels, ignore_index=-100, reduction='sum'))
        tokens += int((labels != -100).sum())
    result['token_nll'] = total / tokens
    return result, records


def train(config, out, resume=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'validation.json').exists(): raise FileExistsError('Completed run already exists.')
    if (out / 'history.jsonl').exists() and not resume: raise FileExistsError('Use --resume for interrupted runs.')
    torch.set_num_threads(config['threads']); torch.use_deterministic_algorithms(True)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for split in ['train', 'val', 'replay']:
        assert digest(DATA / f'{split}.jsonl') == manifest['hashes'][split]
    rows, replay, val = load('train'), load('replay'), load('val')
    nnew = len(rows); rows += replay
    prepared = [prepare(r) for r in rows]
    pools = {}
    families = ['memory', 'code', 'sum', 'logic']
    for family in families:
        for old in [False, True]:
            eligible = np.array([i for i, r in enumerate(rows) if r['family'] == family and (i >= nnew) == old])
            if not len(eligible): continue
            buckets = defaultdict(list)
            for i in eligible: buckets[(len(prepared[i][0]) + 31) // 32].append(i)
            pools[(family, old)] = (eligible, {k: np.array(v) for k, v in buckets.items()})
    model, source = initialize(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.01)
    rng = np.random.default_rng(config['seed'] + 70707)
    best = (-1., -1., float('-inf')); start_step = 1; target_tokens = 0; input_tokens = 0; examples = 0
    counts = Counter(); replay_counts = Counter(); max_grads = defaultdict(float)
    t0 = time.perf_counter()

    def save_checkpoint(step):
        return dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                    best_score=list(best), numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    dataset_hashes=manifest['hashes'], source_checkpoint=str(source.relative_to(ROOT)),
                    source_sha256=digest(source), target_tokens=target_tokens, input_tokens=input_tokens,
                    training_examples=examples, family_counts=dict(counts), replay_counts=dict(replay_counts))

    if resume:
        saved = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        assert saved['config'] == config and saved['dataset_hashes'] == manifest['hashes']
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']; torch.set_rng_state(saved['torch_rng'])
        best = tuple(saved['best_score']); start_step = saved['step'] + 1
        target_tokens, input_tokens, examples = saved['target_tokens'], saved['input_tokens'], saved['training_examples']
        counts.update(saved['family_counts']); replay_counts.update(saved['replay_counts'])
    else:
        initial, records = validate(model, val)
        best = (initial['macro_answer_accuracy'], initial['macro_completion_exact'], -initial['token_nll'])
        write_json(out / 'initial_validation.json', initial); write_rows(out / 'initial_predictions.jsonl', records)
        torch.save(save_checkpoint(0), out / 'best.pt')
        write_json(out / 'config.json', config)
    weights = [.35, .35, .15, .15] if config['curriculum'] else [0., .7, .15, .15]
    losses = []; invocation_start = start_step
    for step in range(start_step, config['steps'] + 1):
        lr = config['lr'] * min(1, step / 50) * (.3 + .7 * (1 + math.cos(math.pi * (step - 1) / max(1, config['steps'] - 1))) / 2)
        for group in optimizer.param_groups: group['lr'] = lr
        family = families[int(rng.choice(4, p=weights))]
        old = family != 'memory' and rng.random() < (.25 if family == 'code' else .5)
        eligible, buckets = pools[(family, old)]
        first = int(rng.choice(eligible)); bucket = (len(prepared[first][0]) + 31) // 32
        chosen = rng.choice(buckets[bucket], config['batch_size'])
        ids, labels, valid, pm = batch_tensors([prepared[i] for i in chosen])
        model.train(); optimizer.zero_grad(set_to_none=True)
        probabilities, _ = model(ids, valid, pm)
        per_token = F.nll_loss(probabilities.log().transpose(1, 2), labels, ignore_index=-100, reduction='none')
        per_example = per_token.sum(1) / (labels != -100).sum(1)
        answer_weight = config.get('answer_weight', 0.)
        if answer_weight:
            answer_mask = (ids == IDS['|']).long().cumsum(1).bool() & (labels != -100)
            answer_loss = (per_token * answer_mask).sum(1) / answer_mask.sum(1).clamp_min(1)
            per_example = (1 - answer_weight) * per_example + answer_weight * answer_loss
        loss = per_example.mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        loss.backward()
        if step <= 3 or step % 100 == 0:
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and ('edge_log_gain' in name or name.startswith('memory.')):
                    max_grads[name] = max(max_grads[name], float(parameter.grad.abs().max()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        target_tokens += int((labels != -100).sum()); input_tokens += int(valid.sum()); examples += len(chosen)
        counts[family] += len(chosen)
        if old: replay_counts[family] += len(chosen)
        losses.append(float(loss.detach()))
        if step % 100 == 0:
            print(json.dumps(dict(run=out.name, step=step, loss=float(np.mean(losses[-100:])), seconds=time.perf_counter() - t0)), flush=True)
        if step % config['eval_every'] == 0 or step == config['steps']:
            metrics, records = validate(model, val)
            score = (metrics['macro_answer_accuracy'], metrics['macro_completion_exact'], -metrics['token_nll'])
            improved = score > best; best = max(score, best)
            checkpoint = save_checkpoint(step); torch.save(checkpoint, out / 'last.pt')
            if improved:
                torch.save(checkpoint, out / 'best.pt'); write_rows(out / 'validation_predictions.jsonl', records)
            entry = dict(step=step, validation=metrics, loss=float(np.mean(losses)), seconds=time.perf_counter() - t0,
                         learning_rate=lr, target_tokens=target_tokens, input_tokens=input_tokens, training_examples=examples)
            with (out / 'history.jsonl').open('a') as f: f.write(json.dumps(entry) + '\n')
            losses = []; print('VALIDATION ' + json.dumps(dict(run=out.name, **entry)), flush=True)
    saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    if saved['step'] == 0:
        (out / 'validation_predictions.jsonl').write_bytes((out / 'initial_predictions.jsonl').read_bytes())
    result = dict(config=config, best_step=saved['step'], validation_score=saved['best_score'],
                  seconds_this_invocation=time.perf_counter() - t0, start_step=invocation_start,
                  parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  source_checkpoint=str(source.relative_to(ROOT)), source_sha256=digest(source),
                  target_tokens=target_tokens, input_tokens=input_tokens, training_examples=examples,
                  family_counts=dict(counts), replay_counts=dict(replay_counts), max_component_gradients=dict(max_grads),
                  dataset_hashes=manifest['hashes'], final_test_read=False)
    write_json(out / 'validation.json', result)
    print('FINISHED ' + json.dumps(dict(run=out.name, **result)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); train(json.loads(Path(args.config).read_text()), ROOT / args.output, args.resume)
