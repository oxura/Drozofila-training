"""Continuation on Russian instructions; only train and development splits load here."""
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
from stage9_russian_tokens import Tokenizer, VOCAB
from stage9_russian_model import initialize, source_path
from stage9_russian_engine import generate

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/stage9_russian'


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def load(split): return [json.loads(x) for x in (DATA / (split + '.jsonl')).read_text().splitlines()]


def target(row, style):
    if 'target' in row: return row['target']
    return row['answer'] if style == 'direct' else row['trace'] + '|' + row['answer']


def scores(rows, outputs, style):
    records = []
    for row, out in zip(rows, outputs):
        answer = out['text'].rsplit('|', 1)[-1]
        record = dict(id=row['id'], expected=row['answer'], expected_completion=target(row, style),
            answer=answer, answer_correct=bool(out['halted'] and answer == row['answer']),
            completion_correct=bool(out['halted'] and out['text'] == target(row, style)), **out)
        for k in ['family', 'role', 'length', 'pair_id', 'paired_id', 'side', 'has_copy_chain']:
            if k in row: record[k] = row[k]
        records.append(record)
    def metric(items):
        return dict(count=len(items), answer_accuracy=float(np.mean([r['answer_correct'] for r in items])),
            completion_exact=float(np.mean([r['completion_correct'] for r in items])),
            halted_fraction=float(np.mean([r['halted'] for r in items])))
    result = metric(records)
    for key in ['family', 'role', 'length']:
        groups = defaultdict(list)
        for r in records:
            if key in r: groups[str(r[key])].append(r)
        if groups: result['by_' + key] = {k: metric(v) for k, v in groups.items()}
    pairs = defaultdict(list)
    for r in records:
        if 'pair_id' in r: pairs[r['pair_id']].append(r)
    if pairs:
        assert all(len(v) == 2 for v in pairs.values())
        result['pairs'] = dict(count=len(pairs), both_correct=float(np.mean([all(x['answer_correct'] for x in v) for v in pairs.values()])),
            answers_differ=float(np.mean([v[0]['answer'] != v[1]['answer'] for v in pairs.values()])))
    return result, records


@torch.no_grad()
def evaluate(model, tokenizer, rows, style, formal=False, limit=192, nll=False):
    model.eval(); prompts = [r['formal'] if formal else r['prompt'] for r in rows]
    result, records = scores(rows, generate(model, tokenizer, prompts, max_new_tokens=limit), style)
    if nll:
        total = tokens = 0
        for start in range(0, len(rows), 32):
            values = [tokenizer.example(p, target(r, style)) for p, r in zip(prompts[start:start + 32], rows[start:start + 32])]
            ids, labels, valid, pm = batch_tensors(values)
            prob, _ = model(ids, valid, pm)
            total += float(F.nll_loss(prob.log().transpose(1, 2), labels, ignore_index=-100, reduction='sum'))
            tokens += int((labels != -100).sum())
        result['token_nll'] = total / tokens
    return result, records


def train(config, out, resume=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'validation.json').exists(): raise FileExistsError('Completed run exists.')
    if (out / 'config.json').exists() and not resume: raise FileExistsError('Use --resume.')
    torch.set_num_threads(config['threads']); torch.manual_seed(config['seed'] + 90900)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for split in ['train', 'replay', 'dev', 'dev_phrases', 'legacy_dev']:
        assert digest(DATA / (split + '.jsonl')) == manifest['hashes'][split]
    vocabulary = json.loads((DATA / 'vocabulary.json').read_text())
    assert digest(DATA / 'vocabulary.json') == manifest['vocabulary_sha256']
    tokenizer = Tokenizer(vocabulary); rows = load('train'); replay = load('replay')
    style = config['style']; prepared = {}; pools = {}
    for key, values, formal in [('russian', rows, False), ('formal', rows, True), ('legacy', replay, False)]:
        prepared[key] = [tokenizer.example(r['formal'] if formal else r['prompt'], target(r, style)) for r in values]
        buckets = defaultdict(list)
        for i, value in enumerate(prepared[key]): buckets[(len(value[0]) + 31) // 32].append(i)
        pools[key] = {k: np.array(v) for k, v in buckets.items()}
    dev = {key: load(key) for key in ['dev', 'dev_phrases', 'legacy_dev']}
    model = initialize(config, vocabulary)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.01)
    rng = np.random.default_rng(config['sampling_seed']); source = source_path(config['condition'], config['seed'])
    source_hash = digest(source); best = (-1., -1., -1e9); start_step = 1
    examples = input_tokens = target_tokens = 0; counts = Counter(); max_grads = defaultdict(float)
    t0 = time.perf_counter(); prior_seconds = 0.; active_train_seconds = 0.

    def validate():
        metrics = {}; records = {}
        for key, values in dev.items():
            metrics[key], records[key] = evaluate(model, tokenizer, values, style, nll=(key != 'legacy_dev'))
        metrics['selection_score'] = [(metrics['dev']['answer_accuracy'] + metrics['dev_phrases']['answer_accuracy']) / 2,
            metrics['legacy_dev']['answer_accuracy'], -(metrics['dev']['token_nll'] + metrics['dev_phrases']['token_nll']) / 2]
        return metrics, records

    def checkpoint(step):
        return dict(config=config, vocabulary=vocabulary, model=model.state_dict(), optimizer=optimizer.state_dict(),
            step=step, best_score=list(best), numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            dataset_hashes=manifest['hashes'], source_checkpoint=str(source.relative_to(ROOT)), source_sha256=source_hash,
            input_tokens=input_tokens, target_tokens=target_tokens, training_examples=examples, stream_counts=dict(counts),
            seconds=prior_seconds + time.perf_counter() - t0, active_training_seconds=active_train_seconds,
            max_sampled_gradients=dict(max_grads))

    if resume:
        saved = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        assert saved['config'] == config and saved['dataset_hashes'] == manifest['hashes'] and saved['source_sha256'] == source_hash
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']; torch.set_rng_state(saved['torch_rng'])
        best = tuple(saved['best_score']); start_step = saved['step'] + 1
        examples, input_tokens, target_tokens = saved['training_examples'], saved['input_tokens'], saved['target_tokens']
        counts.update(saved['stream_counts']); max_grads.update(saved['max_sampled_gradients'])
        prior_seconds = saved['seconds']; active_train_seconds = saved['active_training_seconds']
    else:
        metrics, records = validate(); best = tuple(metrics['selection_score'])
        write_json(out / 'config.json', config); write_json(out / 'initial_validation.json', metrics)
        for key, values in records.items(): write_rows(out / ('initial_' + key + '.jsonl'), values)
        torch.save(checkpoint(0), out / 'best.pt')
        print(json.dumps(dict(run=out.name, step=0, selection=best)), flush=True)
    losses = []
    for step in range(start_step, config['steps'] + 1):
        tick = time.perf_counter()
        # Identical seeded sampling decisions across Russian and formal-only conditions.
        old = rng.random() < .35; russian = rng.random() < .8
        key = 'legacy' if old else ('russian' if russian and not config['formal_only'] else 'formal')
        eligible = prepared[key]; sampling_key = 'legacy' if old else 'russian'
        first = int(rng.integers(len(eligible))); bucket = (len(prepared[sampling_key][first][0]) + 31) // 32
        chosen = rng.choice(pools[sampling_key][bucket], config['batch_size'])
        ids, labels, valid, pm = batch_tensors([eligible[i] for i in chosen])
        lr = config['lr'] * min(1, step / 50) * (.3 + .7 * (1 + math.cos(math.pi * (step - 1) / max(1, config['steps'] - 1))) / 2)
        for group in optimizer.param_groups: group['lr'] = lr
        model.train(); optimizer.zero_grad(set_to_none=True); probabilities, _ = model(ids, valid, pm)
        token_loss = F.nll_loss(probabilities.log().transpose(1, 2), labels, ignore_index=-100, reduction='none')
        loss = (token_loss.sum(1) / (labels != -100).sum(1)).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss.')
        loss.backward()
        if step <= 3 or step % 100 == 0:
            max_grads['new_input_rows'] = max(max_grads['new_input_rows'], float(model.embedding.weight.grad[len(VOCAB):].abs().max()))
            for name, param in model.named_parameters():
                if 'edge_log_gain' in name and param.grad is not None:
                    max_grads[name] = max(max_grads[name], float(param.grad.abs().max()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        examples += len(chosen); input_tokens += int(valid.sum()); target_tokens += int((labels != -100).sum())
        counts[key] += len(chosen); losses.append(float(loss.detach())); active_train_seconds += time.perf_counter() - tick
        if step % 100 == 0:
            print(json.dumps(dict(run=out.name, step=step, loss=float(np.mean(losses[-100:])), seconds=time.perf_counter() - t0)), flush=True)
        if step % config['eval_every'] == 0 or step == config['steps']:
            metrics, records = validate(); score = tuple(metrics['selection_score']); improved = score > best; best = max(best, score)
            saved = checkpoint(step); torch.save(saved, out / 'last.pt')
            if improved:
                torch.save(saved, out / 'best.pt')
                for key, values in records.items(): write_rows(out / ('validation_' + key + '.jsonl'), values)
            entry = dict(step=step, validation=metrics, loss=float(np.mean(losses)), seconds=saved['seconds'],
                active_training_seconds=active_train_seconds, training_examples=examples, input_tokens=input_tokens,
                target_tokens=target_tokens, stream_counts=dict(counts))
            with (out / 'history.jsonl').open('a') as f: f.write(json.dumps(entry) + '\n')
            losses = []; print(json.dumps(dict(run=out.name, step=step, selection=score, best=best)), flush=True)
    saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=True); model.load_state_dict(saved['model'])
    metrics, records = validate()
    for key, values in records.items(): write_rows(out / ('validation_' + key + '.jsonl'), values)
    metrics.update(selected_step=saved['step'], checkpoint_sha256=digest(out / 'best.pt'),
        final_training_examples=examples, final_input_tokens=input_tokens, final_target_tokens=target_tokens,
        total_seconds=prior_seconds + time.perf_counter() - t0, active_training_seconds=active_train_seconds,
        stream_counts=dict(counts), max_sampled_gradients=dict(max_grads))
    write_json(out / 'validation.json', metrics); print('DONE ' + str(out), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    train(json.loads(Path(args.config).read_text()), args.out, args.resume)
