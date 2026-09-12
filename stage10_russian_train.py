"""Learn clause translation, leaving execution weights untouched."""
import argparse
from collections import defaultdict, Counter
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from stage6_train import batch_tensors
from stage9_russian_train import ROOT, DATA as OLD_DATA, digest, write_json, write_rows, load as worlds
from stage9_russian_model import initialize, source_path
from stage9_russian_tokens import Tokenizer
from stage9_russian_engine import generate
from stage10_russian_data import DATA, load
from stage10_russian_pipeline import translate


@torch.no_grad()
def validate(model, tokenizer):
    metrics = {}; records = {}
    for split in ['dev', 'dev_phrases']:
        rows = worlds(split); outputs = translate(model, tokenizer, [r['prompt'] for r in rows])
        from stage10_russian_data import examples
        values = []; correct_clauses = all_clauses = 0
        for row, out in zip(rows, outputs):
            expected = examples(row); flags = [x['halted'] and x['text'] == y['target'] for x, y in zip(out['clauses'], expected)]
            correct = out['translation_halted'] and out['program'] == row['formal']
            correct_clauses += sum(flags); all_clauses += len(flags)
            values.append(dict(id=row['id'], expected=row['formal'], correct=correct, clause_correct=flags, **out))
        metrics[split] = dict(world_exact=sum(r['correct'] for r in values) / len(values),
            clause_exact=correct_clauses / all_clauses, count=len(values), clauses=all_clauses)
        records[split] = values
    metrics['selection_score'] = [metrics['dev_phrases']['world_exact'], metrics['dev_phrases']['clause_exact'], metrics['dev']['world_exact']]
    return metrics, records


def train(config, folder):
    out = Path(folder); out.mkdir(parents=True, exist_ok=True)
    if (out / 'validation.json').exists(): return
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    manifest = json.loads((DATA / 'manifest.json').read_text()); assert digest(DATA / 'train.jsonl') == manifest['hashes']['train']
    vocab = json.loads((OLD_DATA / 'vocabulary.json').read_text()); tokenizer = Tokenizer(vocab)
    rows = load('train'); prepared = [tokenizer.example(r['prompt'], r['target']) for r in rows]
    pools = defaultdict(list)
    for i, r in enumerate(rows): pools[r['kind']].append(i)
    kinds = sorted(pools); pools = {k: np.array(v) for k, v in pools.items()}
    model = initialize(config, vocab); optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
    rng = np.random.default_rng(config['sampling_seed']); source = source_path(config['condition'], config['seed'])
    source_hash = digest(source); best = (-1., -1., -1.); start = 1
    counts = Counter(); examples_seen = tokens = input_tokens = 0; train_seconds = 0.; prior_seconds = 0.; tick0 = time.perf_counter()

    def checkpoint(step):
        return dict(config=config, vocabulary=vocab, model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
            best_score=list(best), numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            source_sha256=source_hash, dataset_hashes=manifest['hashes'], training_examples=examples_seen,
            target_tokens=tokens, input_tokens=input_tokens, kind_counts=dict(counts), active_training_seconds=train_seconds,
            seconds=prior_seconds + time.perf_counter() - tick0)

    if (out / 'last.pt').exists():
        saved = torch.load(out / 'last.pt', weights_only=True); assert saved['config'] == config
        assert saved['source_sha256'] == source_hash and saved['dataset_hashes'] == manifest['hashes']
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']; torch.set_rng_state(saved['torch_rng'])
        best = tuple(saved['best_score']); start = saved['step'] + 1
        examples_seen, tokens, input_tokens = saved['training_examples'], saved['target_tokens'], saved['input_tokens']
        counts.update(saved['kind_counts']); train_seconds = saved['active_training_seconds']; prior_seconds = saved['seconds']
    else:
        metrics, records = validate(model, tokenizer); best = tuple(metrics['selection_score'])
        write_json(out / 'initial_validation.json', metrics); write_json(out / 'config.json', config)
        for k, v in records.items(): write_rows(out / ('initial_' + k + '.jsonl'), v)
        torch.save(checkpoint(0), out / 'best.pt')
        print(json.dumps(dict(run=out.name, step=0, selection=best)), flush=True)
    losses = []
    for step in range(start, config['steps'] + 1):
        tick = time.perf_counter(); kind = kinds[int(rng.integers(len(kinds)))]; chosen = rng.choice(pools[kind], 32)
        ids, labels, valid, pm = batch_tensors([prepared[i] for i in chosen])
        lr = .0005 * min(1, step / 50) * (.3 + .7 * (1 + math.cos(math.pi * (step - 1) / (config['steps'] - 1))) / 2)
        for g in optimizer.param_groups: g['lr'] = lr
        model.train(); optimizer.zero_grad(set_to_none=True); prob, _ = model(ids, valid, pm)
        loss_tokens = F.nll_loss(prob.log().transpose(1, 2), labels, ignore_index=-100, reduction='none')
        loss = (loss_tokens.sum(1) / (labels != -100).sum(1)).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss.')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        counts[kind] += len(chosen); examples_seen += len(chosen); tokens += int((labels != -100).sum()); input_tokens += int(valid.sum())
        losses.append(float(loss.detach())); train_seconds += time.perf_counter() - tick
        if step % 200 == 0: print(json.dumps(dict(run=out.name, step=step, loss=float(np.mean(losses[-200:])), seconds=time.perf_counter() - tick0)), flush=True)
        if step % config['eval_every'] == 0 or step == config['steps']:
            metrics, records = validate(model, tokenizer); score = tuple(metrics['selection_score']); improved = score > best; best = max(best, score)
            saved = checkpoint(step); torch.save(saved, out / 'last.pt')
            if improved:
                torch.save(saved, out / 'best.pt')
                for k, v in records.items(): write_rows(out / ('validation_' + k + '.jsonl'), v)
            entry = dict(step=step, validation=metrics, loss=float(np.mean(losses)), seconds=saved['seconds'])
            with (out / 'history.jsonl').open('a') as f: f.write(json.dumps(entry) + '\n')
            losses = []; print(json.dumps(dict(run=out.name, step=step, selection=score)), flush=True)
    saved = torch.load(out / 'best.pt', weights_only=True); model.load_state_dict(saved['model']); metrics, records = validate(model, tokenizer)
    for k, v in records.items(): write_rows(out / ('validation_' + k + '.jsonl'), v)
    metrics.update(selected_step=saved['step'], checkpoint_sha256=digest(out / 'best.pt'), training_examples=examples_seen,
        target_tokens=tokens, input_tokens=input_tokens, active_training_seconds=train_seconds, total_seconds=prior_seconds + time.perf_counter() - tick0)
    write_json(out / 'validation.json', metrics); print('DONE ' + out.name, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--out', required=True); a = p.parse_args()
    train(json.loads(Path(a.config).read_text()), a.out)
