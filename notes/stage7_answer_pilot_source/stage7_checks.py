"""Scientific integrity checks: labels, splits, causal inference and preservation."""
from collections import Counter
import json
import re
import time
import numpy as np
import torch
from stage6_model import load_model as old_load
from stage6_train import batch_tensors
from stage7_tasks import ROOT, DATA, load, signature, independent_answer, digest, write_json
from stage7_train import prepare
from stage7_model import initialize, SlotMemory


def text_answer(row):
    if row['family'] not in ('memory', 'code'): return independent_answer(row)
    state = {}; parts = row['prompt'][4:].split(';')
    for part in parts[:-1]:
        name, value = part.split('=', 1)
        if value.isdigit(): state[name] = int(value); continue
        found = re.fullmatch(r'\(([a-z]+|\d+)([+*\-])([a-z]+|\d+)\)%10', value)
        assert found, part
        a, op, b = found.groups()
        a = int(a) if a.isdigit() else state[a]; b = int(b) if b.isdigit() else state[b]
        state[name] = (a + b if op == '+' else a - b if op == '-' else a * b) % 10
    return str(state[parts[-1][1:]])


def check_data():
    manifest = json.loads((DATA / 'manifest.json').read_text()); groups = {}; strings = {}; counts = {}
    for split, expected in manifest['hashes'].items():
        assert digest(DATA / f'{split}.jsonl') == expected
        rows = load(split); counts[split] = len(rows)
        assert all(text_answer(r) == r['answer'] for r in rows), split
        strings[split] = {r['prompt'] for r in rows}
        groups[split] = {signature(r) for r in rows if r['family'] in ('memory', 'code')}
        assert len(strings[split]) == len(rows)
        if split in ('train', 'val', 'test'):
            for family in ['memory', 'code']:
                strata = Counter((r['query_last'], r['answer']) for r in rows if r['family'] == family)
                assert len(strata) == 20 and len(set(strata.values())) == 1
    names = [x for x in strings if x != 'replay']
    for i, name in enumerate(names):
        for other in names[:i]: assert not strings[name] & strings[other], (name, other)
    for name in ['val', 'test']:
        assert not groups[name] & groups['train']
        assert not groups[name] & groups['replay']
    assert not groups['val'] & groups['test']
    historical = set()
    for path in (ROOT / 'data/stage6').glob('*.jsonl'):
        historical.update(json.loads(line)['prompt'] for line in path.read_text().splitlines())
    for name in names: assert not strings[name] & historical
    vocabulary = ''.join(r['prompt'] + r['answer'] for r in load('train') + load('replay'))
    assert all(c in vocabulary for c in 'abcd') and all(c not in vocabulary for c in 'ijkl')
    return dict(counts=counts, prompt_overlap=0, binding_group_train_val_test_overlap=0,
                historical_prompt_overlap=0, labels_checked_from_text=True)


def config(mode='prompt', wider=False):
    return dict(condition='mlp', seed=0, mode=mode, wider=wider)


def check_models():
    torch.set_num_threads(2); torch.manual_seed(707)
    rows = [r for r in load('train') if r['family'] == 'memory'][:4]
    ids, labels, valid, pm = batch_tensors([prepare(r) for r in rows])
    result = {}; models = {}
    for mode, wider in [('prompt', False), ('history', False), ('slots', False), ('history', True)]:
        name = 'wider' if wider else mode
        model, source = initialize(config(mode, wider)); model.eval(); models[name] = model
        with torch.no_grad():
            full, _ = model(ids, valid, pm)
            # Perturb future inputs and ensure all earlier probability vectors agree.
            cut = ids.shape[1] // 2; changed = ids.clone(); changed[:, cut:] = 5
            other, _ = model(changed, valid, pm)
            error = float((full[:, :cut] - other[:, :cut]).abs().max())
            assert error < 2e-6, (name, 'future leakage', error)
            # A full sequence and a prompt-prefill plus one-token steps agree.
            tokens, prefix = prepare(rows[0]); tokens = torch.tensor(tokens[:-1])[None]
            v = torch.ones_like(tokens, dtype=torch.bool); p = torch.zeros_like(v); p[:, 1:prefix - 1] = True
            expected, _ = model(tokens, v, p)
            prefix_out, cache = model(tokens[:, :prefix], v[:, :prefix], p[:, :prefix])
            outputs = [prefix_out]
            for t in range(prefix, tokens.shape[1]):
                out, cache = model(tokens[:, t:t + 1], v[:, t:t + 1], p, cache); outputs.append(out)
            actual = torch.cat(outputs, 1)
            parity = float((expected - actual).abs().max())
            assert parity < 1e-5, (name, 'cache mismatch', parity)
            # Check a near-tolerance fp32 discrepancy in fp64 to distinguish
            # accumulated rounding from an incorrect cache/recurrence.
            double_error = None
            if parity >= 2e-6:
                import copy
                double_model = copy.deepcopy(model).double()
                double_expected, _ = double_model(tokens, v, p)
                first_out, double_cache = double_model(tokens[:, :prefix], v[:, :prefix], p[:, :prefix])
                double_outputs = [first_out]
                for t in range(prefix, tokens.shape[1]):
                    value, double_cache = double_model(tokens[:, t:t + 1], v[:, t:t + 1], p, double_cache)
                    double_outputs.append(value)
                double_error = float((double_expected - torch.cat(double_outputs, 1)).abs().max())
                assert double_error < 1e-10, (name, double_error)
            # Left padding must not change outputs, including recurrent memory.
            padded = torch.cat([torch.zeros(1, 7, dtype=torch.long), tokens], 1)
            pv = padded != 0; pp = torch.cat([torch.zeros(1, 7, dtype=torch.bool), p], 1)
            po, _ = model(padded, pv, pp)
            pad_error = float((po[:, 7:] - expected).abs().max())
            assert pad_error < 1e-5, (name, 'padding', pad_error)
        result[name] = dict(parameters=sum(p.numel() for p in model.parameters()),
                            future_error=error, cached_error=parity, padding_error=pad_error,
                            double_cached_error=double_error)
    with torch.no_grad():
        old, _ = old_load(source); old.eval()
        old_probs, _ = old(ids, valid, pm); prompt_probs, _ = models['prompt'](ids, valid, pm)
        preservation = float((old_probs[labels != -100] - prompt_probs[labels != -100]).abs().max())
        assert preservation < 2e-6
        h, _ = models['history'](ids, valid, pm)
        for name in ['slots', 'wider']:
            p, _ = models[name](ids, valid, pm)
            delta = float((p - h).abs().max()); assert delta < 1e-5, (name, delta)
            if delta >= 2e-6:
                import copy
                small, _ = copy.deepcopy(models['history']).double()(ids, valid, pm)
                grown, _ = copy.deepcopy(models[name]).double()(ids, valid, pm)
                double_delta = float((small - grown).abs().max())
                assert double_delta < 1e-10, (name, double_delta)
                result[name]['double_initial_function_error'] = double_delta
            result[name]['initial_function_error'] = delta
    result['prompt']['source_preservation_error'] = preservation
    # Compare the scan against a separate sequential recurrence, including nonzero initial memory.
    memory = SlotMemory(96); h = torch.randn(3, 41, 96); valid = torch.ones(3, 41, dtype=torch.bool)
    valid[0, :9] = False; valid[1, -5:] = False
    initial = torch.randn(3, 8, 16); expected = initial.clone()
    gate = torch.sigmoid(memory.write_gate(h)) * torch.softmax(memory.write_address(h), -1) * valid[..., None]
    values = torch.tanh(memory.value(h))
    for t in range(h.shape[1]): expected = (1 - gate[:, t, :, None]) * expected + gate[:, t, :, None] * values[:, t, None, :]
    _, actual = memory(h, valid, initial)
    error = float((actual - expected).abs().max().detach()); assert error < 2e-6
    result['scan_serial_error'] = error
    # Two verification updates allow the zero-initialized output to expose gradients upstream.
    slot_model = models['slots']; optimizer = torch.optim.AdamW(slot_model.parameters(), lr=.0005)
    gradients = {}; t0 = time.perf_counter()
    for _ in range(2):
        optimizer.zero_grad(); p, _ = slot_model(ids, valid=ids != 0, prompt_mask=pm)
        loss = torch.nn.functional.nll_loss(p.log().transpose(1, 2), labels, ignore_index=-100)
        loss.backward()
        for name, parameter in slot_model.named_parameters():
            if name.startswith('memory.'):
                gradients[name] = max(gradients.get(name, 0), float(parameter.grad.abs().max()))
        optimizer.step()
    assert all(v > 0 for v in gradients.values()), gradients
    result['slot_gradients'] = gradients; result['two_verification_updates_seconds'] = time.perf_counter() - t0
    result['verification_weights_discarded'] = True
    return result


if __name__ == '__main__':
    result = dict(data=check_data(), models=check_models(), final_model_predictions_opened=False)
    out = ROOT / 'results/stage7'; out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'checks.json', result); print(json.dumps(result))
