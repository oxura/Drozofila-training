"""Meaningful preflight: labels, old-function preservation, causal cached inference."""
import json
import sys
import time
import torch
from stage6_tokens import prompt_ids, EOS
from stage6_train import batch_tensors
from stage7_model import MemoryReasoner
from stage9_russian_data import check, load, DATA, write_json
from stage9_russian_tokens import Tokenizer
from stage9_russian_model import initialize, source_path
from stage9_russian_engine import generate


def main():
    torch.set_num_threads(1); torch.manual_seed(90900)
    data_checks = check(); vocab = json.loads((DATA / 'vocabulary.json').read_text()); token = Tokenizer(vocab)
    legacy = load('legacy_dev'); rows = load('dev')
    for r in legacy + load('replay'): assert token.prompt(r['prompt']) == prompt_ids(r['prompt'])
    old = MemoryReasoner('mlp', 0, 'history'); old.load_state_dict(torch.load(source_path('mlp', 0), weights_only=True)['model']); old.eval()
    model = initialize(dict(condition='mlp', seed=0), vocab); model.eval()
    ids, labels, valid, pm = batch_tensors([token.example(r['prompt'], r['target']) for r in legacy[:4]])
    with torch.no_grad():
        p, _ = model(ids, valid, pm); q, _ = old(ids, valid, pm)
        old_error = float((p - q).abs().max()); assert old_error < 1e-7, old_error
        seq = token.prompt(rows[0]['prompt']); x = torch.tensor([seq]); v = torch.ones_like(x, dtype=torch.bool)
        pm = v.clone(); pm[:, [0, -1]] = False
        full, _ = model(x, v, pm); cache = None; parts = []
        for i in range(x.shape[1]):
            part, cache = model(x[:, i:i + 1], v[:, i:i + 1], pm[:, i:i + 1], cache); parts.append(part)
        cache_error = float((full - torch.cat(parts, 1)).abs().max()); assert cache_error < 2e-5, cache_error
        altered = x.clone(); start = x.shape[1] // 2; altered[:, start:] = token.ids['9']
        p, _ = model(altered, v, pm); causal_error = float((p[:, :start] - full[:, :start]).abs().max())
        assert causal_error < 1e-7
        solo = generate(model, token, [rows[0]['prompt']], max_new_tokens=12)[0]
        batched = generate(model, token, [rows[1]['prompt'], rows[0]['prompt']], max_new_tokens=12)[1]
        assert solo == batched
    # Backprop through new word embeddings and a representative old network.
    prepared = [token.example(r['prompt'], r['trace'] + '|' + r['answer']) for r in load('train')[:16]]
    ids, labels, valid, pm = batch_tensors(prepared); optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
    t0 = time.perf_counter()
    for _ in range(8):
        optimizer.zero_grad(); prob, _ = model(ids, valid, pm)
        loss = torch.nn.functional.nll_loss(prob.log().transpose(1, 2), labels, ignore_index=-100)
        loss.backward(); optimizer.step()
    grad = float(model.embedding.weight.grad[len(vocab['output_vocabulary']):].abs().max()); assert grad > 0
    result = dict(data=data_checks, legacy_examples_with_identical_tokenization=len(legacy) + len(load('replay')),
        initial_legacy_probability_max_error=old_error, cached_full_max_error=cache_error, causal_max_error=causal_error,
        left_padding_generation_matches=True, new_embedding_gradient_max=grad, discarded_smoke_updates=8,
        smoke_seconds_per_update=(time.perf_counter() - t0) / 8,
        input_vocabulary=len(vocab['tokens']), output_vocabulary=len(vocab['output_vocabulary']))
    write_json('results/stage9_russian/preflight.json', result); print(json.dumps(result))


if __name__ == '__main__': main()
