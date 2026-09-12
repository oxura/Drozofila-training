"""Checks that protect the scientific interpretation of this pilot."""
import json
from pathlib import Path
import numpy as np
import torch
from model import ConnectomeLM, rewire
from tasks import build_tasks, encode_rows, tokenize
from train import batch
from evaluate import executable_code_matches

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    graph = dict(np.load(ROOT / 'data/graph_256.npz'))
    records, vocab = build_tasks()
    lookup = set(vocab)
    assert all(set(tokenize(r['prompt']) + tokenize(r['answer'])) <= lookup
               for rows in records.values() for r in rows)
    pre, post, swaps = rewire(graph['pre'], graph['post'], seed=9000)
    assert np.array_equal(np.bincount(pre, minlength=256), np.bincount(graph['pre'], minlength=256))
    assert np.array_equal(np.bincount(post, minlength=256), np.bincount(graph['post'], minlength=256))
    assert len(set(zip(pre, post))) == len(pre)
    original = set(zip(graph['pre'], graph['post']))
    overlap = len(original & set(zip(pre, post))) / len(original)
    assert overlap < 0.8
    encoded = encode_rows(records['train'][:8], vocab)
    x, y = batch(encoded, np.arange(8), 'cpu')
    result = dict(partition_sizes={k: len(v) for k, v in records.items()},
                  vocabulary_size=len(vocab), rewired_edge_overlap=overlap,
                  rewiring_swaps=swaps, checks=[])
    for cell in ['rate', 'lif']:
        model = ConnectomeLM(graph, len(vocab), cell=cell).double()
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output.transpose(1, 2), y)
        loss.backward()
        grad = model.edge_log_gain.grad
        assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().max()) > 0
        matrix = model.matrix().detach()
        assert torch.count_nonzero(matrix) == len(graph['weight'])
        assert np.array_equal(matrix[graph['post'], graph['pre']].sign().numpy(), np.sign(graph['weight']))
        if cell == 'rate':
            index = int(grad.abs().argmax())
            eps = 1e-5
            values = []
            for value in [eps, -eps]:
                with torch.no_grad():
                    model.edge_log_gain[index] = value
                    values.append(float(torch.nn.functional.cross_entropy(model(x).transpose(1, 2), y)))
            numerical = (values[0] - values[1]) / (2 * eps)
            assert abs(numerical - float(grad[index])) < 1e-6
        result['checks'].append(dict(cell=cell, loss=float(loss.detach()),
                                      max_edge_gradient=float(grad.abs().max())))
    frozen = ConnectomeLM(graph, len(vocab), condition='frozen')
    assert not frozen.edge_log_gain.requires_grad
    correct = 'def f ( a , b ) : return a + b'.split()
    wrong = 'def f ( a , b ) : return a - b'.split()
    assert executable_code_matches(correct, 'код сложи a b') == (True, True)
    assert executable_code_matches(wrong, 'код сложи a b') == (True, False)
    assert executable_code_matches(['не', 'код'], 'код сложи a b') == (False, False)
    result['status'] = 'passed'
    (ROOT / 'results/checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
