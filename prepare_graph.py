"""Validate the complete download and select a label-independent induced subgraph."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=256)
    args = parser.parse_args()
    t0 = time.perf_counter()
    manifest = json.loads((ROOT / 'data/source_manifest.json').read_text())
    for item in manifest['files']:
        path = ROOT / item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    ids = pd.read_csv(ROOT / 'data/raw/Completeness_783.csv', index_col=0).index.to_numpy()
    columns = ['Presynaptic_ID', 'Postsynaptic_ID', 'Presynaptic_Index',
               'Postsynaptic_Index', 'Connectivity', 'Excitatory x Connectivity']
    df = pd.read_parquet(ROOT / 'data/raw/Connectivity_783.parquet', columns=columns)
    pre = df['Presynaptic_Index'].to_numpy()
    post = df['Postsynaptic_Index'].to_numpy()
    assert pre.min() >= 0 and post.min() >= 0 and pre.max() < len(ids) and post.max() < len(ids)
    assert np.array_equal(ids[pre], df['Presynaptic_ID'].to_numpy())
    assert np.array_equal(ids[post], df['Postsynaptic_ID'].to_numpy())
    weights = df['Excitatory x Connectivity'].to_numpy(dtype=np.float32)
    n = len(ids)
    A = sparse.csr_matrix((weights, (post, pre)), shape=(n, n))
    A.sum_duplicates()
    before_zero = A.nnz
    A.eliminate_zeros()
    summary = dict(source_neurons=n, source_rows=len(df),
                   source_synapse_count=int(df['Connectivity'].sum()),
                   source_zero_weight_rows=int((weights == 0).sum()),
                   source_self_loop_rows=int((pre == post).sum()),
                   aggregate_pairs_before_zero_removal=int(before_zero),
                   active_directed_pairs=int(A.nnz),
                   source_positive_rows=int((weights > 0).sum()),
                   source_negative_rows=int((weights < 0).sum()),
                   source_sha=manifest['commit'])
    del df, pre, post, weights

    # Demonstrates propagation on ALL source neurons, with simplified rate cells.
    absA = abs(A)
    normalizer = np.asarray(absA.sum(axis=1)).ravel()
    normalized = sparse.diags(0.8 / np.maximum(normalizer, 1)) @ A
    center = int(np.argmax(np.asarray(absA.sum(axis=0)).ravel()))
    state = np.zeros(n, dtype=np.float32)
    signal = np.zeros(n, dtype=np.float32)
    signal[center] = 1
    propagation = []
    for step in range(12):
        state = np.tanh(normalized @ state + (signal if step < 3 else 0))
        assert np.isfinite(state).all()
        propagation.append(dict(step=step + 1, active_neurons=int((np.abs(state) > 1e-8).sum()),
                                l2_norm=float(np.linalg.norm(state))))
    summary['full_graph_rate_smoke'] = propagation
    # Strong-neighbor traversal. Not a named anatomical circuit or random sample.
    neighbors = (absA + absA.T).tocsr()
    selected, visited = [], set()
    heap = [(-float('inf'), center)]
    while heap and len(selected) < args.nodes:
        _, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        selected.append(node)
        start, end = neighbors.indptr[node:node + 2]
        for nxt, strength in zip(neighbors.indices[start:end], neighbors.data[start:end]):
            if int(nxt) not in visited:
                heapq.heappush(heap, (-float(strength), int(nxt)))
    assert len(selected) == args.nodes
    selected = np.array(sorted(selected), dtype=np.int64)
    sub = A[selected][:, selected].tocoo()
    keep = sub.row != sub.col
    p, q, w = sub.col[keep], sub.row[keep], sub.data[keep]
    path = ROOT / f'data/graph_{args.nodes}.npz'
    np.savez_compressed(path, root_ids=ids[selected], source_indices=selected,
                        pre=p.astype(np.int64), post=q.astype(np.int64), weight=w)
    summary['pilot'] = dict(nodes=len(selected), edges=len(w),
                            positive_edges=int((w > 0).sum()), negative_edges=int((w < 0).sum()),
                            removed_self_edges=int((~keep).sum()),
                            selection='strong-neighbor traversal from maximum outgoing absolute strength',
                            center_source_index=center, center_root_id=str(ids[center]),
                            selected_fraction=len(selected) / n,
                            graph_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    summary['prepare_seconds'] = time.perf_counter() - t0
    (ROOT / 'results/data_audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
