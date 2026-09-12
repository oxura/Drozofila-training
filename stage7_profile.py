"""Bounded CPU throughput check; profiling weights are discarded."""
import gc
import json
import time
import torch
from stage6_train import batch_tensors
from stage7_tasks import ROOT, load, write_json
from stage7_model import initialize
from stage7_train import prepare


def main():
    data = load('train'); batches = []; identifiers = []
    for family in ['memory', 'code', 'sum', 'logic']:
        rows = [r for r in data if r['family'] == family][:16]
        batches.append(batch_tensors([prepare(r) for r in rows])); identifiers.append([r['id'] for r in rows])
    records = []
    for mode in ['history', 'slots']:
        for threads in [1, 2]:
            torch.set_num_threads(threads); model, _ = initialize(dict(mode=mode, condition='mlp', seed=0, wider=False))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
            started = None
            for step in range(48):
                if step == 8: started = time.perf_counter()
                ids, labels, valid, pm = batches[step % 4]
                optimizer.zero_grad(set_to_none=True); p, _ = model(ids, valid, pm)
                loss = torch.nn.functional.nll_loss(p.log().transpose(1, 2), labels, ignore_index=-100)
                loss.backward(); optimizer.step()
            seconds = time.perf_counter() - started
            records.append(dict(mode=mode, threads=threads, steps=40, seconds=seconds,
                                seconds_per_step=seconds / 40, approximate_core_seconds_per_step=threads * seconds / 40))
            del model, optimizer; gc.collect()
    result = dict(records=records, input_ids=identifiers, profiling_only=True,
                  optimizer_updates=192, weights_discarded=True,
                  note='Single-process measurement, not a guarantee of multi-process scaling; includes longer mixed-family padding than bucketed training.')
    write_json(ROOT / 'results/stage7/profile.json', result); print(json.dumps(result))


if __name__ == '__main__': main()
