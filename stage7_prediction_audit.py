"""Check recovered prediction archives against immutable inputs and saved scores."""
from datetime import datetime, timezone
import json
from stage7_tasks import ROOT, load, digest, write_json
from stage7_train import scores
from stage7_run import OUT, verify


def check_file(path, rows, saved_metrics):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == len(rows), path
    assert len({r['id'] for r in records}) == len(records), path
    outputs = []
    for record in records:
        assert record['generated_tokens'] == len(record['text']) + int(record['halted']), path
        outputs.append({key: record[key] for key in ['text', 'halted', 'generated_tokens']})
    calculated, expected_records = scores(rows, outputs)
    assert records == expected_records, f'Input, label, metadata or derived correctness mismatch: {path}'
    for key, value in calculated.items(): assert saved_metrics[key] == value, (path, key)
    return dict(count=len(records), sha256=digest(path), all_ids_inputs_labels_and_scores_match=True)


def main():
    plan = verify(); datasets = {split: load(split) for split in plan['final_splits']}
    datasets['historical_retention'] = [json.loads(line) for line in (ROOT / 'data/stage6/test.jsonl').read_text().splitlines()]
    checkpoint_hashes = json.loads((OUT / 'checkpoints_before_test.json').read_text())
    verified = {}; core = historical = ablations = 0
    for name in plan['runs']:
        path = OUT / 'confirmation' / name
        saved = json.loads((path / 'final_metrics.json').read_text())
        assert saved['checkpoint_unchanged']
        for split, rows in datasets.items():
            file = path / ('predictions_' + split + '.jsonl')
            result = check_file(file, rows, saved['splits'][split])
            verified[str(file.relative_to(ROOT))] = result
            if split == 'historical_retention': historical += result['count']
            else: core += result['count']
        for setting, metrics in saved['diagnostic_summary']['ablations'].items():
            file = path / ('ablation_' + setting + '.jsonl')
            result = check_file(file, datasets['test'][:plan['diagnostic_rows']], metrics)
            verified[str(file.relative_to(ROOT))] = result; ablations += result['count']
        for filename in ['best.pt', 'last.pt']:
            assert digest(path / filename) == checkpoint_hashes[name + '/' + filename]
    sources = json.loads((OUT / 'source_baselines/metrics.json').read_text()); source_count = 0
    for name, value in sources.items():
        assert digest(ROOT / value['source']) == value['sha256']
        file = OUT / 'source_baselines' / (name + '.jsonl')
        result = check_file(file, datasets['test'], value['metrics'])
        verified[str(file.relative_to(ROOT))] = result; source_count += result['count']
    assert (core, historical, source_count) == (72960, 17280, 11520)
    result = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        main_predictions=core, historical_predictions=historical, source_predictions=source_count,
        ablation_predictions=ablations, files=verified, frozen_checkpoints_unchanged=True,
        note='Archive consistency verification, no additional inference. Forced-prefix and altered-history teacher diagnostics are separate; this audit covers free output and component ablations.')
    write_json(OUT / 'prediction_archive_audit.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}))


if __name__ == '__main__': main()
