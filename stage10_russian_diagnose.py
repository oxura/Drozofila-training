"""Development-only diagnosis after the gate; final tests remain closed."""
from collections import defaultdict
import json
import re
import torch
from stage9_russian_train import ROOT, load, write_json, write_rows
from stage9_russian_model import load_model, initialize
from stage9_russian_tokens import Tokenizer
from stage10_russian_experiment import OUT
from stage10_russian_data import examples
from stage10_russian_evaluate import evaluate_rows


def copy_format_audit():
    records = [json.loads(x) for x in (OUT / 'pilot/pilot/validation_dev_phrases.jsonl').read_text().splitlines()]
    counts = dict(copies=0, valid_copy_shape_ignoring_whitespace=0, both_roles_correct_ignoring_whitespace=0,
        literal_assignment_instead=0, other_format=0)
    for row, record in zip(load('dev_phrases'), records):
        for expected, output in zip(examples(row), record['clauses']):
            if expected['kind'] != 'copy': continue
            counts['copies'] += 1; text = ''.join(output['text'].split())
            if re.fullmatch(r'([a-z]+)=\(([a-z]+)\+0\)%10', text):
                counts['valid_copy_shape_ignoring_whitespace'] += 1
                counts['both_roles_correct_ignoring_whitespace'] += int(text == expected['target'] and output['halted'])
            elif re.fullmatch(r'[a-z]+=[0-9]', text): counts['literal_assignment_instead'] += 1
            else: counts['other_format'] += 1
    write_json(OUT / 'diagnosis/copy_format_audit.json', counts); print(json.dumps(counts))
    return counts


def main():
    torch.set_num_threads(1)
    assert not (OUT / 'test_open.json').exists()
    decision = json.loads((OUT / 'confirmation_plan.json').read_text()); assert not decision['go']
    model, saved = load_model(OUT / 'pilot/pilot/best.pt'); tokenizer = Tokenizer(saved['vocabulary'])
    executor = initialize(dict(condition='mlp', seed=1), saved['vocabulary']); result = {}
    folder = OUT / 'diagnosis'; folder.mkdir(parents=True, exist_ok=True)
    for split in ['dev', 'dev_phrases']:
        rows = load(split); per_kind = defaultdict(list); errors = []
        records = [json.loads(x) for x in (OUT / 'pilot/pilot' / ('validation_' + split + '.jsonl')).read_text().splitlines()]
        for row, record in zip(rows, records):
            for expected, output in zip(examples(row), record['clauses']):
                correct = output['halted'] and output['text'] == expected['target']; per_kind[expected['kind']].append(correct)
                if not correct and len(errors) < 12: errors.append(dict(prompt=expected['prompt'], expected=expected['target'], actual=output['text'], kind=expected['kind']))
        pipeline, answers = evaluate_rows(model, tokenizer, executor, rows)
        oracle, oracle_answers = evaluate_rows(None, tokenizer, executor, rows, oracle=True)
        # Recompute validity rather than trusting a generated answer field.
        for row, answer in zip(rows, answers):
            assert answer['correct'] == (answer['halted'] and answer['execution']['text'].rsplit('|', 1)[-1] == row['answer'])
        write_rows(folder / (split + '_pipeline.jsonl'), answers); write_rows(folder / (split + '_oracle.jsonl'), oracle_answers)
        result[split] = dict(translation_by_kind={k: dict(count=len(v), accuracy=sum(v) / len(v)) for k, v in per_kind.items()},
            pipeline=pipeline, oracle_formal_input=oracle, first_errors_in_fixed_data_order=errors)
        print(json.dumps(dict(split=split, pipeline=pipeline['answer_accuracy'], oracle=oracle['answer_accuracy'], kinds=result[split]['translation_by_kind'])), flush=True)
    result['interpretation'] = 'Open development diagnostics, not independent confirmation. Oracle translation is used only in the explicitly separate diagnostic control.'
    write_json(folder / 'summary.json', result)
    copy_format_audit()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--format-only', action='store_true'); args = p.parse_args()
    if args.format_only: copy_format_audit()
    else: main()
