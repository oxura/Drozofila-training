"""Learned clause translation followed by a frozen neural executor.

Sentence segmentation and joining instructions are supplied interfaces.
Neither clause meanings nor variable values are parsed or computed here.
"""
import re
from stage9_russian_engine import generate


def clauses(text): return [x.strip() for x in re.findall(r'[^.!?]+[.!?]?', text) if x.strip()]


def translate(model, tokenizer, prompts):
    parts = [clauses(p) for p in prompts]; flat = [p for group in parts for p in group]
    outputs = generate(model, tokenizer, flat, max_new_tokens=96)
    results = []; offset = 0
    for group in parts:
        out = outputs[offset:offset + len(group)]; offset += len(group)
        results.append(dict(program='код ' + ';'.join(x['text'] for x in out),
            translation_halted=all(x['halted'] for x in out), clauses=out))
    return results


def solve(translator, tokenizer, executor, prompts):
    results = translate(translator, tokenizer, prompts)
    outputs = generate(executor, tokenizer, [r['program'] for r in results], max_new_tokens=512)
    for result, out in zip(results, outputs):
        result.update(execution=out, answer=out['text'].rsplit('|', 1)[-1],
            halted=result['translation_halted'] and out['halted'])
    return results
