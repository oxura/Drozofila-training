"""Russian instruction -> learned translation -> frozen neural execution."""
import argparse
import json
from pathlib import Path
import sys
import torch
from stage9_russian_model import load_model, initialize
from stage9_russian_tokens import Tokenizer
from stage10_russian_pipeline import solve

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description='Experimental Russian instruction pipeline; sentence boundaries are supplied by the interface.')
    p.add_argument('text', nargs='?'); p.add_argument('--json', action='store_true'); p.add_argument('--checkpoint'); a = p.parse_args()
    torch.set_num_threads(1)
    if a.checkpoint: checkpoint = Path(a.checkpoint)
    else: checkpoint = ROOT / json.loads((ROOT / 'results/stage10_russian/default.json').read_text())['checkpoint']
    translator, saved = load_model(checkpoint); tokenizer = Tokenizer(saved['vocabulary'])
    executor = initialize(dict(condition='mlp', seed=1), saved['vocabulary'])
    prompt = a.text if a.text is not None else sys.stdin.read().strip()
    if not prompt: p.error('Provide Russian instructions.')
    try: result = solve(translator, tokenizer, executor, [prompt])[0]
    except ValueError as error: p.error(str(error))
    result['translator_checkpoint'] = str(checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2) if a.json else result['answer'])


if __name__ == '__main__': main()
