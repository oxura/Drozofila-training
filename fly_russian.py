"""Raw Russian input -> neural generation. No interpreter or task teachers."""
import argparse
import json
from pathlib import Path
import sys
import torch
from stage9_russian_model import load_model
from stage9_russian_tokens import Tokenizer
from stage9_russian_engine import generate

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description='Experimental Russian instruction comprehension; numeric or formal-trace output.')
    p.add_argument('text', nargs='?', help='Russian text; read stdin if omitted.')
    p.add_argument('--checkpoint'); p.add_argument('--json', action='store_true'); p.add_argument('--max-new-tokens', type=int, default=192)
    args = p.parse_args(); torch.set_num_threads(1)
    if args.checkpoint: path = Path(args.checkpoint)
    else:
        defaults = json.loads((ROOT / 'results/stage9_russian/default.json').read_text()); path = ROOT / defaults['checkpoint']
    model, saved = load_model(path); tokenizer = Tokenizer(saved['vocabulary'])
    prompt = args.text if args.text is not None else sys.stdin.read().strip()
    if not prompt: p.error('Provide a nonempty Russian instruction.')
    try: result = generate(model, tokenizer, [prompt], max_new_tokens=args.max_new_tokens)[0]
    except ValueError as error: p.error(str(error))
    result.update(answer=result['text'].rsplit('|', 1)[-1], checkpoint=str(path), target_style=saved['config']['style'])
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result['text'])


if __name__ == '__main__': main()
