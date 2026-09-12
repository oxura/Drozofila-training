"""Stage7 neural CLI: accepts text, does not parse tasks or import their teachers."""
import argparse
import json
from pathlib import Path
import torch
from stage7_model import load_model
from stage7_engine import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('prompt'); p.add_argument('--checkpoint'); p.add_argument('--json', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=512)
    args = p.parse_args()
    if not args.prompt.strip(): p.error('Prompt must not be empty.')
    if not 1 <= args.max_new_tokens <= 4096: p.error('Token limit must be 1..4096.')
    root = Path(__file__).resolve().parent
    path = args.checkpoint
    if path is None:
        path = json.loads((root / 'results/stage7/default_model.json').read_text())['checkpoint']
    path = root / path
    torch.set_num_threads(2); model, saved = load_model(path)
    try: result = generate(model, [args.prompt], args.max_new_tokens)[0]
    except ValueError as error: p.error(str(error))
    result.update(checkpoint=str(path), mode=saved['config']['mode'], condition=saved['config']['condition'])
    print(json.dumps(result, ensure_ascii=False) if args.json else result['text'])


if __name__ == '__main__': main()
