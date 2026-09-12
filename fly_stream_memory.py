"""Run a chosen streaming pilot on literal writes. No teacher or task imports."""
import argparse
import json
from pathlib import Path
import torch
from stage8_stream import encode_prompt, load_model, answer_stream


def main():
    parser = argparse.ArgumentParser(description='Experimental streaming memory for literal assignments and ?name.')
    parser.add_argument('prompt'); parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--json', action='store_true'); args = parser.parse_args()
    try:
        events = encode_prompt(args.prompt)
        if len(events) > 256: raise ValueError('At most 256 events per invocation.')
        if not args.checkpoint.is_file(): raise ValueError('Checkpoint file does not exist.')
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(1); model, _ = load_model(args.checkpoint)
    answer = answer_stream(model, args.prompt)
    if args.json:
        print(json.dumps(dict(answer=answer, checkpoint=str(args.checkpoint.resolve()),
            stage='stage8_pilot', evaluation_status='development only'), ensure_ascii=False))
    else: print(answer)


if __name__ == '__main__': main()
