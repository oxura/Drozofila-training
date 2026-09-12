"""Use a saved pilot checkpoint; all answer tokens come from the trained network."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import ConnectomeLM
from tasks import tokenize

ROOT = Path(__file__).resolve().parent


def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    cfg = ckpt['config']
    if 'variant' in cfg:
        from stage2_train import make_model
        net = make_model(cfg, ckpt['vocab'])
    else:
        graph = dict(np.load(ROOT / f"data/graph_{cfg['nodes']}.npz"))
        net = ConnectomeLM(graph, len(ckpt['vocab']), cfg['condition'], cfg['cell'], cfg['seed'])
    net.load_state_dict(ckpt['model'])
    net.eval()
    return net, ckpt['vocab'], cfg


def answer(net, vocab, prompt):
    tokens = tokenize(prompt)
    lookup = {word: i for i, word in enumerate(vocab)}
    ids = torch.tensor([[1] + [lookup.get(w, 4) for w in tokens] + [2]])
    sequence = net.generate(ids)[0].tolist()
    sequence = sequence[:sequence.index(3)] if 3 in sequence else sequence
    output = [vocab[i] for i in sequence]
    is_memory = any(w in tokens for w in ['повтори', 'разверни'])
    text = ''.join(output) if not is_memory and all(w in list('0123456789-') for w in output) else ' '.join(output)
    return dict(prompt=prompt, answer=text, output_tokens=output,
                unknown_input_words=sorted(set(tokens) - set(vocab)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('prompt', nargs='?')
    p.add_argument('--checkpoint', default='results/stage2/confirmation/memory_rate_fly_seed0/best.pt')
    p.add_argument('--interactive', action='store_true')
    p.add_argument('--json', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    net, vocab, config = load_checkpoint(checkpoint)
    if not args.interactive and not args.prompt:
        p.error('Supply a prompt or --interactive')
    while True:
        try:
            prompt = input('> ') if args.interactive else args.prompt
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip().lower() in ['выход', 'exit', 'quit']:
            break
        result = answer(net, vocab, prompt)
        print(json.dumps(result, ensure_ascii=False) if args.json else result['answer'])
        if result['unknown_input_words'] and not args.json:
            print('Неизвестные входные слова:', ', '.join(result['unknown_input_words']))
        if not args.interactive:
            break


if __name__ == '__main__':
    main()
