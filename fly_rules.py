"""Infer a restricted Boolean rule from user demonstrations, without task oracle."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stage5_model import load_model
from stage5_engine import predict
from stage5_memory import encode

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'results/stage5/confirmation/memory_fly_seed0/best.pt'


def parse_bits(text):
    bits = text.strip().replace(' ', '')
    if not bits or any(c not in '01' for c in bits):
        raise ValueError('Введите строку из нулей и единиц, например 01011001.')
    return [int(c) for c in bits]


def suggest(model, support_x, support_y, excluded=()):
    """Written active-learning policy: choose maximal predictive uncertainty.

    No teacher is consulted. The user/environment must supply the new label.
    Candidate enumeration and entropy selection are not learned neural actions.
    """
    n = len(support_x[0])
    if not 1 <= n <= 12:
        raise ValueError('Предложение следующего примера поддерживает 1–12 признаков.')
    seen = set(map(tuple, support_x)) | set(map(tuple, excluded))
    candidates = [[(i >> j) & 1 for j in range(n)] for i in range(2 ** n)]
    candidates = [x for x in candidates if tuple(x) not in seen]
    if not candidates: return None
    row = dict(support_x=support_x, support_y=support_y, query_x=candidates)
    probabilities = np.asarray(predict(model, [row])[0])
    uncertainty = 4 * probabilities * (1 - probabilities)
    selected = int(uncertainty.argmax())
    return dict(input=candidates[selected], probability_one=float(probabilities[selected]),
                uncertainty=float(uncertainty[selected]), remaining_candidates=len(candidates),
                policy='written_maximum_predictive_uncertainty', needs_user_label=True)


def main():
    parser = argparse.ArgumentParser(description='Новая бинарная зависимость по примерам; память рассматривает до трёх существенных признаков.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--examples', help='Например: 00000000=0;10000000=1;01000000=1;11000000=0')
    group.add_argument('--examples-file', type=Path, help='JSON с support_x, support_y и необязательным query_x')
    parser.add_argument('--query', help='Новые входы через точку с запятой, например 10100000;01100000')
    parser.add_argument('--suggest', action='store_true', help='Предложить следующий вход, ответ на который должен дать пользователь')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT)
    args = parser.parse_args(); torch.set_num_threads(2)
    try:
        if args.examples_file:
            raw = json.loads(args.examples_file.read_text()); sx, sy = raw['support_x'], raw['support_y']
            qx = raw.get('query_x', [])
        else:
            pairs = [p.split('=') for p in args.examples.split(';') if p.strip()]
            sx = [parse_bits(x) for x,y in pairs]; sy = [int(y.strip()) for x,y in pairs]; qx = []
        if args.query: qx = [parse_bits(x) for x in args.query.split(';') if x.strip()]
        # Validate even if only --suggest was requested.
        encode(sx, sy, qx if qx else [sx[0]])
        if not qx and not args.suggest: raise ValueError('Добавьте --query или --suggest.')
        model, saved = load_model(args.checkpoint)
        if model.mode == 'pooled' and len(sx[0]) != 8: raise ValueError('Pooled-модель поддерживает ровно 8 признаков.')
        result = dict(model=dict(condition=model.condition, mode=model.mode, seed=saved['config']['seed']),
                      demonstration_count=len(sx), input_width=len(sx[0]),
                      scope='Бинарные правила с максимум тремя существенными признаками; искусственная память и нейронная оценка её частей.')
        if qx:
            row = dict(support_x=sx, support_y=sy, query_x=qx)
            probabilities = predict(model, [row])[0]; encoded = encode(sx, sy, qx)
            result['predictions'] = [dict(input=x, answer=int(p >= .5), probability_one=p,
                                          forced_by_all_consistent_sparse_rules=bool(f))
                                     for x,p,f in zip(qx, probabilities, encoded['forced'])]
            result['support_consistent_with_sparse_class'] = bool(encoded['consistent'].any())
            result['certainty_note'] = 'Флаг forced — отдельная написанная проверка в ограниченном классе правил; вероятность нейросети не является гарантией.'
        if args.suggest: result['next_question'] = suggest(model, sx, sy)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, IndexError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
