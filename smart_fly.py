"""Use learned arithmetic weights. Russian command grammar is hand-written."""
import argparse
import json
import re
from pathlib import Path
import torch
from stage3_engine import NeuralArithmetic
from stage3_model import load_model

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('query',nargs='?')
    p.add_argument('--checkpoint',default='results/stage3/confirmation/discrete_fly_seed0/best.pt')
    p.add_argument('--compiled',action='store_true',help='Cache network predictions for all finite local states')
    p.add_argument('--interactive',action='store_true')
    p.add_argument('--json',action='store_true')
    p.add_argument('--verify',action='store_true',help='Check a simple arithmetic command through learned inverse operations; not a proof')
    p.add_argument('--language-checkpoint',default='results/stage2/confirmation/memory_rate_fly_seed0/best.pt')
    args=p.parse_args()
    if not args.query and not args.interactive:p.error('Provide a query or --interactive')
    path=Path(args.checkpoint)
    if not path.is_absolute():path=ROOT/path
    torch.set_num_threads(2)
    model,ckpt=load_model(path);engine=NeuralArithmetic(model,compiled=args.compiled)
    language=None
    while True:
        try:
            query=input('> ') if args.interactive else args.query
            if query.strip().lower() in ['exit','quit','выход']:break
            checks={};kind='learned_arithmetic';used_checkpoint=path
            if re.match(r'^\s*(код|повтори|разверни)\b',query.lower()):
                from infer import load_checkpoint,answer
                language_path=Path(args.language_checkpoint)
                if not language_path.is_absolute():language_path=ROOT/language_path
                if language is None:language=load_checkpoint(language_path)
                result=answer(language[0],language[1],query)['answer']
                kind='stage2_sequence_model';used_checkpoint=language_path
            else:
                result=engine.command(query)
                match=re.fullmatch(r'\s*(сложи|вычти|умножь)\s+(-?[0-9]+)\s+(-?[0-9]+)\s*',query.lower())
                if args.verify and match:
                    from stage3_engine import canonical
                    op,a,b=match.groups()
                    if op=='сложи':
                        checks['reversed_operands']=engine.add(b,a)==result
                        checks['inverse_subtraction']=engine.subtract(result,a)==canonical(b)
                    elif op=='вычти':checks['inverse_addition']=engine.add(result,b)==canonical(a)
                    else:
                        checks['reversed_operands']=engine.multiply(b,a)==result
                        previous=engine.multiply(a,engine.subtract(b,'1'))
                        checks['adjacent_product']=engine.add(previous,a)==result
            payload=dict(query=query,answer=result,kind=kind,checkpoint=str(used_checkpoint),
                         compiled=args.compiled if kind=='learned_arithmetic' else False,
                         parser='hand_written',trained_during_query=False,
                         consistency_checks=checks,consistency_is_proof=False)
            print(json.dumps(payload,ensure_ascii=False) if args.json else result)
            if checks and not args.json:
                print('Проверки согласованности:', 'пройдены' if all(checks.values()) else 'есть расхождения',
                      '(это не доказательство правильности)')
        except (EOFError,KeyboardInterrupt):break
        except (ValueError,SyntaxError,RecursionError) as exc:
            if not args.interactive:p.exit(2,f'Ошибка: {exc}\n')
            print(f'Ошибка: {exc}')
        if not args.interactive:break


if __name__=='__main__':main()
