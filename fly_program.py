"""Run a learned tape controller, or plan from a supplied start and goal."""
import argparse
import json
from pathlib import Path
import torch
from stage4_model import load_model
from stage4_engine import NeuralExecutor
from stage4_planner import GoalPlanner

ROOT=Path(__file__).resolve().parent
WORDS={'повтори':'copy','разверни':'reverse','чётные':'even','четные':'even','нечётные':'odd','нечетные':'odd','+1':'inc','-1':'dec'}


def digits(text):
    text=text.replace(' ','').replace(',','')
    if any(c not in '0123456789' for c in text):raise ValueError('Use decimal digits separated by spaces')
    return list(map(int,text))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--program',help='Operation names separated by |, e.g. разверни | чётные | +1')
    p.add_argument('--data',required=True);p.add_argument('--goal',help='Ask the search planner to find a program')
    p.add_argument('--checkpoint',default='results/stage4/confirmation/reactive_fly_seed0/best.pt')
    p.add_argument('--trace',action='store_true');p.add_argument('--direct',action='store_true',help='Disable finite observation cache in --program mode')
    args=p.parse_args()
    if (args.program is None)==(args.goal is None):p.error('Supply exactly one of --program or --goal')
    torch.set_num_threads(2);path=Path(args.checkpoint)
    if not path.is_absolute():path=ROOT/path
    model,_=load_model(path)
    try:
        data=digits(args.data)
        if args.goal is not None:
            result=GoalPlanner(model).solve(data,digits(args.goal))
            result['method']='engineered_search_with_learned_transitions'
        else:
            program=[WORDS.get(x.strip().lower(),x.strip().lower()) for x in args.program.split('|')]
            result=NeuralExecutor(model,compiled=False if args.direct else None).run([dict(program=program,data=data)],keep_trace=args.trace)[0]
            result['program']=program;result['method']='learned_low_level_actions'
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except ValueError as exc:p.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
