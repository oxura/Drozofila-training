"""Teacher trajectories and independent functional oracle. Not imported by inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
from stage4_env import TapeMachine,OPS,LEFT,RIGHT,NEXT,HALT,START

ROOT=Path(__file__).resolve().parent
SEED=2026091504
HELDOUT_PAIRS=(('reverse','even'),('even','inc'),('inc','reverse'),
               ('odd','dec'),('dec','copy'),('copy','odd'))


def functional_oracle(program,data):
    data=list(data)
    for op in program:
        if op=='copy':data=data.copy()
        elif op=='reverse':data=data[::-1]
        elif op=='even':data=[d for d in data if d%2==0]
        elif op=='odd':data=[d for d in data if d%2==1]
        elif op=='inc':data=[(d+1)%10 for d in data]
        elif op=='dec':data=[(d-1)%10 for d in data]
        else:raise ValueError(op)
    return data


def teacher_action(machine):
    op,cell,last=machine.observation()
    if op==len(OPS):return HALT
    if OPS[op]=='reverse':
        if cell==10:return NEXT if last==LEFT else RIGHT
        if cell==11:return LEFT
        if last==RIGHT:return RIGHT
        if 2<=last<=11:return LEFT
        return cell+2
    if cell==10:return RIGHT
    if cell==11:return NEXT
    if 2<=last<=11:return RIGHT
    name=OPS[op]
    if name=='even' and cell%2==1:return RIGHT
    if name=='odd' and cell%2==0:return RIGHT
    if name=='inc':cell=(cell+1)%10
    elif name=='dec':cell=(cell-1)%10
    return cell+2


def trace(program,data):
    machine=TapeMachine(program,data);observations=[];actions=[]
    while not machine.done:
        observations.append(machine.observation());a=teacher_action(machine);actions.append(a);machine.step(a)
    result=machine.result()
    assert result['halted'] and result['program_completed'] and result['answer']==functional_oracle(program,data)
    return observations,actions


def has_heldout_pair(program):
    return any(pair in HELDOUT_PAIRS for pair in zip(program,program[1:]))


def generate(rng, count, lengths, depths, seen, name, pair_mode='exclude'):
    rows=[]
    while len(rows)<count:
        n=int(rng.choice(lengths));k=int(rng.choice(depths))
        program=[OPS[int(i)] for i in rng.integers(0,len(OPS),k)]
        held=has_heldout_pair(program)
        if pair_mode=='exclude' and held or pair_mode=='require' and not held:continue
        data=list(map(int,rng.integers(0,10,n)));key=(tuple(program),tuple(data))
        if key in seen:continue
        seen.add(key)
        rows.append(dict(id=f'{name}_{len(rows)}',program=program,data=data,
                         answer=functional_oracle(program,data),length=n,depth=k))
    return rows


def prepare():
    dest=ROOT/'data/stage4';dest.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED);seen=set();sets={}
    sets['train']=generate(rng,3600,range(7),[1,2,3],seen,'train')
    sets['val']=generate(rng,240,range(7),[1,2,3],seen,'val')
    sets['test']=generate(rng,240,[2,3,4,5,6],[1,2,3],seen,'test')
    sets['composition']=generate(rng,240,[2,3,4,5,6],[2,3],seen,'composition','require')
    sets['long_tape']=sum((generate(rng,96,[n],[1,2,3],seen,f'tape{n}') for n in [16,64,256]),[])
    sets['long_program']=sum((generate(rng,96,[2,3,4,5,6],[n],seen,f'program{n}','any') for n in [4,8,16]),[])
    stress=[]
    for program in [['reverse'],['even'],['odd'],['inc','reverse','even'],['reverse','dec','odd'],list(OPS)]:
        for data in [[],[9],[0]*64,list(range(10))*20]:
            stress.append(dict(id=f'stress_{len(stress)}',program=program,data=data,
                               answer=functional_oracle(program,data),length=len(data),depth=len(program)))
    sets['stress']=stress
    hashes={};coverage=set();actions_total=0
    for row in sets['train']:
        obs,act=trace(row['program'],row['data']);coverage.update(map(tuple,obs));actions_total+=len(act)
    for name,rows in sets.items():
        text=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows)
        path=dest/f'{name}.jsonl'
        if path.exists() and path.read_text()!=text:raise FileExistsError(str(path))
        path.write_text(text);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=dict(seed=SEED,hashes=hashes,counts={k:len(v) for k,v in sets.items()},
                  heldout_adjacent_pairs=HELDOUT_PAIRS,train_lengths=list(range(7)),train_depths=[1,2,3],
                  training_action_steps=actions_total,unique_training_observations=len(coverage),
                  learned=['pointer movement','digit to write','whether to filter','stage transition','halt'],
                  given=['high-level program','meaning of primitive actions','tape and append-only output buffer',
                         'current instruction and current cell','previous predicted action in history-enabled modes',
                         'expert action labels during training'],
                  not_demonstrated=['high-level goal planning','new primitive rules','open-language understanding','general intelligence'])
    path=dest/'manifest.json';text=json.dumps(manifest,indent=2)+'\n'
    if path.exists() and path.read_text()!=text:raise FileExistsError(str(path))
    path.write_text(text);print(json.dumps(manifest,indent=2))


def load(split):
    return [json.loads(line) for line in (ROOT/'data/stage4'/f'{split}.jsonl').read_text().splitlines()]


if __name__=='__main__':prepare()
