"""Development training reads TRAIN/VALIDATION only; final tests are a separate command."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import ConnectomeLM
from stage2_model import MemoryConnectome
from stage2_tasks import ROOT, load
from tasks import encode_rows, tokenize
from train import batch as legacy_batch
from evaluate import evaluate


def make_model(config,vocab):
    graph=dict(np.load(ROOT/'data/graph_256.npz'))
    if config['variant']=='legacy':
        return ConnectomeLM(graph,len(vocab),config['condition'],'rate',config['seed'])
    return MemoryConnectome(graph,len(vocab),config['condition'],
                             'lif' if config['variant']=='memory_lif' else 'rate',config['seed'],
                             memory=config['variant']!='gated',gated=True)


def prepare(rows,vocab):
    lookup={w:i for i,w in enumerate(vocab)}
    result=[]
    for row in rows:
        prompt=[1]+[lookup.get(w,4) for w in tokenize(row['prompt'])]+[2]
        answer=[lookup.get(w,4) for w in tokenize(row['answer'])]+[3]
        result.append((prompt,[1]+answer[:-1],answer))
    return result


def batch(encoded,indices,device):
    data=[encoded[int(i)] for i in indices]
    a=max(len(r[0]) for r in data);b=max(len(r[1]) for r in data)
    prompt=torch.zeros((len(data),a),dtype=torch.long,device=device)
    prev=torch.zeros((len(data),b),dtype=torch.long,device=device)
    target=torch.full_like(prev,-100)
    for i,(p,x,y) in enumerate(data):
        prompt[i,:len(p)]=torch.tensor(p,device=device)
        prev[i,:len(x)]=torch.tensor(x,device=device)
        target[i,:len(y)]=torch.tensor(y,device=device)
    return prompt,prev,target


def validation_panel(rows):
    rng=np.random.default_rng(58341)
    groups=defaultdict(list)
    for row in rows:groups[row['task']].append(row)
    panel=[]
    for task in sorted(groups):
        indices=rng.choice(len(groups[task]),min(120,len(groups[task])),replace=False)
        panel += [groups[task][i] for i in sorted(indices)]
    return panel


def train(config,output,resume_path=None):
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    vocab=json.loads((ROOT/'data/stage2/vocab.json').read_text())
    manifest=json.loads((ROOT/'data/stage2/manifest.json').read_text())
    records=load('train');panel=validation_panel(load('val'))
    model=make_model(config,vocab).to(config['device'])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=1e-4)
    rng=np.random.default_rng(config['seed']+272)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'history.jsonl').exists() and resume_path is None:
        raise FileExistsError(f'Existing run: {output}')
    best=-1;start=1
    if resume_path:
        c=torch.load(resume_path,map_location=config['device'],weights_only=True)
        assert c['vocab']==vocab and c['dataset_hashes']==manifest['hashes']
        model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer'])
        rng.bit_generator.state=c['numpy_rng'];torch.set_rng_state(c['torch_rng'])
        best=c['best_val'];start=c['step']+1
    groups=defaultdict(list)
    for i,row in enumerate(records):groups[row['task']].append(i)
    groups=list(groups.values())
    encoded=encode_rows(records,vocab) if config['variant']=='legacy' else prepare(records,vocab)
    t0=time.perf_counter();losses=[];max_grad=0
    for step in range(start,config['steps']+1):
        # Equal task sampling; only the learning rate has a warmup.
        indices=[int(rng.choice(groups[k%len(groups)])) for k in range(config['batch_size'])]
        rng.shuffle(indices)
        progress=(step-1)/max(config['steps']-1,1)
        lr=config['lr']*(0.15+0.85*(1+math.cos(math.pi*progress))/2)
        lr*=min(step/50,1)
        for group in optimizer.param_groups:group['lr']=lr
        model.train();optimizer.zero_grad(set_to_none=True)
        if config['variant']=='legacy':
            x,y=legacy_batch(encoded,indices,config['device']);logits=model(x)
        else:
            p,x,y=batch(encoded,indices,config['device']);logits=model(p,x)
        per_token=F.cross_entropy(logits.transpose(1,2),y,reduction='none',ignore_index=-100)
        loss=(per_token.sum(1)/(y!=-100).sum(1)).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
        loss.backward()
        core=model if config['variant']=='legacy' else model.core
        gains=getattr(core,'edge_log_gain',None)
        if gains is not None and gains.grad is not None:
            max_grad=max(max_grad,float(gains.grad.abs().max()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        optimizer.step();losses.append(float(loss.detach()))
        if step==1 or step%config['eval_every']==0 or step==config['steps']:
            metrics,_=evaluate(model,panel,vocab)
            improved=metrics['macro_exact']>best
            best=max(best,metrics['macro_exact'])
            entry=dict(step=step,loss=float(np.mean(losses)),validation=metrics,
                       seconds=time.perf_counter()-t0,learning_rate=lr)
            losses=[]
            with (output/'history.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                            config=config,vocab=vocab,dataset_hashes=manifest['hashes'],step=step,best_val=best,
                            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
            torch.save(checkpoint,output/'last.pt')
            if improved:torch.save(checkpoint,output/'best.pt')
            print(json.dumps(dict(run=output.name,**entry)),flush=True)
    checkpoint=torch.load(output/'best.pt',map_location=config['device'],weights_only=True)
    model.load_state_dict(checkpoint['model'])
    metrics,predictions=evaluate(model,load('val'),vocab)
    result=dict(config=config,validation=metrics,best_step=checkpoint['step'],
                parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                max_edge_gradient=max_grad,seconds=time.perf_counter()-t0,
                validation_panel_ids=[r['id'] for r in panel],dataset_hashes=manifest['hashes'],
                test_read=False)
    (output/'validation.json').write_text(json.dumps(result,indent=2))
    (output/'validation_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
    print('FINISHED '+json.dumps(dict(run=output.name,**{k:v for k,v in result.items() if k not in ['validation_panel_ids','dataset_hashes']})),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variants',default='memory_rate')
    p.add_argument('--conditions',default='fly')
    p.add_argument('--seeds',default='0')
    p.add_argument('--steps',type=int,default=1600)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--batch-size',type=int,default=48)
    p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',default='results/stage2/development')
    p.add_argument('--resume')
    args=p.parse_args()
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=True)
        cfg=checkpoint['config'];cfg.update(steps=args.steps,device=args.device,threads=args.threads)
        train(cfg,Path(args.resume).resolve().parent,args.resume)
        return
    for variant in args.variants.split(','):
        assert variant in ['legacy','gated','memory_rate','memory_lif']
        for condition in args.conditions.split(','):
            assert condition in ['fly','rewired','frozen','no_edges','gru']
            if variant=='legacy':assert condition in ['fly','rewired','frozen']
            for seed in map(int,args.seeds.split(',')):
                config=dict(variant=variant,condition=condition,seed=seed,steps=args.steps,lr=args.lr,
                            batch_size=args.batch_size,eval_every=args.eval_every,threads=args.threads,device=args.device)
                train(config,ROOT/args.output/f'{variant}_{condition}_seed{seed}')


if __name__=='__main__':main()
