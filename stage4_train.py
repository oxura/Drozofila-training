"""Supervised action learning. Loads only train and validation, never final sets."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from stage4_model import make_model
from stage4_engine import NeuralExecutor
from stage4_tasks import ROOT,load,trace


@torch.no_grad()
def evaluate_rows(model,rows,batch_size=32,compiled=None,keep_trace=False):
    engine=NeuralExecutor(model,compiled=compiled);predictions=[]
    # Similar lengths reduce padding/finished-agent work for stateful controllers.
    ordered=sorted(rows,key=lambda r:(len(r['data'])*len(r['program']),r['id']))
    for start in range(0,len(ordered),batch_size):
        batch=ordered[start:start+batch_size];results=engine.run(batch,keep_trace=keep_trace)
        for row,result in zip(batch,results):
            correct=result['halted'] and result['program_completed'] and result['answer']==row['answer']
            predictions.append(dict(**row,prediction=result,correct=correct))
    nonempty=[r for r in predictions if r['answer']]
    metrics=dict(exact=float(np.mean([r['correct'] for r in predictions])),
                 nonempty_exact=float(np.mean([r['correct'] for r in nonempty])) if nonempty else None,
                 nonempty_count=len(nonempty),count=len(rows),
                 empty_answer_baseline=float(np.mean([not r['answer'] for r in predictions])),
                 completed_and_halted=float(np.mean([r['prediction']['halted'] and r['prediction']['program_completed'] for r in predictions])),
                 errors=dict(Counter(r['prediction']['error'] for r in predictions if r['prediction']['error'])),
                 mean_steps=float(np.mean([r['prediction']['steps'] for r in predictions])))
    return metrics,predictions


def train(config,out,resume=None):
    torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'history.jsonl').exists() and resume is None:raise FileExistsError(str(out))
    manifest=json.loads((ROOT/'data/stage4/manifest.json').read_text());rows=load('train');val=load('val')
    trajectories=[trace(r['program'],r['data']) for r in rows]
    flat_x=torch.tensor([x for obs,actions in trajectories for x in obs]);flat_y=torch.tensor([a for obs,actions in trajectories for a in actions])
    class_groups=[np.flatnonzero(flat_y.numpy()==i) for i in range(14)]
    assert all(len(g) for g in class_groups)
    model=make_model(config);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=1e-4)
    rng=np.random.default_rng(config['seed']+40404);steps=config['steps'];start=1;best=(-1,float('-inf'));tokens=0
    if resume:
        old=torch.load(resume,map_location='cpu',weights_only=True)
        assert old['dataset_hashes']==manifest['hashes']
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
        rng.bit_generator.state=old['numpy_rng'];torch.set_rng_state(old['torch_rng'])
        start=old['step']+1;best=tuple(old['best_score']);tokens=old['training_action_steps']
        if start>steps:raise ValueError('New total must exceed the saved step')
    t0=time.perf_counter();losses=[];max_grad=0
    for step in range(start,steps+1):
        lr=config['lr']*(.15+.85*(1+math.cos(math.pi*(step-1)/max(1,steps-1)))/2)
        for group in optimizer.param_groups:group['lr']=lr
        model.train();optimizer.zero_grad(set_to_none=True)
        if config['mode'] in ['reactive','blind']:
            labels=rng.integers(14,size=config['batch_size'])
            indices=np.array([rng.choice(class_groups[i]) for i in labels])
            logits,_=model(flat_x[indices]);loss=F.cross_entropy(logits,flat_y[indices]);tokens+=len(indices)
        else:
            ids=rng.integers(len(trajectories),size=config['sequence_batch_size'])
            batch=[trajectories[i] for i in ids];width=max(len(b[1]) for b in batch)
            x=torch.tensor([[[6,10,14]]*width for _ in batch]);y=torch.full((len(batch),width),-100,dtype=torch.long)
            for i,(obs,actions) in enumerate(batch):
                x[i,:len(obs)]=torch.tensor(obs);y[i,:len(actions)]=torch.tensor(actions)
            state=None;matrix=model.matrix();outputs=[]
            for pos in range(width):
                logits,state=model(x[:,pos],state,matrix);outputs.append(logits)
            logits=torch.stack(outputs,dim=1);per_token=F.cross_entropy(logits.transpose(1,2),y,reduction='none',ignore_index=-100)
            loss=(per_token.sum(1)/(y!=-100).sum(1)).mean();tokens+=int((y!=-100).sum())
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
        loss.backward()
        if hasattr(model,'core') and model.core.edge_log_gain.grad is not None:
            max_grad=max(max_grad,float(model.core.edge_log_gain.grad.abs().max()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        if step==1 or step%config['eval_every']==0 or step==steps:
            model.eval();metrics,_=evaluate_rows(model,val)
            loss_mean=float(np.mean(losses));losses=[];score=(metrics['exact'],-loss_mean)
            improved=score>best;best=max(best,score)
            entry=dict(step=step,loss=loss_mean,validation=metrics,seconds=time.perf_counter()-t0,
                       training_action_steps=tokens,learning_rate=lr)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            ckpt=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,step=step,
                      best_score=list(best),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                      dataset_hashes=manifest['hashes'],training_action_steps=tokens)
            torch.save(ckpt,out/'last.pt')
            if improved:torch.save(ckpt,out/'best.pt')
            print(json.dumps(dict(run=out.name,**entry)),flush=True)
    ckpt=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ckpt['model']);model.eval()
    metrics,predictions=evaluate_rows(model,val)
    result=dict(config=config,validation=metrics,best_step=ckpt['step'],seconds=time.perf_counter()-t0,
                parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                max_edge_gradient=max_grad,training_action_steps=tokens,dataset_hashes=manifest['hashes'],final_test_read=False)
    (out/'validation.json').write_text(json.dumps(result,indent=2))
    (out/'validation_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    print('FINISHED '+json.dumps(dict(run=out.name,**result)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['reactive','recurrent','hidden','blind'],default='reactive')
    p.add_argument('--conditions',default='fly');p.add_argument('--seeds',default='0')
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--ticks',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=112);p.add_argument('--sequence-batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=.003);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--eval-every',type=int,default=200);p.add_argument('--output',default='results/stage4/development')
    p.add_argument('--resume');args=p.parse_args()
    if args.resume:
        path=Path(args.resume).resolve();old=torch.load(path,map_location='cpu',weights_only=True)
        cfg=old['config'];cfg.update(steps=args.steps,threads=args.threads,eval_every=args.eval_every)
        train(cfg,path.parent,path);return
    for condition in args.conditions.split(','):
        assert condition in ['fly','rewired','frozen','no_edges','mlp','gru']
        for seed in map(int,args.seeds.split(',')):
            cfg=dict(mode=args.mode,condition=condition,seed=seed,steps=args.steps,ticks=args.ticks,
                     batch_size=args.batch_size,sequence_batch_size=args.sequence_batch_size,lr=args.lr,
                     threads=args.threads,eval_every=args.eval_every)
            train(cfg,ROOT/args.output/f'{args.mode}_{condition}_seed{seed}')


if __name__=='__main__':main()
