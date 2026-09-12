"""Data integrity, cache parity, source-copy causality, and graph gradients."""
import ast
import hashlib
import json
import numpy as np
import torch
from stage6_tasks import ROOT,DATA,load,HELD_PAIRS,pairs,resolved_trace
from stage6_tokens import VOCAB,BOS,SEP,PAD,EOS,encode,prompt_ids
from stage6_train import batch_tensors,prepare
from stage6_model import Reasoner


def independent(row):
    spec=row['spec']
    if row['family']=='sum':
        answer=sum(spec['values']);steps=[str(sum(spec['values'][:i])) for i in range(1,len(spec['values'])+1)]
        return str(answer),','.join(steps)
    if row['family']=='logic':
        # Explicit stack rather than recursive generator execution.
        pending=[(spec['tree'],False)];values=[];steps=[]
        while pending:
            node,ready=pending.pop()
            if isinstance(node,int):values.append(node);continue
            if not ready:
                pending.append((node,True));pending.extend((c,False) for c in reversed(node[1:]));continue
            op=node[0];n=len(node)-1;args=values[-n:];del values[-n:]
            if op=='not':v=int(not args[0])
            elif op=='and':v=int(all(args))
            elif op=='or':v=int(any(args))
            elif op=='xor':v=int(args[0]!=args[1])
            else:raise ValueError(op)
            values.append(v);steps.append(f'{op}={v}')
        return str(values[0]),';'.join(steps)
    env=dict(spec['initial']);steps=[]
    for dst,left,op,right in spec['instructions']:
        a=env.get(left,left);b=env.get(right,right)
        operation={'+':lambda x,y:x+y,'-':lambda x,y:x-y,'*':lambda x,y:x*y}[op]
        env[dst]=operation(a,b)-10*(operation(a,b)//10);steps.append(f'{dst}={env[dst]}')
    return str(env[spec['query']]),';'.join(steps)


def check():
    torch.set_num_threads(2);manifest=json.loads((DATA/'manifest.json').read_text());seen=set();count=0
    for split in manifest['counts']:
        assert hashlib.sha256((DATA/f'{split}.jsonl').read_bytes()).hexdigest()==manifest['hashes'][split]
        rows=load(split)
        for r in rows:
            assert r['prompt'] not in seen;seen.add(r['prompt']);count+=1
            answer,trace=independent(r);assert (answer,trace)==(r['answer'],r['trace'])
            encode(r['prompt']);encode(r['trace']+'|'+r['answer'])
            # Resolved trace values and operand bindings checked without calling
            # the generator's recursive evaluator or arithmetic functions.
            rich=resolved_trace(r).split(';');encode(';'.join(rich))
            if r['family']=='sum':
                running=0
                for term,text in zip(r['spec']['values'],rich):
                    lhs,rhs=text.split('=');a,b=map(int,lhs.split('+'))
                    assert a==running and b==term and int(rhs)==a+b;running=int(rhs)
                assert len(rich)==len(r['spec']['values']) and str(running)==answer
            elif r['family']=='code':
                state=dict(r['spec']['initial'])
                for inst,text in zip(r['spec']['instructions'],rich):
                    dst,left,op,right=inst
                    a=state.get(left,left);b=state.get(right,right)
                    value={'+':a+b,'-':a-b,'*':a*b}[op]%10
                    assert text==f'{dst}:{a}{op}{b}={value}';state[dst]=value
                assert len(rich)==len(r['spec']['instructions']) and str(state[r['spec']['query']])==answer
            else:
                assert len(rich)==len(trace.split(';'))
                for text,simple in zip(rich,trace.split(';')):
                    lhs,rhs=text.split('=');op,arguments=lhs[:-1].split('(');vs=list(map(int,arguments.split(',')))
                    value=(int(not vs[0]) if op=='not' else int(all(vs)) if op=='and' else int(any(vs)) if op=='or' else int(vs[0]!=vs[1]))
                    assert int(rhs)==value and simple==op+'='+rhs
            if split in ['train','val','test']:
                if r['family']=='logic':assert not pairs(r['spec']['tree'])&HELD_PAIRS
                if r['family']=='code':
                    assert all(not(isinstance(a,str) and a==b) for d,a,op,b in r['spec']['instructions'])
        if split=='composition':assert all(pairs(r['spec']['tree'])&HELD_PAIRS for r in rows)
    val=load('val')[:4];prepared=[prepare(r,'trace') for r in val]
    result={}
    # Compare every generated-position probability with a recomputed full prefix,
    # using unequal prompt lengths and left padding, both with and without copy.
    for condition,copy in [('mlp',False),('mlp',True),('fly',True)]:
        model=Reasoner(condition,copy,d=96,layers=3,heads=4);model.eval()
        prompts=[prompt_ids(r['prompt']) for r in val];width=max(map(len,prompts))
        ids=torch.full((len(val),width),PAD,dtype=torch.long);valid=torch.zeros_like(ids,dtype=torch.bool)
        for i,p in enumerate(prompts):ids[i,-len(p):]=torch.tensor(p);valid[i,-len(p):]=True
        pm=valid&(ids!=BOS)&(ids!=SEP)
        with torch.no_grad():
            p,cache=model(ids,valid,pm);largest=0
            for step in range(5):
                next_ids=torch.full((len(val),1),encode('1')[0],dtype=torch.long)
                cached,cache=model(next_ids,torch.ones_like(next_ids,dtype=torch.bool),pm,cache)
                ids=torch.cat([ids,next_ids],1);valid=torch.cat([valid,torch.ones_like(next_ids,dtype=torch.bool)],1)
                full_pm=torch.cat([pm,torch.zeros((len(val),step+1),dtype=torch.bool)],1)
                full,_=model(ids,valid,full_pm)
                largest=max(largest,float((cached[:,-1]-full[:,-1]).abs().max()))
            assert largest<3e-6,(condition,copy,largest)
        result[f'{condition}_copy{copy}_cache_max_difference']=largest
        # Replacing future completion tokens cannot affect the first prediction.
        x,y,v,mask=batch_tensors(prepared);prefix=prepared[0][1]
        with torch.no_grad():
            original,_=model(x,v,mask);altered=x.clone();altered[0,prefix:]=encode('9')[0]
            changed,_=model(altered,v,mask)
            assert torch.allclose(original[0,prefix-1],changed[0,prefix-1],atol=2e-6)
        if condition=='fly':
            probs,_=model(x,v,mask)
            loss=torch.nn.functional.nll_loss(probs.log().transpose(1,2),y,ignore_index=-100);loss.backward()
            grads=[b.ff.core.edge_log_gain.grad for b in model.blocks]
            assert all(g is not None and torch.isfinite(g).all() and g.abs().max()>0 for g in grads)
    for name in ['stage6_model.py','stage6_engine.py','stage6_tokens.py']:
        tree=ast.parse((ROOT/name).read_text())
        assert 'stage6_tasks' not in [n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    result.update(checked_examples=count,all_prompt_strings_disjoint=True,independent_answers_and_traces=True,
                  heldout_structure_verified=True,no_future_target_leakage=True,copy_uses_prompt_only=True,
                  graph_gradients_present=True,inference_has_no_teacher_or_program_parser=True,resolved_operands_and_results_checked=True,vocabulary=len(VOCAB),
                  parameters={f'{c}_copy{copy}':sum(p.numel() for p in Reasoner(c,copy,d=96,layers=3).parameters() if p.requires_grad)
                              for c,copy in [('mlp',False),('mlp',True),('fly',True),('rewired',True),('no_edges',True)]})
    out=ROOT/'results/stage6';out.mkdir(parents=True,exist_ok=True);(out/'checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':check()
