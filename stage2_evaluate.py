"""Open the final benchmark only after all confirmation runs and the plan exist."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import torch
from stage2_tasks import ROOT,load
from stage2_train import make_model
from tasks import tokenize
from evaluate import evaluate


def detailed(model,rows,vocab):
    metrics,predictions=evaluate(model,rows,vocab)
    code=[r for r in predictions if r['task']=='code']
    if code:
        for r in code:
            try:
                tree=ast.parse(r['prediction'])
                names=[a.arg for a in tree.body[0].args.args]
                matches=names==tokenize(r['prompt'])[-2:]
            except (SyntaxError,AttributeError,IndexError):matches=False
            r['requested_signature']=matches
            r['strict_execution_pass']=bool(matches and r['code_pass'])
        metrics['code']['requested_signature']=sum(r['requested_signature'] for r in code)/len(code)
        metrics['code']['strict_execution_pass']=sum(r['strict_execution_pass'] for r in code)/len(code)
    subsets=defaultdict(list)
    for r in predictions:
        key=r['task']+'/'+r['operation']
        if r['task']=='memory':
            symbols=tokenize(r['prompt'])[1:]
            key+='/'+('digits' if all(x.isdigit() for x in symbols) else 'names')+f'/length{len(symbols)}'
        subsets[key].append(int(r['exact']))
    metrics['breakdown']={k:dict(n=len(v),exact=sum(v)/len(v)) for k,v in subsets.items()}
    return metrics,predictions


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/stage2/confirmation')
    args=p.parse_args();folder=ROOT/args.root
    plan=json.loads((folder/'plan.json').read_text())
    assert plan['frozen_before_final_test']
    for run in plan['runs']:
        path=folder/run
        assert (path/'validation.json').exists(),f'Unfinished run: {run}'
        cfg=json.loads((path/'validation.json').read_text())['config']
        assert cfg['steps']==plan['steps']
    marker=folder/'final_test_opened.json'
    if marker.exists():raise FileExistsError('Final evaluation already started; inspect results before recovery')
    marker.write_text(json.dumps(dict(opened_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                     plan_sha256=hashlib.sha256((folder/'plan.json').read_bytes()).hexdigest()),indent=2))
    torch.set_num_threads(2)
    vocab=json.loads((ROOT/'data/stage2/vocab.json').read_text())
    manifest=json.loads((ROOT/'data/stage2/manifest.json').read_text())
    splits={name:load(name) for name in ['test','phrasing','longer']}
    for split in splits:
        assert hashlib.sha256((ROOT/f'data/stage2/{split}.jsonl').read_bytes()).hexdigest()==manifest['hashes'][split]
    for run in plan['runs']:
        path=folder/run
        checkpoint=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
        cfg=checkpoint['config'];model=make_model(cfg,vocab)
        model.load_state_dict(checkpoint['model'])
        output=dict(config=cfg,best_step=checkpoint['step'],
                    checkpoint_sha256=hashlib.sha256((path/'best.pt').read_bytes()).hexdigest())
        for split,rows in splits.items():
            metrics,predictions=detailed(model,rows,vocab)
            output[split]=metrics
            (path/f'{split}_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
        if cfg['condition']!='gru':
            model.disable_edges=True
            output['test_without_edges'],_=detailed(model,splits['test'],vocab)
            model.disable_edges=False
        if cfg['variant'] in ['memory_rate','memory_lif']:
            model.disable_memory=True
            output['test_without_input_memory'],_=detailed(model,splits['test'],vocab)
        (path/'final_metrics.json').write_text(json.dumps(output,indent=2))
        print(json.dumps(dict(run=run,test={k:v for k,v in output['test'].items() if k!='breakdown'},
                             longer=output['longer']['memory'])),flush=True)


if __name__=='__main__':main()
