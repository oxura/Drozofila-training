"""Check environment semantics, dataset splits and absence of an inference oracle."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import torch
from stage4_tasks import ROOT,load,trace,functional_oracle,has_heldout_pair
from stage4_env import TapeMachine,OPS,RIGHT,NEXT,HALT
from stage4_model import TapePolicy
from stage4_engine import NeuralExecutor


def check():
    torch.set_num_threads(2);result={};seen=set();observations={}
    for split in ['train','val','test','composition','long_tape','long_program']:
        rows=load(split)
        assert sum(not r['answer'] for r in rows)==len(rows)//4
        for row in rows:
            key=(tuple(row['program']),tuple(row['data']));assert key not in seen;seen.add(key)
            if split in ['train','val','test','long_tape']:assert not has_heldout_pair(row['program'])
            if split=='composition':assert has_heldout_pair(row['program'])
            # Structural and oracle integrity, not model evaluation / selection.
            assert row['answer']==functional_oracle(row['program'],row['data'])
            if split=='train':
                obs,actions=trace(row['program'],row['data'])
                for x,y in zip(obs,actions):
                    if tuple(x) in observations:assert observations[tuple(x)]==y
                    observations[tuple(x)]=y
    result['disjoint_input_program_pairs']=True;result['heldout_adjacencies_verified']=True
    result['empty_output_baseline']=.25;result['training_observation_action_rules']=len(observations)
    # Generic actions have the same effects under every high-level instruction.
    effects=[]
    for op in OPS:
        machine=TapeMachine([op],[8,3])
        for action in [RIGHT,7,RIGHT,4,NEXT,HALT]:machine.step(action)
        effects.append(machine.result())
    assert all(e==effects[0] for e in effects)
    assert effects[0]['answer']==[5,2]
    result['environment_actions_independent_of_task']=True
    policy=TapePolicy();x=torch.tensor(list(observations)[:64]);y=torch.tensor([observations[tuple(r)] for r in x.tolist()])
    logits,_=policy(x);torch.nn.functional.cross_entropy(logits,y).backward()
    assert torch.isfinite(policy.core.edge_log_gain.grad).all() and policy.core.edge_log_gain.grad.abs().max()>0
    result['edge_gradient_present']=True
    rows=[dict(program=['copy','reverse'],data=[1,2,3]),dict(program=['even','inc'],data=[8,1,4])]
    raw=NeuralExecutor(policy,compiled=False).run(rows);cached=NeuralExecutor(policy,compiled=True).run(rows)
    assert raw==cached;result['random_weight_direct_cache_parity']=True
    # Always halt cannot succeed without executing the program.
    with torch.no_grad():
        for p in policy.parameters():p.zero_()
        policy.output.bias[HALT]=10
    bad=NeuralExecutor(policy).run([dict(program=['copy'],data=[1,2])])[0]
    assert not bad['program_completed'];result['no_hidden_teacher_fallback']=True
    result['parameters']={c:sum(p.numel() for p in TapePolicy(c).parameters() if p.requires_grad)
                          for c in ['fly','rewired','frozen','no_edges','mlp','gru']}
    directory=ROOT/'results/stage4';directory.mkdir(parents=True,exist_ok=True)
    (directory/'checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':check()
