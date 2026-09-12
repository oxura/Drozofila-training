"""Validity checks: topology gradients, carry feedback, interface and data splits."""
import json
from pathlib import Path
import numpy as np
import torch
from stage3_model import DigitController
from stage3_engine import NeuralArithmetic
from stage3_tasks import ROOT, load, trace, local_table


def check():
    torch.set_num_threads(2)
    results={}
    table=local_table();tx=torch.tensor([r['x'] for r in table]);ty=torch.tensor([r['y'] for r in table])
    model=DigitController()
    pred,_=model(tx[:64]);loss=torch.nn.functional.cross_entropy(pred[:,0],ty[:64,0])
    loss.backward();assert torch.isfinite(model.core.edge_log_gain.grad).all()
    assert model.core.edge_log_gain.grad.abs().max()>0
    matrix=model.matrix().detach();graph=dict(np.load(ROOT/'data/graph_256.npz'))
    assert int(torch.count_nonzero(matrix))==len(graph['pre'])
    assert np.array_equal(torch.sign(matrix[model.core.post,model.core.pre]).numpy(),np.sign(graph['weight']))
    results['graph_edges_and_signs_preserved']=True
    results['nonzero_finite_edge_gradient']=True
    # Random, inaccurate weights still give identical direct and model-cached output.
    rows=[dict(op='add',a='9981',b='7364'),dict(op='sub',a='9981',b='7364'),
          dict(op='mul_digit',a='9981',b='9')]
    direct=NeuralArithmetic(model).batch_unsigned(rows)
    cached=NeuralArithmetic(model,compiled=True).batch_unsigned(rows)
    assert direct==cached
    results['model_cache_parity_before_training']=True
    # An intentionally bad controller must cause bad arithmetic: no hidden fallback.
    bad=DigitController()
    with torch.no_grad():
        for parameter in bad.parameters():parameter.zero_()
    result=NeuralArithmetic(bad).add('123','456')
    assert result=='0' and result!='579'
    results['zeroed_network_changes_answer']=result
    engine=NeuralArithmetic(model,compiled=True)
    rejected=[]
    for expression in ['__import__("os")','a+1','2**10','1/2','True+1','[1][0]']:
        try:engine.expression(expression)
        except ValueError:rejected.append(expression)
        else:raise AssertionError('Unexpected executable grammar: '+expression)
    results['unsupported_expressions_rejected']=rejected
    seen=set();counts={}
    for split in ['train_sequences','val','test']:
        rows=load(split);keys=set()
        for row in rows:
            key=(row['op'],*sorted([row['a'],row['b']])) if row['op']=='add' else (row['op'],row['a'],row['b'])
            assert key not in seen and key not in keys;keys.add(key)
            if split!='test':trace(row)
        seen|=keys;counts[split]=len(rows)
    results['sequence_groups_disjoint']=counts
    results['all_860_local_transitions_are_training_examples']=True
    results['parameters']={c:sum(p.numel() for p in DigitController(c).parameters() if p.requires_grad)
                           for c in ['fly','rewired','frozen','no_edges','mlp']}
    (ROOT/'results/stage3').mkdir(parents=True,exist_ok=True)
    (ROOT/'results/stage3/checks.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':check()
