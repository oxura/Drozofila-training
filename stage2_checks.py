"""Causal decoding, padding, graph constraints, and strict code-contract checks."""
import json
import numpy as np
import torch
from stage2_tasks import ROOT,create
from stage2_train import make_model,prepare,batch
from stage2_evaluate import detailed
from tasks import tokenize


def main():
    torch.set_num_threads(2)
    rows,vocab=create()
    examples=[next(r for r in rows['train'] if r['task']==t) for t in ['arithmetic','memory','code']]
    p,x,y=batch(prepare(examples,vocab),[0,1,2],'cpu')
    results=[]
    for variant,condition in [('gated','fly'),('memory_rate','fly'),('memory_lif','fly'),
                               ('memory_rate','rewired'),('memory_rate','frozen'),
                               ('memory_rate','gru'),('memory_rate','no_edges')]:
        m=make_model(dict(variant=variant,condition=condition,seed=0),vocab)
        output=m(p,x);loss=torch.nn.functional.cross_entropy(output.transpose(1,2),y)
        loss.backward();assert torch.isfinite(loss)
        first=output[:,0].detach().clone();first[:,[0,1,2,4]]=-torch.inf
        assert torch.equal(m.generate(p,max_tokens=1)[:,0],first.argmax(-1))
        assert torch.equal(m.generate(p,max_tokens=3),m.generate(torch.nn.functional.pad(p,(0,3)),max_tokens=3))
        # Changing a future answer token cannot change earlier output logits.
        altered=x.clone();altered[:,-1]=5
        assert torch.equal(m(p,altered)[:,:-1],output[:,:-1])
        edge_grad=0
        if condition not in ['gru','no_edges','frozen']:
            edge_grad=float(m.core.edge_log_gain.grad.abs().max())
            assert edge_grad>0
            matrix=m.matrix().detach()
            assert int(torch.count_nonzero(matrix))==7514
            assert torch.equal(matrix[m.core.post,m.core.pre].sign(),m.core.sign)
        if condition=='no_edges':assert torch.count_nonzero(m.matrix())==0
        results.append(dict(variant=variant,condition=condition,loss=float(loss.detach()),edge_gradient=edge_grad))
    # An alpha-renamed function must not pass the requested signature check.
    class Fake(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
        def generate(self,ids):
            lookup={w:i for i,w in enumerate(vocab)}
            seq=[lookup[w] for w in tokenize('def f ( a , b ) : return a + b')]+[3]
            return torch.tensor([seq]*len(ids))
    row=dict(task='code',operation='сложи',prompt='код сложи x y',answer='def f ( x , y ) : return x + y',group='check',id='check')
    metrics,_=detailed(Fake(),[row],vocab)
    assert metrics['code']['execution_pass']==1 and metrics['code']['strict_execution_pass']==0
    (ROOT/'results/stage2/checks.json').write_text(json.dumps(dict(status='passed',models=results),indent=2))
    print('Stage-2 checks passed')


if __name__=='__main__':main()
