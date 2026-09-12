"""Execute only the model's actions. No teacher or functional oracle is imported."""
import torch
from stage4_env import TapeMachine,ACTIONS


class NeuralExecutor:
    def __init__(self,model,compiled=None):
        self.model=model.eval();self.table=None
        if compiled is None:compiled=model.mode in ['reactive','blind']
        if compiled:
            if model.mode not in ['reactive','blind']:raise ValueError('Hidden-state policies cannot use a finite observation cache')
            observations=torch.cartesian_prod(torch.arange(7),torch.arange(12),torch.arange(15))
            with torch.no_grad():logits,_=model(observations)
            self.table=logits.argmax(-1).reshape(7,12,15).tolist()

    @torch.no_grad()
    def run(self,rows,keep_trace=False):
        machines=[TapeMachine(row['program'],row['data']) for row in rows]
        history=[[] for _ in rows];state=None;matrix=self.model.matrix() if self.table is None else None
        while any(not m.done for m in machines):
            observations=[m.observation() for m in machines]
            if self.table is None:
                logits,state=self.model(torch.tensor(observations),state,matrix);actions=logits.argmax(-1).tolist()
            else:actions=[self.table[op][cell][last] for op,cell,last in observations]
            for index,(m,obs,action) in enumerate(zip(machines,observations,actions)):
                if m.done:continue
                if keep_trace:history[index].append(dict(observation=obs,action=ACTIONS[action]))
                m.step(action)
        results=[]
        for index,machine in enumerate(machines):
            result=machine.result()
            if keep_trace:result['trace']=history[index]
            results.append(result)
        return results
