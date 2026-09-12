"""Artificial working-memory interface around the unchanged connectome mask.

This is an engineered sequence learner, not a reconstruction of fly plasticity.
Attention reads only encoder neuron states; there is no symbolic answer path.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model import ConnectomeLM, SurrogateSpike


class MemoryConnectome(nn.Module):
    def __init__(self,graph,vocab_size,condition='fly',cell='rate',seed=0,
                 memory=True,gated=True,gru_size=80):
        super().__init__()
        torch.manual_seed(seed)
        self.condition=condition;self.cell=cell;self.memory=memory;self.gated=gated
        self.disable_memory=False
        self.disable_edges=False
        self.core=ConnectomeLM(graph,vocab_size,
                               condition if condition in ['fly','rewired','frozen'] else 'fly',cell,seed)
        if condition=='gru':
            n=gru_size
            self.core.embedding=nn.Embedding(vocab_size,n,padding_idx=0)
            self.core.decoder=nn.Linear(n,vocab_size)
            self.gru=nn.GRUCell(n,n)
            del self.core.edge_log_gain
        else:
            n=self.core.n
            if condition=='no_edges':self.core.edge_log_gain.requires_grad_(False)
        self.n=n
        self.update_logit=nn.Parameter(torch.zeros(n),requires_grad=gated and condition!='gru')
        if memory:
            self.key=nn.Linear(n,32,bias=False)
            self.query=nn.Linear(n,32,bias=False)
            self.score=nn.Linear(32,1,bias=False)
            self.context_gain=nn.Parameter(torch.ones(n))

    def matrix(self):
        if self.condition=='gru':return None
        self.core.disable_edges=self.disable_edges or self.condition=='no_edges'
        return self.core.matrix()

    def step(self,tokens,state,matrix,context=None):
        h,spikes=state
        drive=self.core.embedding(tokens)
        if context is not None:drive=drive+context*self.context_gain
        if self.condition=='gru':
            h=self.gru(drive,h)
        elif self.cell=='rate':
            candidate=torch.tanh(drive+F.linear(h,matrix))
            gate=self.update_logit.sigmoid() if self.gated else 0.75
            h=(1-gate)*h+gate*candidate
        else:
            beta=(0.5+0.49*self.update_logit.sigmoid()) if self.gated else 0.85
            voltage=beta*h+drive+F.linear(spikes,matrix)
            spikes=SurrogateSpike.apply(voltage-1)
            h=voltage-spikes.detach()
        return h,spikes

    def encode(self,ids,matrix):
        z=self.core.embedding.weight.new_zeros(ids.shape[0],self.n)
        state=z,z.clone();states=[]
        valid=ids!=0
        for t in range(ids.shape[1]):
            next_state=self.step(ids[:,t],state,matrix)
            keep=valid[:,t,None]
            state=tuple(torch.where(keep,new,old) for new,old in zip(next_state,state))
            states.append(state[0])
        memory=torch.stack(states,dim=1)
        keys=self.key(memory) if self.memory else None
        return state,memory,keys,valid

    def context(self,state,memory,keys,valid):
        if not self.memory or self.disable_memory:return None
        logits=self.score(torch.tanh(keys+self.query(state[0])[:,None,:])).squeeze(-1)
        weights=logits.masked_fill(~valid,-torch.inf).softmax(dim=-1)
        return torch.bmm(weights[:,None,:],memory).squeeze(1)

    def forward(self,prompts,decoder_inputs):
        matrix=self.matrix()
        state,memory,keys,valid=self.encode(prompts,matrix)
        outputs=[]
        for t in range(decoder_inputs.shape[1]):
            context=self.context(state,memory,keys,valid)
            state=self.step(decoder_inputs[:,t],state,matrix,context)
            outputs.append(self.core.decoder(state[0]))
        return torch.stack(outputs,1)

    @torch.no_grad()
    def generate(self,prompt_ids,max_tokens=32):
        self.eval();matrix=self.matrix()
        state,memory,keys,valid=self.encode(prompt_ids,matrix)
        token=torch.full((prompt_ids.shape[0],),1,device=prompt_ids.device,dtype=torch.long)
        done=torch.zeros_like(token,dtype=torch.bool);outputs=[]
        for _ in range(max_tokens):
            state=self.step(token,state,matrix,self.context(state,memory,keys,valid))
            logits=self.core.decoder(state[0])
            logits[:,[0,1,2,4]]=-torch.inf
            token=logits.argmax(-1)
            token=torch.where(done,torch.full_like(token,3),token)
            outputs.append(token);done|=token==3
            if bool(done.all()):break
        return torch.stack(outputs,1)
