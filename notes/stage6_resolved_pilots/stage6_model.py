"""Small shared causal transformer, optional connectome FF and prompt-copy head.

No arithmetic, Boolean semantics, program parser, or task-specific state machine
is present here. Attention, positional encoding and copying are supplied neural
architecture; arithmetic and program execution must be learned from token loss.
"""
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import ConnectomeLM
from stage6_tokens import VOCAB

ROOT=Path(__file__).resolve().parent


def rotary(x,positions):
    half=x.shape[-1]//2
    frequency=torch.exp(-math.log(10000)*torch.arange(half,device=x.device,dtype=x.dtype)/half)
    angles=positions[:,None,:,None]*frequency
    a,b=x[...,:half],x[...,half:]
    return torch.cat([a*angles.cos()-b*angles.sin(),a*angles.sin()+b*angles.cos()],dim=-1)


class GraphFF(nn.Module):
    def __init__(self,d,condition,seed):
        super().__init__();self.condition=condition;self.disable_edges=False
        graph=dict(np.load(ROOT/'data/graph_256.npz'))
        self.core=ConnectomeLM(graph,1,'frozen' if condition=='no_edges' else condition,'rate',seed)
        del self.core.embedding;del self.core.decoder
        self.input=nn.Linear(d,256);self.output=nn.Linear(256,d)

    def forward(self,x):
        drive=self.input(x);h=torch.tanh(drive)
        if self.condition!='no_edges' and not self.disable_edges:
            h=.25*h+.75*torch.tanh(drive+F.linear(h,self.core.matrix()))
        return self.output(h)


class Block(nn.Module):
    def __init__(self,d,heads,condition,seed):
        super().__init__();self.heads=heads
        self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.qkv=nn.Linear(d,3*d);self.project=nn.Linear(d,d)
        self.ff=(nn.Sequential(nn.Linear(d,312),nn.GELU(),nn.Linear(312,d)) if condition=='mlp'
                 else GraphFF(d,condition,seed))

    def forward(self,x,positions,allowed,cache=None):
        b,t,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        q,k=rotary(q,positions),rotary(k,positions)
        if cache is not None:k=torch.cat([cache[0],k],dim=2);v=torch.cat([cache[1],v],dim=2)
        h=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed).transpose(1,2).reshape(b,t,d)
        x=x+self.project(h);x=x+self.ff(self.norm2(x))
        return x,(k,v)


class Reasoner(nn.Module):
    def __init__(self,condition='mlp',copy=False,seed=0,d=64,layers=2,heads=4):
        super().__init__();torch.manual_seed(seed)
        self.condition,self.use_copy=condition,copy
        self.embedding=nn.Embedding(len(VOCAB),d,padding_idx=0)
        self.blocks=nn.ModuleList([Block(d,heads,condition,seed+i*100) for i in range(layers)])
        self.norm=nn.LayerNorm(d);self.output=nn.Linear(d,len(VOCAB))
        if copy:
            self.copy_q=nn.Linear(d,32,bias=False);self.copy_k=nn.Linear(d,32,bias=False)
            self.copy_gate=nn.Linear(d,1)

    def distribution(self,h,copy_keys=None,copy_ids=None,copy_mask=None):
        probabilities=torch.softmax(self.output(h),dim=-1)
        if self.use_copy:
            scores=torch.einsum('btd,bsd->bts',self.copy_q(h),copy_keys)/math.sqrt(32)
            weights=torch.softmax(scores.masked_fill(~copy_mask[:,None,:],-1e4),dim=-1)
            copied=torch.zeros_like(probabilities).scatter_add(2,copy_ids[:,None,:].expand(-1,h.shape[1],-1),weights)
            gate=torch.sigmoid(self.copy_gate(h))
            probabilities=gate*probabilities+(1-gate)*copied
        return probabilities.clamp_min(1e-9)

    def forward(self,ids,valid,prompt_mask,cache=None):
        # Cached and full teacher-forced paths share exactly the same operations.
        if cache is None:
            positions=(valid.long().cumsum(1)-1).clamp_min(0)
            t=ids.shape[1];causal=torch.ones(t,t,dtype=torch.bool,device=ids.device).tril()
            allowed=causal[None,None,:,:]&valid[:,None,None,:]
            # Padding queries may attend to themselves to avoid all-masked rows.
            allowed=allowed|((~valid)[:,None,:,None]&torch.eye(t,dtype=torch.bool,device=ids.device)[None,None,:,:])
            past=[None]*len(self.blocks)
        else:
            positions=cache['valid'].sum(1,keepdim=True).long()
            valid=torch.cat([cache['valid'],valid],dim=1)
            allowed=valid[:,None,None,:];past=cache['layers']
        h=self.embedding(ids);layers=[]
        for block,old in zip(self.blocks,past):h,new=block(h,positions,allowed,old);layers.append(new)
        h=self.norm(h)
        if cache is None:
            copy_keys=self.copy_k(h) if self.use_copy else None
            copy_ids=ids;copy_mask=prompt_mask
        else:copy_keys,copy_ids,copy_mask=cache['copy_keys'],cache['copy_ids'],cache['copy_mask']
        probs=self.distribution(h,copy_keys,copy_ids,copy_mask)
        new_cache=dict(layers=layers,valid=valid,copy_keys=copy_keys,copy_ids=copy_ids,copy_mask=copy_mask)
        return probs,new_cache


def make_model(config):
    return Reasoner(config['condition'],config['copy'],config['seed'],config.get('d',64),config.get('layers',2),config.get('heads',4))


def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=make_model(saved['config']);model.load_state_dict(saved['model']);model.eval()
    return model,saved
