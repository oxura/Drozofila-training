"""Free autoregressive inference, accepting strings only. No teacher/task import."""
import torch
from stage6_tokens import PAD,BOS,SEP,EOS,encode,decode,prompt_ids


@torch.no_grad()
def generate(model,prompts,max_new_tokens=256,batch_size=32):
    results=[]
    for start in range(0,len(prompts),batch_size):
        batch=prompts[start:start+batch_size];encoded=[prompt_ids(p) for p in batch]
        width=max(map(len,encoded));ids=torch.full((len(batch),width),PAD,dtype=torch.long)
        valid=torch.zeros_like(ids,dtype=torch.bool)
        for i,row in enumerate(encoded):ids[i,-len(row):]=torch.tensor(row);valid[i,-len(row):]=True
        prompt_mask=valid&(ids!=BOS)&(ids!=SEP)
        probabilities,cache=model(ids,valid,prompt_mask);probabilities=probabilities[:,-1]
        tokens=[[] for _ in batch];done=torch.zeros(len(batch),dtype=torch.bool);halted=[False]*len(batch)
        for _ in range(max_new_tokens):
            probabilities[:,[PAD,BOS,SEP]]=0
            chosen=probabilities.argmax(-1)
            for i,token in enumerate(chosen.tolist()):
                if done[i]:continue
                if token==EOS:done[i]=True;halted[i]=True
                else:tokens[i].append(token)
            if bool(done.all()):break
            next_ids=torch.where(done,torch.zeros_like(chosen),chosen)[:,None]
            probabilities,cache=model(next_ids,(~done)[:,None],prompt_mask,cache)
            probabilities=probabilities[:,-1]
        results.extend(dict(text=decode(row),halted=stop,generated_tokens=len(row)+int(stop)) for row,stop in zip(tokens,halted))
    return results
