"""Independent goal-planning test. No supplied plan is given to the solver."""
from pathlib import Path
import hashlib
import json
import numpy as np
from stage4_tasks import ROOT,OPS,functional_oracle


def prepare():
    rng=np.random.default_rng(2026091544);rows=[];seen=set()
    while len(rows)<64:
        data=list(map(int,rng.integers(0,10,int(rng.integers(2,7)))))
        witness=[OPS[int(i)] for i in rng.integers(0,6,int(rng.integers(2,7)))]
        goal=functional_oracle(witness,data)
        if (len(goal)==0)!=(len(rows)%4==0) or goal==data:continue
        key=(tuple(data),tuple(goal))
        if key in seen:continue
        seen.add(key);rows.append(dict(id=f'goal_{len(rows)}',initial=data,goal=goal,witness_for_evaluation_only=witness))
    root=ROOT/'data/stage4_planning';root.mkdir(parents=True,exist_ok=True)
    text=''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows);path=root/'goals.jsonl'
    if path.exists() and path.read_text()!=text:raise FileExistsError(str(path))
    path.write_text(text)
    manifest=dict(seed=2026091544,count=len(rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  max_depth=6,max_expanded_states=512,empty_goals=.25,
                  evaluated_runs=['reactive_fly_seed0','reactive_mlp_seed0','hidden_fly_seed0','hidden_gru_seed0'],
                  selection='Fixed seed 0 and these four controllers before opening goal evaluation',
                  planner_inputs=['initial','goal'],oracle_used_only_for_scoring=True,
                  search_algorithm='hand-written breadth-first search over learned transitions')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))


if __name__=='__main__':prepare()
