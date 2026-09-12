"""Training-only atomic curriculum: arithmetic pairs, variable reads, logic gates."""
import hashlib
import json
import random
from stage6_tasks import DATA,arithmetic_row,random_program,random_tree,logic_row,pairs,HELD_PAIRS,load


def main():
    path=DATA/'atomic_train.jsonl'
    if path.exists():raise FileExistsError(path)
    rng=random.Random(2026091766);seen=set();rows=[]
    def add(row):
        prompt=row['prompt']
        if prompt in seen:return False
        bucket=int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4],'little')%10
        if bucket>=8:return False
        row['id']='atomic-'+str(len(rows));rows.append(row);seen.add(prompt);return True
    for a in range(100):
        for b in range(100):add(arithmetic_row([a,b]))
    for _ in range(12000):
        while not add(random_program(rng,1)):pass
    for _ in range(100000):
        t=random_tree(rng,2)
        if not isinstance(t,int) and not pairs(t)&HELD_PAIRS:add(logic_row(t))
    held={r['prompt'] for name in ['val','test','long','very_long','composition','deep_logic','alias','renamed','large_values'] for r in load(name)}
    assert not seen&held
    original={r['prompt'] for r in load('train')}
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows))
    manifest=dict(seed=2026091766,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),count=len(rows),
                  families={f:sum(r['family']==f for r in rows) for f in ['sum','code','logic']},
                  overlap_with_original_training=len(seen&original),overlap_with_validation_or_final=0,
                  selection='Constructed from diagnosis of train/validation errors; no final predictions read.',
                  same_primitive_operators=True,source_registers='abcd',maximum_arithmetic_operand=99,
                  notes='Training-label augmentation only. Inference remains unchanged and has no atomic task executor.')
    (DATA/'atomic_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
