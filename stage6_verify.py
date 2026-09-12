"""Operational CLI/isolation checks; never changes a confirmation checkpoint."""
import builtins
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def isolated_child():
    original=builtins.__import__
    forbidden={'stage6_tasks','stage6_atomic','stage6_train','stage6_checks','stage6_evaluate','stage6_report'}
    def checked(name,*args,**kwargs):
        if name.split('.')[0] in forbidden:raise RuntimeError('Teacher/evaluator import forbidden during inference: '+name)
        return original(name,*args,**kwargs)
    builtins.__import__=checked
    sys.argv.remove('--child-isolated')
    import fly_reason
    fly_reason.main()


def main():
    selected=json.loads((ROOT/'results/stage6/default_model.json').read_text())
    checkpoint=ROOT/selected['checkpoint'];before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # First validation prompt in each family, fixed by file order, no selection
    # for success and no labels supplied to the CLI process.
    validation=[json.loads(line) for line in (ROOT/'data/stage6/val.jsonl').read_text().splitlines()]
    cases=[next(row for row in validation if row['family']==family) for family in ['sum','logic','code']]
    outputs=[]
    for row in cases:
        cmd=[sys.executable,str(Path(__file__)), '--child-isolated',row['prompt'],'--json']
        result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,check=True)
        out=json.loads(result.stdout);assert isinstance(out['text'],str) and isinstance(out['halted'],bool)
        outputs.append(dict(validation_id=row['id'],family=row['family'],prompt=row['prompt'],generated=out))
    invalid=[]
    for arguments in [['сумма 🙂'],['сумма 1,2','--max-new-tokens','0']]:
        result=subprocess.run([sys.executable,'fly_reason.py',*arguments],cwd=ROOT,capture_output=True,text=True)
        assert result.returncode==2 and 'error:' in result.stderr
        invalid.append(dict(arguments=arguments,exit_code=result.returncode,error=result.stderr.splitlines()[-1]))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before
    record=dict(checkpoint=selected['checkpoint'],checkpoint_unchanged=True,teacher_imports_blocked_during_inference=True,
                cases=outputs,invalid_inputs=invalid,scope='Operational checks, not an accuracy benchmark; examples fixed by validation order.')
    path=ROOT/'results/stage6/cli_verification.json';path.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k not in ['cases','invalid_inputs']},ensure_ascii=False))


if __name__=='__main__':
    if '--child-isolated' in sys.argv:isolated_child()
    else:main()
