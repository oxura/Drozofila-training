"""Operational CLI checks with teacher modules blocked from import."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BLOCKED_RUNNER = '''import importlib.abc,sys,runpy
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in {'stage6_tasks','stage7_tasks','stage6_train','stage7_train','stage7_evaluate'}:
            raise RuntimeError('Teacher/trainer import forbidden in inference: '+fullname)
sys.meta_path.insert(0,Block())
sys.argv=['fly_memory.py']+sys.argv[1:]
runpy.run_path('fly_memory.py',run_name='__main__')
'''


def main():
    examples = ['код a=2;b=3;c=0;d=1;a=7;b=8;?a',
                'код a=2;b=3;c=0;d=1;a=(a+b)%10;b=(b+1)%10;?a',
                'сумма 7,5,11,15', 'логика xor(1,and(1,0))']
    records = []
    for prompt in examples:
        command = [sys.executable, '-c', BLOCKED_RUNNER, prompt, '--json']
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        value = json.loads(result.stdout); assert 'text' in value and isinstance(value['halted'], bool)
        records.append(dict(prompt=prompt, output=value))
    errors = []
    for args in [['', '--json'], ['unsupported!', '--json'], [examples[0], '--max-new-tokens', '0']]:
        result = subprocess.run([sys.executable, 'fly_memory.py', *args], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 2
        errors.append(dict(arguments=args, exit_code=result.returncode))
    output = dict(teacher_imports_blocked=True, demonstrations=records, invalid_input_checks=errors,
                  note='Operational tests, not accuracy estimates. Every demonstration output retained without correction.')
    (ROOT / 'results/stage7/cli_verification.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__': main()
