"""Checks required before freezing the streaming pilot, not benchmark results."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from torch.nn import functional as F
from stage8_stream import StreamMemory, FEATURES, encode_prompt
from stage8_data import ROOT, check, write_json


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    data = check(); result = dict(data=data)
    prompt = 'код a=2;b=3;aa=0;bb=1;a=7;b=8;?a'
    encoded = encode_prompt(prompt)
    assert encoded.shape == (7, FEATURES)
    assert encoded[-1, 1] == 1 and encoded[-1, -10:].sum() == 0
    assert encoded[0, 0] == 1 and encoded[0, -8] == 1
    for bad in ['', 'a=2;?a', 'код a=12;?a', 'код a=(a+b)%10;?a', 'код a=2;?aaa']:
        try: encode_prompt(bad)
        except ValueError: pass
        else: raise AssertionError(bad)
    torch.manual_seed(82000); baseline = StreamMemory('gru', 64)
    torch.manual_seed(82000); model = StreamMemory('slots', 64)
    x = torch.randn(4, 9, FEATURES); y = torch.tensor([1, 2, 3, 4])
    assert torch.equal(baseline(x)[0], model(x)[0])
    result['initial_function_preserved'] = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for _ in range(3):
        optimizer.zero_grad(); loss = F.cross_entropy(model(x)[0][:, -1], y)
        loss.backward(); optimizer.step()
    gradients = {name: float(p.grad.abs().sum()) for name, p in model.memory.named_parameters()}
    assert all(value > 0 for value in gradients.values()), gradients
    result['memory_parameter_gradient_sums_after_three_smoke_updates'] = gradients
    model.eval()
    with torch.no_grad():
        full, full_state = model(x); state = None; streamed = []
        for t in range(x.shape[1]):
            out, state = model(x[:, t:t + 1], state); streamed.append(out)
        error = float((full - torch.cat(streamed, 1)).abs().max())
        assert error < 2e-6
        assert torch.allclose(full_state[1], state[1], atol=2e-6, rtol=1e-5)
        altered = x.clone(); altered[:, 5:] = torch.randn_like(altered[:, 5:])
        assert torch.equal(full[:, :5], model(altered)[0][:, :5])
    result.update(streaming_max_logit_difference=error, causal_prefix_unchanged=True,
        persistent_state_floats_per_example=64 + 8 * 16)
    blocked = '''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in {'stage8_data','stage8_pilot','stage7_tasks','stage7_train','stage6_tasks','stage6_train'}:
   raise RuntimeError('Teacher import forbidden: '+fullname)
sys.meta_path.insert(0,Block())
import torch
torch.set_num_threads(1)
from stage8_stream import load_model,answer_stream
model,_=load_model(sys.argv[1])
value=answer_stream(model,sys.argv[2])
assert isinstance(value,int) and 0<=value<=9
'''
    with tempfile.TemporaryDirectory(prefix='stage8-smoke-') as temporary:
        checkpoint = Path(temporary) / 'model.pt'
        torch.save(dict(config=dict(mode='slots', width=64), model=model.state_dict()), checkpoint)
        subprocess.run([sys.executable, '-c', blocked, str(checkpoint), prompt], cwd=ROOT, check=True)
    result.update(inference_with_teacher_imports_blocked=True,
        smoke_updates=3, smoke_example_presentations=12,
        note='Random-input numerical checks, not training/evaluation evidence; these smoke weights are discarded.')
    write_json(ROOT / 'results/stage8_pilot/preflight_checks.json', result)
    print(json.dumps(result))


if __name__ == '__main__': main()
