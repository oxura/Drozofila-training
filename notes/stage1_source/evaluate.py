"""Free-running decoding and execution-based evaluation of the tiny code grammar."""
import ast
from collections import defaultdict
import torch
from tasks import tokenize, OPS


def executable_code_matches(tokens, prompt):
    code = ' '.join(tokens)
    try:
        tree = ast.parse(code)
        allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
                   ast.BinOp, ast.Add, ast.Sub, ast.Call, ast.Name, ast.Load, ast.Constant)
        if any(not isinstance(n, allowed) for n in ast.walk(tree)):
            return False, False
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            return False, False
        fn = tree.body[0]
        if (fn.name != 'f' or len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return)
                or len(fn.args.args) != 2 or fn.decorator_list
                or fn.args.defaults or fn.args.kwonlyargs or fn.args.posonlyargs
                or fn.args.vararg or fn.args.kwarg):
            return False, False
        names = {arg.arg for arg in fn.args.args}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and n.id not in names | {'min', 'max'}:
                return False, False
            if isinstance(n, ast.Call) and (not isinstance(n.func, ast.Name)
                    or n.func.id not in {'min', 'max'} or len(n.args) != 2 or n.keywords):
                return False, False
        env = {'__builtins__': {}, 'min': min, 'max': max}
        exec(compile(tree, '<generated-toy-function>', 'exec'), env)
        operation = next(op for op in OPS if op in tokenize(prompt))
        values = [(-3, 7), (0, 0), (9, -4), (100, 2), (1.5, -2.0)]
        return True, all(env['f'](a, b) == OPS[operation](a, b) for a, b in values)
    except (SyntaxError, TypeError, ValueError, NameError, StopIteration, RecursionError):
        return False, False


@torch.no_grad()
def evaluate(model, rows, vocab, batch_size=64):
    lookup = {w: i for i, w in enumerate(vocab)}
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        prompt = [1] + [lookup.get(w, 4) for w in tokenize(row['prompt'])] + [2]
        groups[len(prompt)].append((index, prompt))
    predictions = [None] * len(rows)
    device = next(model.parameters()).device
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            ids = torch.tensor([p for _, p in batch], device=device)
            output = model.generate(ids).cpu().tolist()
            for (index, _), sequence in zip(batch, output):
                end = sequence.index(3) if 3 in sequence else len(sequence)
                predictions[index] = [vocab[i] for i in sequence[:end]]
    sums = defaultdict(lambda: dict(n=0, exact=0, code_valid=0, code_pass=0))
    examples = []
    for row, prediction in zip(rows, predictions):
        exact = prediction == tokenize(row['answer'])
        stats = sums[row['task']]
        stats['n'] += 1; stats['exact'] += int(exact)
        valid, passed = executable_code_matches(prediction, row['prompt']) if row['task'] == 'code' else (False, False)
        stats['code_valid'] += int(valid); stats['code_pass'] += int(passed)
        examples.append(dict(**row, prediction=' '.join(prediction), exact=exact,
                             code_valid=valid, code_pass=passed))
    metrics = {task: dict(n=s['n'], exact_match=s['exact'] / s['n'],
                         **({'valid_code': s['code_valid'] / s['n'], 'execution_pass': s['code_pass'] / s['n']}
                            if task == 'code' else {})) for task, s in sums.items()}
    metrics['macro_exact'] = sum(s['exact_match'] for s in metrics.values()) / len(metrics)
    return metrics, examples
