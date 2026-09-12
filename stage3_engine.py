"""Inference has NO arithmetic oracle. Network predicts each digit and carry.

Alignment, tape traversal, signed routing, expression parsing and composition
are engineered. Optional transition cache is built from MODEL predictions only.
It is a compiled learned finite-state controller, not a table of correct labels.
"""
import ast
import re
import torch

OPS = ('add','sub','mul_digit')


def canonical(value):
    value = str(value)
    if not re.fullmatch(r'-?[0-9]+', value): raise ValueError('Expected a decimal integer')
    negative = value.startswith('-')
    digits = value.lstrip('-').lstrip('0') or '0'
    return ('-' if negative and digits!='0' else '')+digits


class NeuralArithmetic:
    def __init__(self, model, compiled=False):
        self.model = model.eval()
        self.compiled = compiled
        if compiled and model.mode!='discrete':
            raise ValueError('A continuous hidden state cannot be compiled into this finite table')
        self.table = None
        if compiled:
            # Include even invalid/unreachable states, so a neural error stays an error.
            x = torch.cartesian_prod(torch.arange(3),torch.arange(10),torch.arange(10),torch.arange(10))
            with torch.no_grad():
                logits,_ = model(x)
            self.table = logits.argmax(-1).reshape(3,10,10,10,2)

    @torch.no_grad()
    def batch_unsigned(self, rows, return_trace=False, zero_carry=False):
        """All rows have the same input width; subtraction requires a >= b.

        No int(a) +/-/* int(b) is executed here. int is used only on SINGLE digits.
        """
        width = max(max(len(r['a']),len(r['b'])) for r in rows)+1
        op = torch.tensor([OPS.index(r['op']) for r in rows])
        a = torch.tensor([[int(ch) for ch in r['a'][::-1].ljust(width,'0')] for r in rows])
        b = torch.tensor([[int(ch) for ch in r['b'][::-1].ljust(width,'0')] for r in rows])
        carry = torch.zeros(len(rows),dtype=torch.long)
        matrix = self.model.matrix() if self.table is None else None
        state = None; out=[]; history=[]
        for pos in range(width):
            right = torch.where(op==2,b[:,0],b[:,pos])
            carry_in = torch.zeros_like(carry) if zero_carry else carry
            x = torch.stack([op,a[:,pos],right,carry_in],dim=1)
            if self.table is None:
                logits,state = self.model(x,state,matrix)
                prediction = logits.argmax(-1)
            else:
                prediction = self.table[op,a[:,pos],right,carry_in]
            digit,carry = prediction[:,0],prediction[:,1]
            out.append(digit)
            if return_trace:
                history.append(dict(position=pos,inputs=x.tolist(),predictions=prediction.tolist()))
        values = torch.stack(out,dim=1).tolist()
        answers = [''.join(map(str,v))[::-1].lstrip('0') or '0' for v in values]
        return (answers,history) if return_trace else answers

    def unsigned(self, op, a, b):
        return self.batch_unsigned([dict(op=op,a=a,b=b)])[0]

    def add(self,a,b):
        a,b = canonical(a),canonical(b)
        sa,sb = a.startswith('-'),b.startswith('-')
        aa,bb = a.lstrip('-'),b.lstrip('-')
        if sa==sb:
            result=self.unsigned('add',aa,bb)
            return canonical(('-' if sa else '')+result)
        # Magnitude comparison / operand routing is an external interface.
        if (len(aa),aa)<(len(bb),bb): aa,bb,sa=bb,aa,sb
        result=self.unsigned('sub',aa,bb)
        return canonical(('-' if sa else '')+result)

    def subtract(self,a,b):
        b=canonical(b)
        return self.add(a,b[1:] if b.startswith('-') else '-'+b)

    def multiply(self,a,b):
        a,b=canonical(a),canonical(b)
        negative=a.startswith('-') != b.startswith('-')
        aa,bb=a.lstrip('-'),b.lstrip('-')
        if len(aa)<len(bb): aa,bb=bb,aa
        # Grade-school partial-product schedule is given; all digit products and
        # all additions of partial products are supplied by the learned network.
        result='0'
        for shift,digit in enumerate(bb[::-1]):
            partial=self.unsigned('mul_digit',aa,digit)
            result=self.unsigned('add',result,canonical(partial+'0'*shift))
        return canonical(('-' if negative else '')+result)

    def expression(self, text):
        """Restricted arithmetic grammar; never exec/eval user-supplied code."""
        if len(text)>20000: raise ValueError('Expression too long for this interface')
        tree=ast.parse(text,mode='eval')
        nodes=list(ast.walk(tree))
        if len(nodes)>300: raise ValueError('Too many operations')
        def run(node):
            if isinstance(node,ast.Constant) and type(node.value) is int:
                return str(node.value)
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
                value=run(node.operand)
                return canonical(('-'+value if not value.startswith('-') else value[1:])
                                 if isinstance(node.op,ast.USub) else value)
            if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult)):
                a,b=run(node.left),run(node.right)
                function=self.add if isinstance(node.op,ast.Add) else self.subtract if isinstance(node.op,ast.Sub) else self.multiply
                return function(a,b)
            raise ValueError('Supported: decimal integers, parentheses, +, -, *')
        return run(tree.body)

    def command(self,text):
        match=re.fullmatch(r'\s*(сложи|вычти|умножь)\s+(-?[0-9]+)\s+(-?[0-9]+)\s*',text.lower())
        if match:
            op,a,b=match.groups()
            return {'сложи':self.add,'вычти':self.subtract,'умножь':self.multiply}[op](a,b)
        return self.expression(text)
