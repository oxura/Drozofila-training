"""Task-neutral tape machine. All movement, writes, stage changes and halt are actions.

No task-specific execution policy lives here. The program is a user-supplied list
of high-level operations; choosing that high-level plan is NOT learned in stage 4.
"""
OPS=('copy','reverse','even','odd','inc','dec')
LEFT,RIGHT,NEXT,HALT=0,1,12,13
START=14
ACTIONS=('left','right')+tuple(f'write_{d}' for d in range(10))+('next','halt')


class TapeMachine:
    def __init__(self, program, data):
        if not program or any(op not in OPS for op in program):raise ValueError('Unknown or empty program')
        if any(type(d) is not int or not 0<=d<10 for d in data):raise ValueError('Expected decimal digits')
        self.program=tuple(program);self.tape=list(data);self.output=[]
        self.pc=0;self.cursor=-1;self.last=START;self.steps=0
        self.done=False;self.error=None
        # Resource bound exceeds every correct expert trace; it is not an action schedule.
        self.budget=4*(len(data)+2)*len(program)+4

    def observation(self):
        instruction=OPS.index(self.program[self.pc]) if self.pc<len(self.program) else len(OPS)
        cell=10 if self.cursor<0 else 11 if self.cursor>=len(self.tape) else self.tape[self.cursor]
        return [instruction,cell,self.last]

    def step(self,action):
        if self.done:return
        self.steps+=1
        if action==LEFT:self.cursor-=1
        elif action==RIGHT:self.cursor+=1
        elif 2<=action<=11:self.output.append(action-2)
        elif action==NEXT:
            if self.pc>=len(self.program):self.error='next_after_program'
            else:
                self.tape=self.output;self.output=[];self.cursor=-1;self.pc+=1
        elif action==HALT:self.done=True
        else:self.error='unknown_action'
        self.last=action
        if not -1<=self.cursor<=len(self.tape):self.error='pointer_out_of_bounds'
        if self.error:self.done=True
        if self.steps>=self.budget and not self.done:self.error='step_budget';self.done=True

    def result(self):
        return dict(answer=list(self.tape),program_completed=self.pc==len(self.program),
                    halted=self.done and self.last==HALT and self.error is None,
                    error=self.error,steps=self.steps)
