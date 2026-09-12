"""Breadth-first planning using ONLY learned execution as the transition model.

The search procedure, operator inventory and exact goal comparison are engineered.
The neural network did not learn breadth-first search or invent these operations.
"""
from collections import deque
from stage4_env import OPS
from stage4_engine import NeuralExecutor


class GoalPlanner:
    def __init__(self,model,max_depth=6,max_states=512):
        self.executor=NeuralExecutor(model);self.max_depth=max_depth;self.max_states=max_states

    def solve(self,initial,goal):
        initial,goal=tuple(initial),tuple(goal)
        if initial==goal:return dict(found=True,program=[],predicted_result=list(initial),expanded=0,visited=1)
        queue=deque([(initial,[])]);visited={initial};expanded=0;invalid=0
        while queue and expanded<self.max_states:
            state,path=queue.popleft()
            if len(path)>=self.max_depth:continue
            expanded+=1
            requests=[dict(program=[op],data=list(state)) for op in OPS]
            transitions=self.executor.run(requests)
            for op,result in zip(OPS,transitions):
                if not result['halted'] or not result['program_completed']:
                    invalid+=1;continue
                following=tuple(result['answer']);program=path+[op]
                if following==goal:
                    return dict(found=True,program=program,predicted_result=list(following),expanded=expanded,
                                visited=len(visited),invalid_model_transitions=invalid)
                if following not in visited:
                    visited.add(following);queue.append((following,program))
        return dict(found=False,program=None,expanded=expanded,visited=len(visited),
                    invalid_model_transitions=invalid,depth_limit=self.max_depth,expansion_limit=self.max_states)
