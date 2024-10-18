from dataclasses import dataclass

from .base_eval import BaseEval

@dataclass
class BbhEval(BaseEval):
    
    def evaluate(self, llm) -> tuple:
        pass