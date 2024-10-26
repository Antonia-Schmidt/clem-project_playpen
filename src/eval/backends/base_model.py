from abc import ABC, abstractmethod
from dataclasses import dataclass

from deepeval.models.base_model import DeepEvalBaseLLM


@dataclass
class Model(DeepEvalBaseLLM, ABC):
    model_name: str
    api_key: str | None = None

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def load_model(self):
        return self.model

    def get_model_name(self) -> str:
        return self.model_name
