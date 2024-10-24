from dataclasses import dataclass
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM

@dataclass
class HuggingfaceModel(DeepEvalBaseLLM):
    model_name: str
    api_key: str = None
    
    def __post_init__(self) -> None:
        assert self.model_name is not None, "model_name must be provided"
        if self.api_key is not None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=self.api_key)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.api_key)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        
        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, *args, **kwargs) -> str:
        pass

    def get_model_name(self, *args, **kwargs) -> str:
        return self.model_name


