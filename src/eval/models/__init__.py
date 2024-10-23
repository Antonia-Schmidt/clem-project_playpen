import json

from .base_llm import BaseLlm
from .gpt_llm import GptLlm
from .hf_llm import HfLlm

__all__ = ["BaseLlm", "GptLlm", "HfLlm"]

def get_model(model_name):
    model_registry = json.loads('model_registry.json')
    model_entry = next((entry for entry in model_registry if entry.get('model_name') == model_name), None)

    if model_entry is None:
        return ValueError(f'Model {model_name} not found.')
    if model_entry['backend'] == 'huggingface':
