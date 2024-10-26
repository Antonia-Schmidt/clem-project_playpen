import json
import os
from typing import Dict

from src.eval.backends.huggingface_local import HuggingfaceModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_credentials(backend, file_name="key.json") -> Dict:
    key_file = os.path.join(project_root, file_name)
    with open(key_file) as f:
        try:
            creds = json.load(f)
        except json.JSONDecodeError:
            raise ValueError("The key.json file is not a valid JSON.")
    assert backend in creds, f"No '{backend}' in {file_name}."
    assert "api_key" in creds[backend], f"No 'api_key' in {file_name}."
    return creds[backend]


def get_model(model_name) -> ValueError | HuggingfaceModel:
    model_registry_path = os.path.join(project_root, "backends", "model_registry.json")
    model_registry = json.load(open(model_registry_path))
    model_entry = next(
        (entry for entry in model_registry if entry.get("model_name") == model_name),
        None,
    )

    if model_entry is None:
        return ValueError(f"Model {model_name} not found.")
    if model_entry["backend"] == "huggingface":
        api_key = (
            load_credentials(model_entry["backend"])["api_key"]
            if model_entry["requires_api_key"] == True
            else None
        )
        model = HuggingfaceModel(model_entry["model_id"], api_key)

    # TODO: OpenAI
    return model
