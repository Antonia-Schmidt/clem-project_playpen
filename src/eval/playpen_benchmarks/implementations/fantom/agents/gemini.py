import os
from types import SimpleNamespace
import google.generativeai as genai
from .base import AsyncBaseAgent


class AsyncGeminiAgent(AsyncBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__()
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        self.model = genai.GenerativeModel(self.args.model)

    def generate(self, prompt, temperature=None, max_tokens=None):
        output = self.model.generate_content(prompt,
                                             generation_config = genai.GenerationConfig(
                                                 max_output_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                                                 temperature=self.args.temperature if temperature is None else temperature,
                                             )
                                            )
        return output

    def preprocess_input(self, text):
        return text

    def postprocess_output(self, output):
        try:
            response = output.text
        except:
            response = "ERROR from model."
        return response