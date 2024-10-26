import anthropic
from .base import AsyncBaseAgent
from types import SimpleNamespace

class AsyncClaudeAgent(AsyncBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__()
        self.client = anthropic.Anthropic() # defaults to api_key=os.environ.get("ANTHROPIC_API_KEY")
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()

    def generate(self, prompt, temperature=None, max_tokens=None):
        message = self.client.messages.create(
            model=self.args.model,
            max_tokens = self.args.max_tokens if max_tokens is None else max_tokens,
            temperature = self.args.temperature if temperature is None else temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message

    def preprocess_input(self, text):
        return text

    def postprocess_output(self, output):
        return output.content[0].text
