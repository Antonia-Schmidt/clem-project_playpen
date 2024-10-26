# https://github.com/openai/openai-python
import os
import time
import asyncio
import openai
from openai import OpenAI, AsyncOpenAI
from types import SimpleNamespace
from .base import BaseAgent


class GPT3BaseAgent(BaseAgent):
    def __init__(self, kwargs: dict):
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def _set_default_args(self):
        if not hasattr(self.args, 'model'):
            self.args.model = "gpt-3.5-turbo-instruct"
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 0.9
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0
        if not hasattr(self.args, 'n'):
            self.args.n = 1

    def generate(self, prompt, temperature=None, max_tokens=None):
        while True:
            try:
                completion = self.client.completions.create(model=self.args.model,
                                                            prompt=prompt,
                                                            temperature=self.args.temperature if temperature is None else temperature,
                                                            max_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                                                            top_p=self.args.top_p,
                                                            frequency_penalty=self.args.frequency_penalty,
                                                            presence_penalty=self.args.presence_penalty,
                                                            stop=self.args.stop_tokens if hasattr(self.args, 'stop_tokens') else None,
                                                            logprobs=self.args.logprobs if hasattr(self.args, 'logprobs') else 0,
                                                            echo=self.args.echo if hasattr(self.args, 'echo') else False,
                                                            n=self.args.n if hasattr(self.args, 'n') else 1)
                break
            except (RuntimeError, openai.RateLimitError, openai.APIError, openai.APIConnectionError) as e:
                print("Error: {}".format(e))
                time.sleep(2)
                continue

        return completion
    
    def preprocess_input(self, text):
        return text

    def postprocess_output(self, outputs):
        responses = [c.text.strip() for c in outputs.choices]

        return responses[0]

    def parse_ordered_list(self, numbered_items):
        ordered_list = numbered_items.split("\n")
        output = [item.split(".")[-1].strip() for item in ordered_list if item.strip() != ""]

        return output

    def interact(self, prompt, temperature=None, max_tokens=None):
        outputs = self.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        responses = self.postprocess_output(outputs)

        return responses

class ConversationalGPTBaseAgent(GPT3BaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

    def _set_default_args(self):
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 1.0
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0

    def generate(self, prompt, temperature=None, max_tokens=None):
        while True:
            try:
                completion = self.client.chat.completions.create(model=self.args.model,
                                                                 messages=[{"role": "user", "content": f"{prompt}"}],
                                                                 temperature=self.args.temperature if temperature is None else temperature,
                                                                 max_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
                break
            except (openai.APIError, openai.RateLimitError) as e:
                print("Error: {}".format(e))
                time.sleep(1)
                continue

        return completion

    def json_generate(self, prompt, temperature=None, max_tokens=None):
        while True:
            try:
                completion = self.client.chat.completions.create(model=self.args.model,
                                                                 response_format={ "type": "json_object" },  
                                                                 messages=[
                                                                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                                                                    {"role": "user", "content": f"{prompt}"}
                                                                    ],
                                                                 temperature=self.args.temperature if temperature is None else temperature,
                                                                 max_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
                break
            except (openai.APIError, openai.RateLimitError) as e:
                print("Error: {}".format(e))
                time.sleep(1)
                continue

        return completion

    def postprocess_output(self, outputs):
        responses = [c.message.content.strip() for c in outputs.choices]

        return responses[0]

    def interact(self, prompt, temperature=0, max_tokens=256, history=None, json_mode=False):
        if json_mode:
            output = self.json_generate(prompt, temperature=temperature, max_tokens=max_tokens)
        else:
            output = self.generate(prompt, temperature=temperature, max_tokens=max_tokens, history=history)
        response = self.postprocess_output(output)

        return response

    def generate(self, prompt, temperature=None, max_tokens=None, history=None):
        messages = []
        if history is not None:
            for idx, msg in enumerate(history):
                if idx % 2 == 0:
                    messages.append({"role": "user", "content": f"{msg}"})
                else:
                    messages.append({"role": "assistant", "content": f"{msg}"})
        messages.append({"role": "user", "content": f"{prompt}"})
        while True:
            try:
                completion = self.client.chat.completions.create(model=self.args.model,
                                                                 messages=messages,
                                                                 temperature=self.args.temperature if temperature is None else temperature,
                                                                 max_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
                break
            except (openai.APIError, openai.RateLimitError) as e:
                print("Error: {}".format(e))
                time.sleep(1)
                continue

        return completion

    def batch_interact(self, prompts, temperature=1, max_tokens=256):
        raise NotImplementedError

class AsyncConversationalGPTBaseAgent(ConversationalGPTBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)
        self._set_default_args()
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    async def batch_generate(self, prompts, temperature=0, max_tokens=256):
        completions = await asyncio.gather(*[self.client.chat.completions.create(model=self.args.model,
                                                                                 messages=[{"role": "user", "content": f"{prompt}"}],
                                                                                 temperature=temperature,
                                                                                 max_tokens=max_tokens)
                                             for prompt in prompts])
        return completions

    def batch_interact(self, prompts, temperature=1, max_tokens=256):
        while True:
            try:
                outputs = asyncio.run(self.batch_generate(prompts, temperature, max_tokens))
            except Exception as e:
                print("Error: {}".format(e))
                time.sleep(2)
                continue
            break
        responses = [self.postprocess_output(output) for output in outputs]

        return responses

    def interact(self, prompt, temperature=0, max_tokens=256):
        outputs = self.batch_interact([prompt], temperature, max_tokens)

        return outputs[0]
