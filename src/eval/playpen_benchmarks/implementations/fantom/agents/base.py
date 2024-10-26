import time
import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

class BaseAgent(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def generate(self, prompt):
        pass
    
    @abstractmethod
    def interact(self, prompt):
        pass

    @abstractmethod
    def preprocess_input(self, text):
        pass

    @abstractmethod
    def postprocess_output(self, output):
        pass

class AsyncBaseAgent(BaseAgent):
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=16)

    def _set_default_args(self):
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 512
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 1.0
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0

    @abstractmethod
    def generate(self, prompt, temperature=None, max_tokens=None):
        pass

    def interact(self, prompt, temperature=None, max_tokens=None):
        output = self.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        response = self.postprocess_output(output)
        return response

    async def batch_generate(self, prompts, temperature=None, max_tokens=None):
        loop = asyncio.get_running_loop()
        completions = await asyncio.gather(*[
            loop.run_in_executor(self.executor, self.generate, prompt, temperature, max_tokens)
            for prompt in prompts
        ])
        return completions

    def batch_interact(self, prompts, temperature=None, max_tokens=None):
        while True:
            try:
                outputs = asyncio.run(self.batch_generate(prompts, temperature=None, max_tokens=None))
                responses = [self.postprocess_output(output) for output in outputs]
            except Exception as e:
                print("Error: {}".format(e))
                time.sleep(2)
                continue
            break

        return responses