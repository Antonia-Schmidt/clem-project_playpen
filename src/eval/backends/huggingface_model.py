import logging
from dataclasses import dataclass

from huggingface_hub.utils import GatedRepoError
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.backends.base_model import Model

logger = logging.getLogger(__name__)


@dataclass
class HuggingfaceModel(Model):
    def __post_init__(self) -> None:
        assert self.model_name is not None, "model_name must be provided"
        self.model = None
        self.tokenizer = None

        try:
            self._load_model_and_tokenizer()
        except GatedRepoError as e:
            logger.error(
                f"Access to model {self.model_name} is restricted. Error: {str(e)}"
            )
            logger.info(
                f"Please visit https://huggingface.co/{self.model_name} to request access."
            )
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")

    def _load_model_and_tokenizer(self):
        kwargs = {"token": self.api_key} if self.api_key else {}
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)

    def generate(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError(
                "Model or tokenizer not initialized. Please check the logs for errors."
            )

        device = "cuda"  # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        self.model.to(device)

        generated_ids = self.model.generate(**model_inputs, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]
