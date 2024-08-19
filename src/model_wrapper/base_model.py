"""
This class represents an abstraction of the wrapper around a huggingface model and its base functionality.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from src.config.configurations import CustomBitsAndBitesConfiguration

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class BaseModel:
    def __int__(
        self, model_name: str, device_map: dict, bnb_config: CustomBitsAndBitesConfiguration, output_dir: str,
        dataset_dir: str,
    ):
        self.model_name: str = ""

        # configurations
        self.device_map: dict = {}
        self.bnb_config: CustomBitsAndBitesConfiguration = ""

        # path
        self.output_dir: str = ""
        self.dataset_dir: str = ""

    def load_model(self) -> PreTrainedModel:
        """
        Load the huggingface base model using bitsandbytes quantization
        :return:
        """
        logging.info(f"Loading model {self.model_name}")
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config.get_bnb_config(),
            device_map=self.device_map,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        return model

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        logging.info(f"Loading Tokenizer for:  {self.model_name}")
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

        return tokenizer
