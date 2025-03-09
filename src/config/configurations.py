import torch
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
import os

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class CustomUnslothModelConfig:
    def __init__(self, max_seq_length: int = 2048, dtype: str = None, load_in_4_bit: bool = True, use_gradient_checkpointing: bool = True):
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4_bit = load_in_4_bit
        self.use_gradient_checkpointing: bool = True

    def as_dict(self):
        return {
            'max_seq_length': self.max_seq_length,
            'dtype': self.dtype,
            'load_in_4_bit': self.load_in_4_bit,
            'use_gradient_checkpointing': self.use_gradient_checkpointing,
        }


class CustomLoraConfiguration(object):
    def __init__(self, lora_r: int = 64, lora_alpha: int = 32, lora_dropout: int = 0,
                 lora_targets=None, use_rslora: bool = False, loftq_config: any = None):

        if lora_targets is None:
            lora_targets = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"]

        self.lora_r: int = lora_r
        self.lora_alpha: int = lora_alpha
        self.lora_dropout: float = lora_dropout  # NOTE: must be 0 cause unsloth cannot handle other
        self.target_modules = lora_targets
        self.use_rslora: bool = use_rslora
        self.loftq_config: any = loftq_config
        self.lora_bias: any = "none"

    def get_lora_config(self) -> LoraConfig:
        return LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.target_modules,
        )

    def as_dict(self):
        return {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_targets": self.target_modules,
            "use_rslora": self.use_rslora,
            "loftq_config": self.loftq_config,
            "lora_bias": self.lora_bias,
        }


class CustomBitsAndBitesConfiguration:
    def __init__(self, use_4bit: bool = True, bnb_4bit_compute_dtype: str = "float16", bnb_4bit_quant_type: str = "nf4",
                 use_nested_quant: bool = False):
        self.use_4bit: bool = use_4bit
        self.bnb_4bit_compute_dtype: str = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type: str = bnb_4bit_quant_type
        self.use_nested_quant: bool = use_nested_quant

    def get_bnb_config(self) -> BitsAndBytesConfig:
        # Load tokenizer and model with QLoRA configuration
        compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=self.use_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.use_nested_quant,
        )

    def as_dict(self):
        return {
            "use_4bit": self.use_4bit,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "use_nested_quant": self.use_nested_quant,
        }


class CustomTrainingArguments:
    def __init__(
            self,
            output_dir: str = "./results",
            num_train_epochs: int = 4,
            per_device_eval_batch_size: int = 2,
            per_device_train_batch_size: int = 12,
            optim: str = "paged_adamw_32bit",
            lr_scheduler_type: str = "linear",
            max_steps: int = -1,
            weight_decay: float = 0.01,
            learning_rate: float = 2e-4,
            max_grad_norm: float = 0.3,
            gradient_accumulation_steps: int = 4,
            fp16: bool = False,
            bf16: bool = False,
            warmup_ratio: float = 0.03,
            group_by_length: bool = True,
            save_steps: int = 0,
            logging_steps: int = 10,
            warmup_steps: int = 5,
            hub_model_id: str = None,
            do_eval=True,
            evaluation_strategy="steps",
            eval_steps=100,  # Evaluate every 100 steps
    ):
        self.output_dir: str = output_dir
        self.num_train_epochs: int = num_train_epochs
        self.per_device_train_batch_size: int = per_device_train_batch_size
        self.gradient_accumulation_steps: int = gradient_accumulation_steps
        self.optim: str = optim
        self.save_steps: int = save_steps
        self.logging_steps: int = logging_steps
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.fp16: bool = fp16
        self.bf16: bool = bf16
        self.max_grad_norm: float = max_grad_norm
        self.max_steps: int = max_steps
        self.warmup_ratio: float = warmup_ratio
        self.group_by_length: bool = group_by_length
        self.lr_scheduler_type: str = lr_scheduler_type
        self.report_to: list = ["tensorboard", "wandb"]
        self.warmup_steps = warmup_steps
        self.seed: int = 7331
        self.run_name = "Default Run Unnamed"
        self.hub_model_id = hub_model_id
        self.do_eval = do_eval
        self.evaluation_strategy = evaluation_strategy
        self.eval_steps = eval_steps

        if num_train_epochs and max_steps:
            print("Cannot use num training epochs and max steps togehter, will use only training epochs to use max_steps set num train epochs to none")
            self.max_steps = None
       

    def get_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            fp16=self.fp16,
            bf16=self.bf16,
            max_grad_norm=self.max_grad_norm,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            group_by_length=self.group_by_length,
            lr_scheduler_type=self.lr_scheduler_type,
            report_to=self.report_to,
            warmup_steps=self.warmup_steps,
            seed=self.seed,
            run_name=self.run_name,
            hub_model_id=self.hub_model_id,
            do_eval=self.do_eval,
            evaluation_strategy=self.evaluation_strategy,
            eval_steps=self.eval_steps, 
        )

    def as_dict(self):
        return {
            "save_steps": self.save_steps,
            "optim": self.optim,
            "learning_rate": self.learning_rate,
            "logging_steps": self.logging_steps,
            "weight_decay": self.weight_decay,
            "group_by_length": self.group_by_length,
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "max_grad_norm": self.max_grad_norm,
            "max_steps": self.max_steps,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "lr_scheduler_type": self.lr_scheduler_type,
            "report_to": self.report_to,
            "seed": self.seed,
            "run_name": self.run_name,
            "hub_model_id": self.hub_model_id

        }


class CustomInferenceConfig:
    def __init__(
            self,
            repetition_penalty: float = 1.19,
            do_sample: bool = True,
            return_full_text: bool = False,
            max_new_tokens: int = 180
    ):
        self.repetition_penalty: float = repetition_penalty
        self.do_sample: bool = do_sample
        self.return_full_text: bool = return_full_text
        self.max_new_tokens: int = max_new_tokens


class EnsembleAdapter:
    def __init__(
            self,
            adapter_path: str,
            adapter_name: str,
            adapter_task_description: str,
            prepare_input_prompt: type(abs),
            clean_output: type(abs),
            inference_config: CustomInferenceConfig,
            sample_output: str = "",
    ):
        """
        The Ensemble Adapter represents a wrapper around a LoRA adapter and adds meta information and functionality
        around the model adapter that shall be part of an ensemble
        :param adapter_path: the path to the directory of the adapter.
        :param adapter_name: the internal name of the adapter (must be unique for the ensemble)
        :param adapter_task_description: A brief description of what the adapter is supposed to do
        :param prepare_input_prompt: function that takes a dictionary as input parameters and returns a valid inference
        prompt in the format the adapter is fine-tuned on.
        :param clean_output: A function that is called after the inference to parse the output of adapter model

        Note: mandatory fields for the input parameter of the prepare input method are:
        {
            db_id: the database id of the current input,
            prev_output: the output of the previous model as a list of strings,
            prev_clean_output: the cleaned output of the previous model,
        }

        If adapter_path is set to none, the ensemble step will be executed solely on the loaded base model without any
        adapters.

        """
        self.adapter_path: str = adapter_path
        self.adapter_name: str = adapter_name
        self.adapter_task_description: str = adapter_task_description
        self.prepare_input: type(abs) = prepare_input_prompt
        self.clean_output: type(abs) = clean_output
        self.sample_output: str = sample_output
        self.inference_config: CustomInferenceConfig = inference_config


class EnsembleAdapterStack:
    def __init__(self, adapters: list[EnsembleAdapter]):
        """
        Represents a stack of model adapters that are supposed to be executed after each other to perform
        a desired task.
        :param adapters:
        """
        self.adapters: list[EnsembleAdapter] = adapters
        self.adapter_names: list = [adapter.adapter_name for adapter in self.adapters]
        self.adapter_ids: list = [adapter.adapter_path for adapter in self.adapters]

        # check whether the adapters are properly constructed
        self.check_adapters()

        # check the flow through the different processing functions
        self.check_trajectory_flow()

        # prepare the ensemble workflow configuration
        self.ensemble_configuration: dict = self.prepare_ensemble_configuration()

        # print the ensemble
        self.print_adapter_stack()

    def check_adapters(self):
        logging.info("Start Checking all Adapters")
        for adapter in self.adapters:
            if adapter.adapter_path is not None and not os.path.exists(adapter.adapter_path):
                exit(-1)

            if adapter.adapter_path is not None and not os.path.exists(adapter.adapter_path + '/adapter_config.json'):
                logging.warning(f"Adapter: {adapter.adapter_name} has no valid adapter path {adapter.adapter_path}")
                exit(-1)
        logging.info("Successfully checked all Adapter directories")

        if len(self.adapter_names) != len(set(self.adapter_names)):
            logging.warning(
                f"Adapter-names must be unique, but encountered at least two adapters with the same name in {self.adapter_names}")
            exit(-1)

        logging.info("Successfully checked all Adapter names for uniqueness")

    def check_trajectory_flow(self):
        sample_input: dict = {
            'db_id': 'sample_database_id',
            'prev_output': ['sample_in_1', 'sample_in_2', 'sample_in_3'],
            'prev_clean_output': ['sample_in_1', 'sample_in_2', 'sample_in_3']
        }

        logging.info("Start checking prepare input prompts")
        for adapter in self.adapters:
            try:
                prompt: str = adapter.prepare_input(sample_input)
            except Exception as e:
                logging.warning(f"Prepare Input Prompt for adapter {adapter.adapter_name} Yields an error {e}")
                exit(-1)

            if type(prompt) != str:
                logging.warning(
                    f"Prepare Input Prompt for adapter {adapter.adapter_name} does not return the correct format")
                exit(-1)

        logging.info("Successfully checked all prepare input functions")
        logging.info("Start checking clean output functions")

        for adapter in self.adapters:
            try:
                clean_output: list = adapter.clean_output(adapter.sample_output)
            except Exception as e:
                logging.warning(f"Prepare Input Prompt for adapter {adapter.adapter_name} Yields an error {e}")
                exit(-1)

            if type(clean_output) != list:
                logging.warning(
                    f"Clean Output for adapter {adapter.adapter_name} does not return the correct format expected list")
                exit(-1)

        logging.info("Successfully checked all clean output functions")

    def prepare_ensemble_configuration(self):
        logging.info("Start Registering Ensemble Members (Adapters)")
        ensemble_configuration: dict = {}
        for i, adapter in enumerate(self.adapters):
            adapter_key: str = f"S{i+1}"
            ensemble_configuration[adapter_key] = {
                "adapter": adapter
            }
            logging.info(f"Registered S{i+1}: {adapter.adapter_name} as step number {i + 1} in the ensemble")

        logging.info("Successfully registered all ensemble members")
        return ensemble_configuration

    def print_adapter_stack(self):
        print("-" * 20)
        print("Adapter Stack has the following trajectory:")
        print()
        for adapter in self.adapters:
            print(f'{adapter.adapter_name:<20}: {adapter.adapter_task_description}')


if __name__ == '__main__':
    def sample_prepare_input(input_param: dict):
        return "Sample Prompt"


    def clean_output(input_param: str):
        return ["Sample Prompt"]


    adapter_1: EnsembleAdapter = EnsembleAdapter(
        adapter_path='../../output/code_llama_7B_fine_tuned_table_extractions/model_adapter',
        adapter_name='TableExtraction',
        adapter_task_description='Extract the Tables from the query',
        prepare_input_prompt=sample_prepare_input,
        clean_output=clean_output,
        sample_output="t1, t2, t3\n ###End",
        inference_config=CustomInferenceConfig()
    )

    adapter_2: EnsembleAdapter = EnsembleAdapter(
        adapter_path='../../output/code_llama_7B_fine_tuned_table_extractions/model_adapter',
        adapter_name='ColumnExtraction',
        adapter_task_description='Extract the Columns from the query',
        prepare_input_prompt=sample_prepare_input,
        clean_output=clean_output,
        sample_output="t1, t2, t3\n ###End",
        inference_config=CustomInferenceConfig()
    )

    adapter_3: EnsembleAdapter = EnsembleAdapter(
        adapter_path='../../output/code_llama_7B_fine_tuned_table_extractions/model_adapter',
        adapter_name='ComponentExtraction',
        adapter_task_description='Extract typical query components',
        prepare_input_prompt=sample_prepare_input,
        clean_output=clean_output,
        sample_output="t1, t2, t3\n ###End",
        inference_config=CustomInferenceConfig()
    )

    ensemble_stack: EnsembleAdapterStack = EnsembleAdapterStack(
        adapters=[adapter_1, adapter_2, adapter_3]
    )
