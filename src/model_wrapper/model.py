from unsloth import FastLanguageModel, unsloth_save_model
from unsloth.chat_templates import get_chat_template

import datetime
import json
import logging
import os


import pandas as pd
import torch
from datasets import Dataset, load_dataset
from pandas import DataFrame
from transformers import (
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling
)

from transformers.pipelines.pt_utils import KeyDataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# set Environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "clembench-playpen-sft"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"  # log all model checkpoints


from src.config.configurations import (
    CustomBitsAndBitesConfiguration,
    CustomLoraConfiguration,
    CustomTrainingArguments,
    CustomUnslothModelConfig,
)


class CustomTextToSqlModel:
    def __init__(
        self,
        model_name: str,
        chat_template: str,
        path_dataset_train: str,
        path_dataset_test: str,
        output_dir: str,
        lora_config: CustomLoraConfiguration,
        bnb_config: CustomBitsAndBitesConfiguration,
        unsloth_config: CustomUnslothModelConfig,
        training_arguments: CustomTrainingArguments,
        max_seq_length: int,
        packing: bool = False,
        device_map: dict = None,
        custom_stopping_criterion: callable = None,
        train: bool = False,
        inference: bool = False,
    ):
        # get the current time for the run
        now = datetime.datetime.now()
        run_name: str = (
            now.strftime("%Y-%m-%dT%H-%M-%S")
            + f'_{training_arguments.hub_model_id.replace("/", "_")}'
        )
        print(run_name)
        print(device_map)

        self.model_name: str = model_name
        self.chat_template = chat_template
        self.output_dir: str = output_dir
        self.run_name: str = run_name
        self.path_dataset_train: str = path_dataset_train
        self.path_dataset_test: str = path_dataset_test
        self.lora_config: CustomLoraConfiguration = lora_config
        self.bnb_config: CustomBitsAndBitesConfiguration = bnb_config
        self.unsloth_config: CustomUnslothModelConfig = unsloth_config
        self.device_map: dict = device_map if device_map is not None else {"": 0}
        self.training_arguments: CustomTrainingArguments = training_arguments
        self.max_seq_length: int = max_seq_length
        self.packing: bool = packing
        self.custom_stopping_criterion: callable = custom_stopping_criterion

        # when doing inference this model is loaded as well pipeline
        self.adapter_model = None
        self.pipeline: Pipeline = None

        # load dataset
        self.dataset_train: Dataset = None
        self.dataset_test: Dataset = None

        # initialize model and tokenizer
        self.tokenizer: PreTrainedTokenizerBase = None
        self.model: PreTrainedModel = None

        # initialize trainer
        self.trainer: SFTTrainer = None

        # build directories
        self.do_train = train
        self.do_inference = inference
        self.directories: dict = self.build_directories()
        self.model_adapter_name: str = self.directories["model_adapter_dir"]


        # update training args for propper logging
        self.training_arguments.run_name = run_name
        self.training_arguments.output_dir = self.directories["model_dir"]

    """
    Load the model and tokenizer using unsloth FastLanguageModel from pretrained model weights
    """
    def load_model_and_tokenizer(self):
        logging.info(f"Loading model and tokenizer for: {self.model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.unsloth_config.max_seq_length,
            dtype=self.unsloth_config.dtype,
            load_in_4bit=self.unsloth_config.load_in_4_bit,
            fix_tokenizer=False
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

        # transform into peft model
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_config.lora_r,
            target_modules=self.lora_config.target_modules,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,  # Currently, only supports dropout = 0
            bias=self.lora_config.lora_bias,  # Currently, only supports bias = "none"
            use_gradient_checkpointing=self.unsloth_config.use_gradient_checkpointing,
            random_state=7331,
            max_seq_length=self.unsloth_config.max_seq_length,
            use_rslora=self.lora_config.use_rslora,  # We support rank stabilized LoRA
            loftq_config=self.lora_config.loftq_config,  # And LoftQ
        )

        # add chat-tempalte to tokenizer
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=self.chat_template,  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        )

        # set model and tokenizer
        self.model = model
        self.tokenizer = tokenizer

    """
    Safe the configuration settings for repeatability
    """
    def save_config(self):
        file: str = self.directories["config_dir"] + "/configuration.txt"
        logging.info(f"Saving cofigurations to {file}")

        with open(file, "w") as convert_file:
            convert_file.write(json.dumps(self.bnb_config.as_dict()))
            convert_file.write(json.dumps(self.training_arguments.as_dict()))
            convert_file.write(json.dumps(self.unsloth_config.as_dict()))
            convert_file.write(json.dumps(self.lora_config.as_dict()))


    """
    Load model, datasets and initialize trainer for cold start full prompt training
    """
    def initialize_training(self):
        logging.info("Prepare Model for cold start training ...")
        if self.model is None:
            self.load_model_and_tokenizer()
        if self.dataset_train is None:
            self.dataset_train = self.load_dataset_train()
        if self.dataset_test is None:
            self.dataset_test = self.load_dataset_test()
        if self.trainer is None:
            self.trainer = self.initialize_trainer()
        # save dataset
        self.dataset_train.to_csv(self.directories["dataset_train"])

    """
    Load model, datasets and initialize trainer for training with a completion only collator
    """
    def initialize_training_with_collator(self):
        logging.info("Prepare Model for completion only collator training ...")
        if self.model is None:
            self.load_model_and_tokenizer()
        if self.dataset_train is None:
            self.dataset_train = self.load_dataset_train()
        if self.dataset_test is None:
            self.dataset_test = self.load_dataset_test()
        if self.trainer is None:
            self.trainer = self.initialize_trainer_with_collator()
        # save dataset
        self.dataset_train.to_csv(self.directories["dataset_train"])

    """
    Load model, datasets and initialize trainer for training with a multistep collator
    """
    def initialize_training_with_multi_step_collator(self):
        logging.info("Prepare model for training with multi-step collator training...")
        if self.model is None:
            self.load_model_and_tokenizer()
        if self.dataset_train is None:
            self.dataset_train = self.load_dataset_train()
        if self.dataset_test is None:
            self.dataset_test = self.load_dataset_test()
        if self.trainer is None:
            self.trainer = self.initialize_trainer_with_multi_turn_collator()
        # save dataset
        self.dataset_train.to_csv(self.directories["dataset_train"])

    """
    Load the training dataset
    """
    def load_dataset_train(self) -> Dataset:
        logging.info("Loading Dataset for Training")

        df_data: DataFrame = pd.read_csv(self.path_dataset_train, index_col=0)
        ds: Dataset = Dataset.from_pandas(df=df_data)

        # dataset is assumed to be shuffled already
        # return ds.shuffle()

        # return without shuffle
        logging.warning("Loading Dataset without Shuffling, Ignore the warning if the dataset was shuffeled before running the traning!")
        return ds

    """
    Load a dataset from a given path for e.g. warm-up or rehearsal training
    """
    def load_dataset_from_path(self, path: str):
        print("Loading Dataset from path",  path)

        df_data: DataFrame = pd.read_csv(path)
        ds: Dataset = Dataset.from_pandas(df=df_data)

        # dataset is assumed to be shuffled already
        # return ds.shuffle()

        # return without shuffle
        logging.warning("Loading Dataset without Shuffling, Ignore the warning if the dataset was shuffeled before running the traning!")
        return ds

    """
    Load the dev set from self.test path
    """
    def load_dataset_test(self) -> Dataset:
        logging.info("Loading development data to evaluate the training")

        df_data: DataFrame = pd.read_csv(self.path_dataset_test, index_col=0)
        ds: Dataset = Dataset.from_pandas(df=df_data)

        # dataset is assumed to be shuffled already
        # return ds.shuffle()

        # return without shuffle
        logging.warning("Loading Dataset without Shuffling, Ignore the warning if the dataset was shuffeled before running the traning!")
        return ds

    """
    Initialize trainer for cold start training
    """
    def initialize_trainer(self, collator=None) -> SFTTrainer:
        logging.info("Initialize Trainer")
        return SFTTrainer(
            model=self.model,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_test,  # Add this line
            peft_config=self.lora_config.get_lora_config(),
            dataset_text_field="text",
            max_seq_length=self.unsloth_config.max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_arguments.get_training_args(),
            packing=self.packing,
            dataset_num_proc=1,
            data_collator=collator,
        )

    """
    Initialize trainer for completion only collator trianing
    """
    def initialize_trainer_with_collator(self) -> SFTTrainer:
        logging.info("Initialize Trainer")
        
        # Define the response template that marks where assistant responses begin
        # This should match exactly how assistant responses are marked in your tokenized text
        response_template_with_context = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_with_context,
            tokenizer=self.tokenizer,
            mlm=False
        )

        # must be false for training collator
        self.packing = False

        return self.initialize_trainer(collator)

    """
    Initialize trainer for multistep collator training
    """
    def initialize_trainer_with_multi_turn_collator(self) -> SFTTrainer:
        logging.info("Initialize Trainer")
        
        # Define the response template that marks where assistant responses begin
        # This should match exactly how assistant responses are marked in your tokenized text
        collator = SpecialTokenCollator(
            assistant_header="<|start_header_id|>assistant<|end_header_id|>",
            eot_token="<|eot_id|>",
            tokenizer=self.tokenizer,
            mlm=False
        )

        # must be false for training collator
        self.packing = False

        return self.initialize_trainer(collator)

    """
    Initialize trainer for warm-up training
    """
    def initialize_trainer_with_collator_warmup(self, warm_up_set) -> SFTTrainer:
        logging.info("Initialize Trainer")
        
        # Define the response template that marks where assistant responses begin
        # This should match exactly how assistant responses are marked in your tokenized text
        # response_template_with_context = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        response_template_with_context = "[/INST]"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_with_context,
            tokenizer=self.tokenizer,
            mlm=False
        )   

        # override the training args to train the full set for warmup
        args = self.training_arguments.get_training_args()
        args.num_train_epochs = 1
        args.max_steps = 0
        args.eval_steps = 20
        
        return SFTTrainer(
            model=self.model,
            train_dataset=warm_up_set,
            eval_dataset=self.dataset_test,
            peft_config=self.lora_config.get_lora_config(),
            dataset_text_field="text",
            max_seq_length=self.unsloth_config.max_seq_length,
            tokenizer=self.tokenizer,
            args=args,
            packing=False,  # Must be False when using completion-only training
            dataset_num_proc=1,
            data_collator=collator
        )
    
    def initialize_training_with_full_precision_LoRA(self):
        # load dataset
        self.dataset_train = self.load_dataset_train()
        self.dataset_test = self.load_dataset_test()
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.unsloth_config.max_seq_length,
            dtype=self.unsloth_config.dtype,
            load_in_4bit=True,
            fix_tokenizer=False
        )

        model = FastLanguageModel.get_peft_model(
            model,  # Specify the existing model
            r=self.lora_config.lora_r,
            target_modules=self.lora_config.target_modules,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,  # Currently, only supports dropout = 0
            bias=self.lora_config.lora_bias,  # Currently, only supports bias = "none"
            use_gradient_checkpointing=self.unsloth_config.use_gradient_checkpointing,
            random_state=7331,
            max_seq_length=self.unsloth_config.max_seq_length,
            use_rslora=self.lora_config.use_rslora,  # We support rank stabilized LoRA
            loftq_config=self.lora_config.loftq_config,  # And LoftQ
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template=self.chat_template,  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        )

        # set model and tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = self.initialize_trainer()

    def train_model(self):
        self.initialize_training()

        logging.info("Start Model Training")
        self.trainer.train()
    
    def train_model_with_completion_only_collator(self):
        self.initialize_training_with_collator()

        logging.info("Start Model Training with collator for completion only")
        self.trainer.train()

    def train_model_with_multi_step_completion_collator(self):
        self.initialize_training_with_multi_step_collator()
        print(self.tokenizer.pad_token)

        self.trainer.train()

    def train_full_precision_LoRA(self):
        self.initialize_training_with_full_precision_LoRA()
        logging.info("Start Model Training with full precision")

        self.trainer.train()

    def train_model_with_wramup(self, path_warmup_ds: str):
        # prepare datasets
        dataset_warmup = self.load_dataset_from_path(path_warmup_ds)
        self.dataset_test = self.load_dataset_test()
        self.dataset_train = self.load_dataset_train()

        self.load_model_and_tokenizer()
      

        # Warmup phase (using warmup dataset)
        warmup_trainer = self.initialize_trainer_with_collator_warmup(dataset_warmup)
        self.model = warmup_trainer.model
        logging.info("Start Model Training with warm-up")

        warmup_trainer.train()
        # Verify adapter continuity
        logging.info(f"Active adapters after warmup: {self.model.active_adapters}")

        # Main training phase (using primary dataset)
        main_trainer = self.initialize_trainer()
        
        # Start main training
        logging.info("Starting main training")
        main_trainer.train()

        self.trainer = main_trainer

    
    def train_model_with_wramup_and_completion_only(self, path_warmup_ds: str):
        # prepare datasets
        dataset_warmup = self.load_dataset_from_path(path_warmup_ds)
        self.dataset_test = self.load_dataset_test()
        self.dataset_train = self.load_dataset_train()

        self.load_model_and_tokenizer()
      

        # Warmup phase (using warmup dataset)
        warmup_trainer = self.initialize_trainer_with_collator_warmup(dataset_warmup)
        self.model = warmup_trainer.model
        logging.info("Start Model Training with warm-up")
        warmup_trainer.train()
        # Verify adapter continuity
        logging.info(f"Active adapters after warmup: {self.model.active_adapters}")

        # Main training phase (using primary dataset)
        main_trainer = self.initialize_trainer_with_collator()
        
        # Start main training
        logging.info("Starting main training")
        main_trainer.train()

        self.trainer = main_trainer

    def save_model(self):
        self.trainer.push_to_hub()

        unsloth_save_model(
            self.trainer.model,
            self.trainer.tokenizer,
            self.directories["model_adapter_dir"],
            push_to_hub=False,
            token=None,
        )


    def build_directories(self):
        """
        Build the directory for this particular inference or training run:
        base_dir:(self.output)
            |_unique_run_name
                |_model_adapter
                |_inference_results
                |_dataset
        :return: a dictionary containing all paths to save results
        """
        root_path = os.path.join(self.output_dir, self.run_name)
        model_adapter_dir = root_path + "/model_adapter"
        inference_dir = root_path + "/inference"
        dataset_dir = root_path + "/dataset"
        config_dir = root_path + "/config"
        model_dir = root_path + "/model"

        inference_results_raw: str = inference_dir + "/inference_predictions.csv"
        dataset_train: str = dataset_dir + "/training_data.csv"
        dataset_test: str = dataset_dir + "/dev_data.csv"

        if os.path.exists(path=root_path):
            # check if only inference is done and if the inference dir is empty
            if (
                self.do_inference
                and not self.do_train
                and not os.path.exists(path=inference_results_raw)
            ):
                logging.info(
                    f"Output directory {self.run_name} exists but inference is empty and can be used"
                )

            # check if only training is done and if the model adapter dir is empty
            elif (
                not self.do_inference
                and self.do_train
                and not os.path.exists(path=model_adapter_dir)
            ):
                logging.info(
                    f"Output directory {self.run_name} exists but inference is empty and can be used"
                )

            # to prevent overriding existing runs
            else:
                logging.warning(
                    f"run with name {self.run_name} already exists! please chose another name or clear directories"
                )
                exit(69)

        # create the output folder
        if not os.path.exists(path=self.output_dir):
            os.mkdir(self.output_dir)
            logging.info(f"created new directory: {self.output_dir}")

        # create the adapter folder
        if not os.path.exists(path=root_path):
            os.mkdir(root_path)
            logging.info(f"created new directory: {root_path}")

        # create the inference folder
        if not os.path.exists(path=inference_dir):
            os.mkdir(inference_dir)
            logging.info(f"created new directory: {inference_dir}")

        # create the inference folder
        if not os.path.exists(path=dataset_dir):
            os.mkdir(dataset_dir)
            logging.info(f"created new directory: {dataset_dir}")

        # create the config directory
        if not os.path.exists(path=config_dir):
            os.mkdir(config_dir)
            logging.info(f"created new directory: {config_dir}")

        # create the config directory
        if not os.path.exists(path=model_dir):
            os.mkdir(model_dir)
            logging.info(f"created new directory: {model_dir}")

        return {
            "root_dir": root_path,
            "model_adapter_dir": model_adapter_dir,
            "inference_dir": inference_dir,
            "dataset_dir": dataset_test,
            "inference_results_raw": inference_results_raw,
            "dataset_train": dataset_train,
            "dataset_test": dataset_test,
            "config_dir": config_dir,
            "model_dir": model_dir,
        }

class SpecialTokenCollator(DataCollatorForLanguageModeling):
    def __init__(self, assistant_header, eot_token, tokenizer, *args, **kwargs):
        super().__init__(tokenizer=tokenizer, *args, **kwargs)
        self.assistant_header = tokenizer.encode(
            assistant_header, add_special_tokens=False
        )
        self.eot_token = tokenizer.encode(eot_token, add_special_tokens=False)[0]
        
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        
        for i, input_ids in enumerate(batch["input_ids"]):
            # Find all assistant headers
            header_positions = []
            curr_idx = 0
            while curr_idx <= len(input_ids) - len(self.assistant_header):
                if input_ids[curr_idx:curr_idx+len(self.assistant_header)].tolist() == self.assistant_header:
                    header_positions.append(curr_idx)
                    curr_idx += len(self.assistant_header)
                else:
                    curr_idx += 1
            
            # Create mask for assistant responses
            if header_positions:
                loss_mask = torch.zeros_like(input_ids)
                for pos in header_positions:
                    start = pos + len(self.assistant_header)
                    end = self.find_eot(input_ids, start)
                    loss_mask[start:end] = 1
                
                batch["labels"][i] = torch.where(
                    loss_mask.bool(),
                    batch["labels"][i],
                    -100
                )
                
        return batch
    
    def find_eot(self, input_ids, start_idx):
        eot_positions = (input_ids[start_idx:] == self.eot_token).nonzero()
        if eot_positions.numel() > 0:
            eot_pos = start_idx + eot_positions[0].item()
            # Return position AFTER the EOT token (+0 includes EOT, +1 would go past it)
            return eot_pos + 1  # Include EOT itself in loss
        return len(input_ids)

