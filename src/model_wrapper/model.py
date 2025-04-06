from unsloth import FastLanguageModel, unsloth_save_model
from unsloth.chat_templates import get_chat_template

import datetime
import gc
import json
import logging
import os
import re
from huggingface_hub import create_repo


import pandas as pd
import torch
from datasets import Dataset, load_dataset
from pandas import DataFrame
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    pipeline,
    DataCollatorForLanguageModeling
)

from transformers.pipelines.pt_utils import KeyDataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from huggingface_hub import create_repo

from typing import Any, Dict, List

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# set Environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "clembench-playpen-sft"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"  # log all model checkpoints


from src.config.configurations import (
    CustomBitsAndBitesConfiguration,
    CustomInferenceConfig,
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
        path_dataset_inference: str,
        output_dir: str,
        lora_config: CustomLoraConfiguration,
        bnb_config: CustomBitsAndBitesConfiguration,
        unsloth_config: CustomUnslothModelConfig,
        training_arguments: CustomTrainingArguments,
        inference_config: CustomInferenceConfig,
        max_seq_length: int,
        packing: bool = False,
        device_map: dict = None,
        custom_stopping_criterion: callable = None,
        train: bool = False,
        inference: bool = False,
        model_adapter: str = None,
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
        self.path_dataset_inference: str = path_dataset_inference
        self.lora_config: CustomLoraConfiguration = lora_config
        self.bnb_config: CustomBitsAndBitesConfiguration = bnb_config
        self.unsloth_config: CustomUnslothModelConfig = unsloth_config
        self.inference_config: CustomInferenceConfig = inference_config
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
        self.dataset_inference: Dataset = None
        self.inference_results: dict = {"raw_output": [], "clean_output": []}

        # initialize model and tokenizer
        self.tokenizer: PreTrainedTokenizerBase = None
        self.model: PreTrainedModel = None

        # initialize trainer
        self.trainer: SFTTrainer = None

        # build directories
        self.do_train = train
        self.do_inference = inference
        self.directories: dict = self.build_directories()

        # update training args for propper logging
        self.training_arguments.run_name = run_name
        self.training_arguments.output_dir = self.directories["model_dir"]

        # if training is true, build new directory to save the model adapter in
        # if training is false use the given model adapter to allow also external adapters and none for
        # zero shot inference
        self.model_adapter_name: str = (
            self.directories["model_adapter_dir"] if self.do_train else model_adapter
        )

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

    def save_config(self):
        file: str = self.directories["config_dir"] + "/configuration.txt"
        with open(file, "w") as convert_file:
            convert_file.write(json.dumps(self.bnb_config.as_dict()))
            convert_file.write(json.dumps(self.training_arguments.as_dict()))
            convert_file.write(json.dumps(self.unsloth_config.as_dict()))
            convert_file.write(json.dumps(self.lora_config.as_dict()))

    def load_adapter_model(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("cleared cache to reload model for inference")
        if self.model_adapter_name is None:
            logging.info(
                "adapter is None thus no adapter will be applied for inference"
            )
            self.load_model_and_tokenizer()
            self.adapter_model = self.model
        else:
            logging.info(f"found adapter {self.model_adapter_name} for inference")
            self.load_model_and_tokenizer()
            self.adapter_model = PeftModel.from_pretrained(
                self.model, self.model_adapter_name
            )

    def load_pipeline(self):
        logging.info("Preparing Pipeline")
        pipe_adapter: Pipeline = pipeline(
            task="text-generation",
            model=self.adapter_model,
            tokenizer=self.tokenizer,
            # repetition_penalty=self.inference_config.repetition_penalty,
            do_sample=self.inference_config.do_sample,
            return_full_text=self.inference_config.return_full_text,
            max_new_tokens=self.inference_config.max_new_tokens,
        )

        self.pipeline = pipe_adapter

    def initialize_training(self):
        logging.info("Prepare Model for Training ...")
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
    
    def initialize_training_with_collator(self):
        logging.info("Prepare Model for Training ...")
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

    def initialize_training_with_multi_step_collator(self):
        logging.info("Prepare Model for Training ...")
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

    def load_dataset_train(self) -> Dataset:
        logging.info("Loading Dataset for Training")

        df_data: DataFrame = pd.read_csv(self.path_dataset_train, index_col=0)
        ds: Dataset = Dataset.from_pandas(df=df_data)

        # return ds.shuffle()

        # return without shuffle
        logging.warning("Loading Dataset without Shuffling, Ignore the warning if the dataset was shuffeled before running the traning!")
        return ds
    
    def load_dataset_from_path(self, path: str):
        print("Loading Dataset from path",  path)

        df_data: DataFrame = pd.read_csv(path)
        ds: Dataset = Dataset.from_pandas(df=df_data)

        # return without shuffle
        logging.warning("Loading Dataset without Shuffling, Ignore the warning if the dataset was shuffeled before running the traning!")
        return ds

    def load_dataset_test(self) -> Dataset:
        logging.info("Loading Dataset for Testing")

        df_data: DataFrame = pd.read_csv(self.path_dataset_test, index_col=0)
        ds: Dataset = Dataset.from_pandas(df=df_data)
        # return ds.shuffle()

        # return without shuffle
        logging.warning("Loading Dataset without Shuffling, Ignore the warning if the dataset was shuffeled before running the traning!")
        return ds

    def load_dataset_inference(self) -> Dataset:
        logging.info("Loading Dataset for inference")
        ds: Dataset = load_dataset(
            "csv", data_files=self.path_dataset_inference, split="train"
        )
        return ds

    def initialize_trainer(self) -> SFTTrainer:
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
        )
    
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
        
        return SFTTrainer(
            model=self.model,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_test,
            peft_config=self.lora_config.get_lora_config(),
            dataset_text_field="text",
            max_seq_length=self.unsloth_config.max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_arguments.get_training_args(),
            packing=False,  # Must be False when using completion-only training
            dataset_num_proc=1,
            data_collator=collator
        )

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
        
        return SFTTrainer(
            model=self.model,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_test,
            peft_config=self.lora_config.get_lora_config(),
            dataset_text_field="text",
            max_seq_length=self.unsloth_config.max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_arguments.get_training_args(),
            packing=False,  # Must be False when using completion-only training
            dataset_num_proc=1,
            data_collator=collator
        )

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
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
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


        trainer = SFTTrainer(
            model=model,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_test,  # Add this line
            peft_config=self.lora_config.get_lora_config(),
            dataset_text_field="text",
            max_seq_length=self.unsloth_config.max_seq_length,
            tokenizer=tokenizer,
            args=self.training_arguments.get_training_args(),
            packing=self.packing,
            dataset_num_proc=1,
        )
        self.trainer = trainer

    def merge_adapter_and_model():
        # merge the adapter weigths into the model
        pass

    def train_model(self):
        self.initialize_training()

        logging.info("Start Model Training")
        self.trainer.train()
    
    def train_model_with_collator(self):
        self.initialize_training_with_collator()
        logging.info("Start Model Training with collator for completion only")
        self.trainer.train()

    def train_model_with_multi_step_collator(self):
        self.initialize_training_with_multi_step_collator()
        print(self.tokenizer.pad_token)

        self.trainer.train()

    def train_full_precision_LoRA(self):
        self.initialize_training_with_full_precision_LoRA()
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
        print("Starting Warmup")
        warmup_trainer.train()
        # Verify adapter continuity
        print(f"Active adapters after warmup: {self.model.active_adapters}")

        # Main training phase (using primary dataset)
        main_trainer = self.initialize_trainer()
        
        # Start main training
        print("Starting main training")
        main_trainer.train()
        print(f"Active adapters after warmup: {self.model.active_adapters}")
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
        print("Starting Warmup")
        warmup_trainer.train()
        # Verify adapter continuity
        print(f"Active adapters after warmup: {self.model.active_adapters}")

        # Main training phase (using primary dataset)
        main_trainer = self.initialize_trainer_with_collator()
        
        # Start main training
        print("Starting main training")
        main_trainer.train()
        print(f"Active adapters after warmup: {self.model.active_adapters}")

        self.trainer = main_trainer

    def train_model_with_periodic_save(self, start_index: int, output_base_path: str):
        self.initialize_training()
        logging.info("Start Model Training")
    
        # Access the training arguments for number of epochs
        num_epochs = self.training_arguments.num_train_epochs
    
        for epoch in range(int(num_epochs)):
            run_number = start_index + epoch

            print("RUN NUMBER: ", run_number)
            if run_number <= 9:
                experiment_name = f'DT000{run_number}'
            elif run_number <= 99:
                experiment_name = f'DT00{run_number}'
            else:
                experiment_name = f'DT0{run_number}'

            logging.info(f"Starting Epoch {epoch + 1}/{num_epochs}")

            # generate the model hub ID
            model_hub_id: str = f'clembench-playpen/{self.model_name.replace("/", "-")}_SFT_E{epoch + 1}_{experiment_name}'
            print("Training model with hub_id: ", model_hub_id)

            # create repo
            create_repo(repo_id=model_hub_id, repo_type="model", private=False, exist_ok=True)

            # set model hub ID
            self.trainer.hub_model_id = model_hub_id

            # create sub directory for model
            save_dir = output_base_path + f'/{experiment_name}'
            config_dir = save_dir + '/config'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(config_dir, exist_ok=True)

            # save the configurations
            file: str = config_dir + "/configuration.txt"
            with open(file, "w") as convert_file:
                convert_file.write(json.dumps(self.bnb_config.as_dict()))
                convert_file.write(json.dumps(self.training_arguments.as_dict()))
                convert_file.write(json.dumps(self.unsloth_config.as_dict()))
                convert_file.write(json.dumps(self.lora_config.as_dict()))
    
            # Train for one epoch
            self.trainer.train(resume_from_checkpoint=None)

            # save the model
            unsloth_save_model(
                self.trainer.model,
                self.trainer.tokenizer,
                save_dir,
                push_to_hub=False,
                token=None,
            )
            self.trainer.push_to_hub()

            logging.info(f"Model saved at {save_dir}")


    def save_model(self):
        self.trainer.push_to_hub()

        unsloth_save_model(
            self.trainer.model,
            self.trainer.tokenizer,
            self.directories["model_adapter_dir"],
            push_to_hub=False,
            token=None,
        )
    def initialize_inference(self):
        logging.info("Prepare Model for Inference")
        self.model = self.load_model()

        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer()
        if self.adapter_model is None:
            self.load_adapter_model()
        if self.pipeline is None:
            self.load_pipeline()
        if self.dataset_inference is None:
            self.dataset_inference = self.load_dataset_inference()
        # save dataset
        self.dataset_inference.to_csv(self.directories["dataset_test"])

    def inference(self):
        self.initialize_inference()

        logging.info("Start Inference")
        sample = 1
        for result in self.pipeline(
            KeyDataset(self.dataset_inference, "text"),
            batch_size=1,
            stopping_criteria=[self.custom_stopping_criteria],
        ):
            generated_text = result[0]["generated_text"]
            self.inference_results["raw_output"].append(generated_text)
            self.inference_results["clean_output"].append(
                self.clean_prediction(generated_text)
            )
            print(f"{sample}/{len(self.dataset_inference)}")
            print(generated_text)
            sample += 1

        self.save_inference()

    def clean_prediction(self, query):
        """
        Do basic cleanup of the prediction by cutting off all unnecessary information
        :param query: the query
        :return: the cleaned query
        """
        # Regular expression pattern start from select and cut-off after /n
        pattern = re.compile(r"SELECT.*?(\n|\r\n|\r|;)", re.DOTALL)

        # Find all matches
        match = re.search(pattern, query)
        if not match:
            clean_output = query
        else:
            clean_output = match.group()

        clean_output = re.sub("\n", "", clean_output)
        clean_output = re.sub(";", "", clean_output)

        # add spaces around the operators
        operators = ["<=", "!=", ">="]
        for operator in operators:
            clean_output = re.sub(f"{operator}", f" {operator} ", clean_output)

        # remove multiple whitespaces
        clean_output = clean_output.strip()
        clean_output = re.sub(r"\s+", " ", clean_output)

        return clean_output

    def custom_stopping_criteria(
        self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs
    ) -> bool:
        decoded_text = self.tokenizer.batch_decode(input_ids)
        t1: bool = "###Task:" in decoded_text[0][-15:]
        t2: bool = "###Question:" in decoded_text[0][-15:]
        t3: bool = "###Input" in decoded_text[0][-15:]
        t4: bool = ";" in decoded_text[0][-15:]
        t5: bool = "###End" in decoded_text[0][-15:]
        t6: bool = "###" in decoded_text[0][-15:]
        t7: bool = "### Response" in decoded_text[0][-15:]
        t8: bool = "###Instruction" in decoded_text[0][-15:]

        t9: bool = "### End" in decoded_text[0][-7:]
        t10: bool = "###End" in decoded_text[0][-7:]
        return t9 or t10

    def save_inference(self):
        result_data: DataFrame = pd.DataFrame(self.inference_results)
        result_data.to_csv(self.directories["inference_results_raw"], index=False)

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

