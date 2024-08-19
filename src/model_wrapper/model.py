from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, pipeline, \
    Pipeline
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from peft import PeftModel
from transformers.pipelines.pt_utils import KeyDataset
from pandas import DataFrame
import os
import gc
import torch
import re
import pandas as pd

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.config.configurations import CustomLoraConfiguration, CustomBitsAndBitesConfiguration, CustomTrainingArguments, \
    CustomInferenceConfig


class CustomTextToSqlModel:
    def __init__(
            self,
            model_name: str,
            model_adapter_name: str,
            path_dataset_train: str,
            path_dataset_inference: str,
            output_dir: str,
            run_name: str,
            lora_config: CustomLoraConfiguration,
            bnb_config: CustomBitsAndBitesConfiguration,
            training_arguments: CustomTrainingArguments,
            inference_config: CustomInferenceConfig,
            max_seq_length: int,
            packing: bool = False,
            device_map: dict = None,
            custom_stopping_criterion: callable = None,
            train: bool = False,
            inference: bool = False,

    ):
        self.model_name: str = model_name
        self.output_dir: str = output_dir
        self.run_name: str = run_name
        self.path_dataset_train: str = path_dataset_train
        self.path_dataset_inference: str = path_dataset_inference
        self.lora_config: CustomLoraConfiguration = lora_config
        self.bnb_config: CustomBitsAndBitesConfiguration = bnb_config
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
        self.dataset_inference: Dataset = None
        self.inference_results: dict = {'raw_output': [], 'clean_output': []}

        # initialize model and tokenizer
        self.tokenizer: PreTrainedTokenizerBase = None
        self.model: PreTrainedModel = None

        # initialize trainer
        self.trainer: SFTTrainer = None

        # build directories
        self.do_train = train
        self.do_inference = inference
        self.directories: dict = self.build_directories()

        # if training is true, build new directory to save the model adapter in
        # if training is false use the given model adapter to allow also external adapters and none for
        # zero shot inference
        self.model_adapter_name: str = self.directories['model_adapter_dir'] if self.do_train else model_adapter_name

    def load_model(self) -> PreTrainedModel:
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

    def load_adapter_model(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("cleared cache to reaload model for inference")
        if self.model_adapter_name is None:
            logging.info("adapter is None thus no adapter will be applied for inference")
            self.model = self.load_model()
            self.adapter_model = self.model
        else:
            logging.info(f"found adapter {self.model_adapter_name} for inference")
            self.model = self.load_model()
            self.adapter_model = PeftModel.from_pretrained(self.model, self.model_adapter_name)

    def load_pipeline(self):
        logging.info("Preparing Pipeline")
        pipe_adapter: Pipeline = pipeline(
            task="text-generation",
            model=self.adapter_model,
            tokenizer=self.tokenizer,
            # repetition_penalty=self.inference_config.repetition_penalty,
            do_sample=self.inference_config.do_sample,
            return_full_text=self.inference_config.return_full_text,
            max_new_tokens=self.inference_config.max_new_tokens
        )

        self.pipeline = pipe_adapter

    def initialize_training(self):
        logging.info("Prepare Model for Training ...")
        if self.model is None:
            self.model = self.load_model()
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer()
        if self.dataset_train is None:
            self.dataset_train = self.load_dataset_train()
        if self.trainer is None:
            self.trainer = self.initialize_trainer()
        # save dataset
        self.dataset_train.to_csv(self.directories['dataset_train'])

    def load_dataset_train(self) -> Dataset:
        logging.info("Loading Dataset for Training")
        ds: Dataset = load_dataset('csv', data_files=self.path_dataset_train, split='train')
        ds.shuffle()
        return ds.shuffle()

    def load_dataset_inference(self) -> Dataset:
        logging.info("Loading Dataset for inference")
        ds: Dataset = load_dataset('csv', data_files=self.path_dataset_inference, split='train')
        return ds

    def initialize_trainer(self) -> SFTTrainer:
        logging.info("Initialize Trainer")
        return SFTTrainer(
            model=self.model,
            train_dataset=self.dataset_train,
            peft_config=self.lora_config.get_lora_config(),
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_arguments.get_training_args(),
            packing=self.packing,
        )

    def train_model(self):
        self.initialize_training()

        logging.info("Start Model Training")
        self.trainer.train()

    def save_model(self):
        self.trainer.model.save_pretrained(self.directories['model_adapter_dir'])

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
        self.dataset_inference.to_csv(self.directories['dataset_test'])

    def inference(self):
        self.initialize_inference()

        logging.info("Start Inference")
        sample = 1
        for result in self.pipeline(KeyDataset(self.dataset_inference, "text"), batch_size=1,
                                    stopping_criteria=[self.custom_stopping_criteria]
                                    ):
            generated_text = result[0]["generated_text"]
            self.inference_results["raw_output"].append(generated_text)
            self.inference_results["clean_output"].append(self.clean_prediction(generated_text))
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
        pattern = re.compile(r'SELECT.*?(\n|\r\n|\r|;)', re.DOTALL)

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

    def custom_stopping_criteria(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        decoded_text = self.tokenizer.batch_decode(input_ids)
        t1: bool = "###Task:" in decoded_text[0][-15:]
        t2: bool = "###Question:" in decoded_text[0][-15:]
        t3: bool = "###Input" in decoded_text[0][-15:]
        t4: bool = ';' in decoded_text[0][-15:]
        t5: bool = '###End' in decoded_text[0][-15:]
        t6: bool = '###' in decoded_text[0][-15:]
        t7: bool = "### Response" in decoded_text[0][-15:]
        t8: bool = "###Instruction" in decoded_text[0][-15:]

        t9: bool = '### End' in decoded_text[0][-7:]
        t10: bool = '###End' in decoded_text[0][-7:]
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
        dataset_dir = root_path + '/dataset'

        inference_results_raw: str = inference_dir + "/inference_predictions.csv"
        dataset_train: str = dataset_dir + '/training_data.csv'
        dataset_test: str = dataset_dir + '/dev_data.csv'

        if os.path.exists(path=root_path):
            # check if only inference is done and if the inference dir is empty
            if self.do_inference and not self.do_train and not os.path.exists(path=inference_results_raw):
                logging.info(f"Output directory {self.run_name} exists but inference is empty and can be used")

            # check if only training is done and if the model adapter dir is empty
            elif not self.do_inference and self.do_train and not os.path.exists(path=model_adapter_dir):
                logging.info(f"Output directory {self.run_name} exists but inference is empty and can be used")

            # to prevent overriding existing runs
            else:
                logging.warning(
                    f"run with name {self.run_name} already exists! please chose another name or clear directories")
                exit(69)

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

        return {
            'root_dir': root_path,
            'model_adapter_dir': model_adapter_dir,
            'inference_dir': inference_dir,
            'dataset_dir': dataset_test,
            'inference_results_raw': inference_results_raw,
            'dataset_train': dataset_train,
            'dataset_test': dataset_test,
        }
