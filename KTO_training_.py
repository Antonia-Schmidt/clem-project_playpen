import os
from unsloth import FastLanguageModel, PatchDPOTrainer
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login
from trl import KTOTrainer, KTOConfig
import wandb

login("")
os.environ['WANDB_API_KEY'] = ""
cache_dir_models = ""

wandb.login()
max_seq_length = 1024
dtype = None
load_in_4bit = True

model_name = "mazzaqq/SFT-base_merged_fp16"  #"mazzaqq/SFT-base_merged_fp16"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    cache_dir=cache_dir_models,
)

#TODO: recude 4 variables named 'dataset' in 1
split_portion = ""
dataset = load_dataset("clembench-playpen/binary_dataset_wordle_wordlewithclue", split=f"train{split_portion}") #TODO: transform name into a login variable
dataset_dict = dataset.train_test_split(test_size=0.04)
dataset_new = {'prompt': [], 'completion':[], 'label':[]}

#TODO: put in a function and place it in kto_dataset_creator_turn_scores
for data_point in dataset_dict['train']:
  dataset_new['prompt'].append(data_point['prompt'])
  dataset_new['completion'].append(data_point['completion'])
  dataset_new['label'].append(data_point['label'])
for data_point in dataset_dict['test']:
  dataset_new['prompt'].append(data_point['prompt'])
  dataset_new['completion'].append(data_point['completion'])
  dataset_new['label'].append(data_point['label'])

new_dataset = Dataset.from_dict(dataset_new)
new_dataset = new_dataset.train_test_split(test_size=0.04)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

PatchDPOTrainer()

project_name = "llama3.1_kto_playpen"
entity = "wandb"

wandb.init(project=project_name, name = "llama3.1_kto_playpen")

training_arguments = KTOConfig(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2,
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        learning_rate = 5e-6,
        undesirable_weight = round(new_dataset['train']['label'].count(True) / new_dataset['train']['label'].count(False), 1),
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        report_to="wandb",
        output_dir = "outputs",
        beta = 0.1,
        max_length = max_seq_length,                  #changed also here
        max_prompt_length = max_seq_length,
        )

kto_trainer = KTOTrainer(
    model = model,
    args=training_arguments,
    train_dataset = new_dataset["train"],
    eval_dataset = new_dataset["test"],
    tokenizer = tokenizer,
)

kto_trainer.train()
model.save_pretrained("kto_trained")
model.push_to_hub("clembench-playpen/meta-llama-Meta-Llama-3.1-8B-Instruct-KTO_SFT")
