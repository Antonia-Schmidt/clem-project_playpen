import os
from unsloth import FastLanguageModel, PatchDPOTrainer
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login, create_repo
from trl import DPOTrainer, DPOConfig
import wandb
import argparse
import time
import json

def load_hf_dataset(tokenizer):

    #TODO: il modello e il dataset di training non è detto che siano nello stesso posto, differenzia le variabili hf_repo_model e hf_repo_datasets
    dataset = load_dataset(f"{args.hf_repo}/{args.dataset_name}", split = "train")

    dataset_dict = dataset.train_test_split(test_size=0.04)
    dataset_new = {'chosen':[], 'rejected':[]}
    #TODO: reduce to one line the following (one function)
    for data_point in dataset_dict['train']['chosen']:
      dataset_new['chosen'].append(tokenizer.apply_chat_template(data_point, tokenize=False, add_generation_prompt=False))
    for data_point in dataset_dict['train']['rejected']:
      dataset_new['rejected'].append(tokenizer.apply_chat_template(data_point, tokenize=False, add_generation_prompt=False))
    for data_point in dataset_dict['test']['chosen']:
      dataset_new['chosen'].append(tokenizer.apply_chat_template(data_point, tokenize=False, add_generation_prompt=False))
    for data_point in dataset_dict['test']['rejected']:
      dataset_new['rejected'].append(tokenizer.apply_chat_template(data_point, tokenize=False, add_generation_prompt=False))
    new_dataset = Dataset.from_dict(dataset_new)
    dataset_dict = new_dataset.train_test_split(test_size=0.04)
    return dataset_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DPO training')
    #TODO: infer model condition from the dataset and not given as parameter
    parser.add_argument('--aborted_interactions', default=True, choices=[True, False], help='integrating aborted interactions as negative samples')
    #TODO: take this out in common with DPO_training.py and KTO_training.py
    parser.add_argument('--hf_login', default="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", help='hf login token')
    parser.add_argument('--wandb_login', default="29fc614754925fca7f38d6d2193b3f5afa8485a9", help='wandb login token')
    parser.add_argument('--base_model', default="llama-SFT-base_merged_fp16_D90053_copy_32GB", help='base model for training')
    parser.add_argument('--dataset_name', help='base model for training')
    parser.add_argument('--cache_dir', default = '/mnt/cimec-storage6/shared/hf_llms_checkpoints/', help='cache directory to store models')
    parser.add_argument('--hf_repo', default='clembench-playpen', help='huggingface repository to store the created datasets')
    args = parser.parse_args()

    login(f"{args.hf_login}")
    os.environ['WANDB_API_KEY'] = f"{args.wandb_login}"
    wandb.login()
    cache_dir_models = args.cache_dir

    max_seq_length = 1024
    model_name = f"{args.hf_repo}/{args.base_model}"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        cache_dir=cache_dir_models,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer.truncation_side = 'left'
    dataset_dict = load_hf_dataset(tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    PatchDPOTrainer()

    project_name = f"playpen_{args.base_model}"
    entity = "wandb"
    wandb.init(project=project_name, name=f"dpo_{args.dataset_name}")

    training_arguments = DPOConfig(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=3,
        warmup_ratio=0.1,
        num_train_epochs=1,
        learning_rate=5e-6,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        # max_steps=20,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="linear",
        seed=42,
        report_to="wandb",
        #push_to_hub=False,
        output_dir="./outputs", #f"{args.hf_repo}/meta-llama-Meta-Llama-3.1-8B-Instruct_DPO_{args.neg}neg{'_'+args.model_condition if args.model_condition else ''}_sblurry___",       #TODO: this does not work with dpo_trainer.push_to_hub
        logging_steps=1,
        #TODO: use this lines to avoid overfitting with old and new data
        eval_steps=0.2,  # 0.2
        save_steps=0.2,  # 0.2
        eval_strategy = "steps",
        logging_strategy = "steps",
        save_strategy = "steps",
        load_best_model_at_end=True
)

    dpo_trainer = DPOTrainer(
        model=model,
        args=training_arguments,
        beta=0.1,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        tokenizer=tokenizer,
        max_length=max_seq_length,
        max_prompt_length=max_seq_length,
    )

    #TODO: migliora questa funzione (riduci, sposta)
    train_dataloader = dpo_trainer.get_train_dataloader()
    #Modified from the original HF trainer.py
    def compute_tokens(train_dl: train_dataloader) -> int:
        train_tokens = 0
        for batch in train_dl:
            tokens_chosen = batch["chosen_input_ids"].numel()
            tokens_rejected = batch["rejected_input_ids"].numel()
            tokens = tokens_chosen + tokens_rejected
            train_tokens += tokens
        return train_tokens

    training_tokens = compute_tokens(train_dataloader)

    #TODO: restore the second line (to modify btw)
    #trained_model_id = f"meta-llama-3.1_DPO_{args.neg}neg{'_Aborted' if args.aborted_interactions else ''}{'_'+args.model_condition if args.model_condition else ''}_END_07"
    #trained_model_id = f"{args.base_model}_{args.dataset_name}"
    trained_model_id = f"D40005_{args.dataset_name}"

    model_hub_id = f"{args.hf_repo}/{trained_model_id}"

    create_repo(repo_id=model_hub_id, repo_type="model", private=False, exist_ok=True)
    dpo_trainer.hub_model_id = model_hub_id

    start_time = time.time()
    dpo_trainer.train()
    end_time = time.time()

    model.save_pretrained("playpen_lora") #TODO: needed to save_pretrained?
    dpo_trainer.push_to_hub()

    #TODO: restore this to save training logs (train time, tokens)
    training_time = (end_time - start_time)/3600
    with open('training_metrics.txt', 'a') as f:
        f.write(f"{trained_model_id},{training_tokens},{training_time}\n")

    new_entry = {
        "model_name": trained_model_id,
        "base_model": model_name,
        "backend": "huggingface_local",
        "requires_api_key": True,
        "huggingface_id": model_hub_id,
        "premade_chat_template": True,
        "eos_to_cull": "<\\|eot_id\\|>",            #TODO: change here
        "open_weight": True,
        "parameters": "8B",
        "load_with_unsloth": True
    }
    json_file_path = "/mnt/cimec-storage6/users/davide.mazzaccara/clembench/backends/model_registry.json"

    with open(json_file_path, "r+") as file:
        data = json.load(file)
        data.append(new_entry)
        file.seek(0)
        json.dump(data, file, indent=4)