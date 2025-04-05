import os
from unsloth import FastLanguageModel, PatchDPOTrainer
import torch
from datasets import load_dataset
from huggingface_hub import login, create_repo
from trl import KTOTrainer, KTOConfig
import wandb
import argparse
import time
import json

#TODO:  rewrite this in a decent way
def print_info_max_len(dataset, tokenizer):
    counter_samples_exceeding_max_len = 0
    for sample in dataset['train']:
        sample = tokenizer.apply_chat_template(sample['prompt']+sample['completion'])
        if len(sample) > 1024:
            print(len(sample))
            counter_samples_exceeding_max_len += 1
    print(counter_samples_exceeding_max_len)


def load_hf_dataset(dataset_name, max_seq_length):
    split_portion = ""
    dataset = load_dataset(f"{dataset_name}", split=f"train{split_portion}")
    #dataset = dataset.filter(lambda sample: len(' '.join(map(lambda x: x['content'], sample["prompt"]))) + len(' '.join(map(lambda x: x['content'], sample["completion"]))) <= max_seq_length)
    dataset = dataset.train_test_split(test_size=0.04)
    dataset = dataset.remove_columns(['game', 'game_id', 'model', 'benchmark_version', 'experiment', 'episode', 'Aborted', 'Lose', 'Success',
         'target', 'player', 'turn_score'])
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KTO training')
    parser.add_argument('--model_condition', default=False, choices=[False, 'best_models', 'same_family_model'], help='restriction to the negative samples to be from best models or from the family of the model to train (llama)')
    #TODO: take this out in common with DPO_training.py and KTO_training.py
    parser.add_argument('--hf_login', default="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", help='hf login token')
    parser.add_argument('--wandb_login', default="29fc614754925fca7f38d6d2193b3f5afa8485a9", help='wandb login token')
    parser.add_argument('--base_model', default="llama-SFT-base_merged_fp16_D90053_copy_32GB", help='base model for training')
    parser.add_argument('--cache_dir', default = '/mnt/cimec-storage6/shared/hf_llms_checkpoints/', help='cache directory to store models')
    parser.add_argument('--dataset_name', default='KTO_Aborted_WordleOnly', help='training datasets')
    parser.add_argument('--n_epochs', default=1, type=int, help='number of training epochs')
    parser.add_argument('--hf_repo_base', default='clembench-playpen', help='huggingface repository where the base model is stored')
    parser.add_argument('--hf_repo_target', default='clembench-playpen', help='huggingface repository to save the trained model')
    args = parser.parse_args()

    login(f"{args.hf_login}")
    os.environ['WANDB_API_KEY'] = f"{args.wandb_login}"
    wandb.login()
    cache_dir_models = args.cache_dir

    max_seq_length = 1024
    dtype = None
    load_in_4bit = True
    model_name = f"{args.hf_repo_base}/{args.base_model}"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        cache_dir=cache_dir_models,
        max_seq_length=max_seq_length,      #default truncation mode is 'keep end' (thus deleting initial tokens)
        fix_tokenizer=False
    )

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

    dataset = load_hf_dataset(f"{args.hf_repo_target}/{args.dataset_name}", max_seq_length)

    PatchDPOTrainer() #TODO: best patchDPO or patchKTO?????

    project_name = f"playpen_{args.base_model}"
    entity = "wandb"

    wandb.init(project=project_name, name = f"kto_{args.dataset_name}{'_'+str(args.n_epochs)+'eps' if args.n_epochs != 1 else ''}")

    training_arguments = KTOConfig(
            per_device_train_batch_size = 4,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps = 2,
            warmup_ratio = 0.1,
            num_train_epochs = args.n_epochs,
            learning_rate = 5e-6,
            undesirable_weight = round(dataset['train']['label'].count(True) / dataset['train']['label'].count(False), 1),      #as suggested in https://huggingface.co/docs/trl/main/en/kto_trainer the proportion should 1
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
            max_length = max_seq_length,
            max_prompt_length = max_seq_length,
            #news
            eval_steps=0.2, #0.2
            save_steps=0.2, #0.2
            eval_strategy="steps",
            logging_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True
            )

    kto_trainer = KTOTrainer(
        model = model,
        args=training_arguments,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        tokenizer = tokenizer,
    )

    #TODO: this number is largely dependent on the max_len (see tokenizer.decode(batch['answer_input_ids'][0]))
    train_dataloader = kto_trainer.get_train_dataloader()
    #Modified from the original HF trainer.py num_tokens function
    def compute_tokens(train_dl: train_dataloader) -> int:
        train_tokens = 0
        for batch in train_dl:
            #TODO: check training use this information or completion_input_ids (prompt_input_ids + answer_input_ids)
            tokens = batch['answer_input_ids'].numel()
            train_tokens += tokens
        return train_tokens

    training_tokens = compute_tokens(train_dataloader)

    trained_model_id = f"{args.base_model}_{args.dataset_name}{'_'+str(args.n_epochs)+'eps' if args.n_epochs != 1 else ''}_KTO_noSFT"
    model_hub_id = f"{args.hf_repo_target}/{trained_model_id}"
    create_repo(repo_id=model_hub_id, repo_type="model", private=False, exist_ok=True)
    kto_trainer.hub_model_id = model_hub_id

    start_time = time.time()
    kto_trainer.train()
    end_time = time.time()

    model.save_pretrained("kto_trained")
    kto_trainer.push_to_hub()

    # Save training logs (train time, tokens)
    training_time = (end_time - start_time) / 3600
    with open('training_metrics.txt', 'a') as f:
        f.write(f"{trained_model_id},{training_tokens},{training_time}\n")

    # # TODO: migliora: vale solo per Llama 3.1 per ora
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
    # #TODO: solve a bug here, there is an issue: does it work this way? see if it is possible to reduce opening the file twice using json_file_path, 'a'
    json_file_path = "/mnt/cimec-storage6/users/davide.mazzaccara/clembench/backends/model_registry.json"

    with open(json_file_path, "r+") as file:
        data = json.load(file)
        data.append(new_entry)
        file.seek(0)
        json.dump(data, file, indent=4)