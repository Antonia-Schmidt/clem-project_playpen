from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model in full precision
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

model_to_merge = "llama-3.1-70B-Instruct_playpen_SFT_DFINAL_0.6K-steps"

# Load the LoRA adapter
peft_model_id = f"clembench-playpen/{model_to_merge}"
model = PeftModel.from_pretrained(base_model, peft_model_id)

# Merge the LoRA adapter weights into the base model
merged_model = model.merge_and_unload()

# Push the merged model to the Hub
merged_model.push_to_hub(f"clembench-playpen/{model_to_merge}_merged_full_precision")
tokenizer.push_to_hub(f"clembench-playpen/{model_to_merge}_merged_full_precision")