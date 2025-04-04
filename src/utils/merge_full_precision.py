from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model in full precision
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Load the LoRA adapter
peft_model_id = "clembench-playpen/llama-3.1-8B-Instruct-v1.6-only-full-precision-lora_playpen_SFT-e3_DABL01_1.4K-steps"
model = PeftModel.from_pretrained(base_model, peft_model_id)

# Merge the LoRA adapter weights into the base model
merged_model = model.merge_and_unload()

# Push the merged model to the Hub
merged_model.push_to_hub("clembench-playpen/llama-3.1-8B-Instruct-v1.6-only-full-prcision-lora_playpen_SFT_merged_fp16")
tokenizer.push_to_hub("clembench-playpen/llama-3.1-8B-Instruct-v1.6-only-full-prcision-lora_playpen_SFT_merged_fp16")