import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
model_path = os.path.join(project_root, "models", "qwen2_arabic_finetuned")
output_dir_full = os.path.join(project_root, "models", "qwen2_arabic_full")
output_dir_merged = os.path.join(project_root, "models", "qwen2_arabic_merged_full")

# Load the configuration
config = PeftConfig.from_pretrained(model_path)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load the fine-tuned model
model = PeftModel.from_pretrained(base_model, model_path)

# Save the full model (not merged)
os.makedirs(output_dir_full, exist_ok=True)
model.save_pretrained(output_dir_full, safe_serialization=True)

# Save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(output_dir_full)

print(f"Full model saved to {output_dir_full}")

# Merge the adapter with the base model
merged_model = model.merge_and_unload()

# Save the full merged model
os.makedirs(output_dir_merged, exist_ok=True)
merged_model.save_pretrained(output_dir_merged, safe_serialization=True)

# Save the tokenizer for the merged model
tokenizer.save_pretrained(output_dir_merged)

# Save the configuration for the merged model
merged_model.config.save_pretrained(output_dir_merged)

print(f"Merged model saved to {output_dir_merged}")