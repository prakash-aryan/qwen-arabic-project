#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Create necessary directories
mkdir -p data results models tools

# Step 1: Download and prepare datasets
echo "Downloading and preparing datasets..."
python src/get_datasets.py

# Step 2: Preprocess and combine datasets
echo "Preprocessing and combining datasets..."
python src/preprocess_datasets.py

# Step 3: Validate the dataset
echo "Validating the dataset..."
python src/validate_dataset.py

# Step 4: Fine-tune the model
echo "Fine-tuning the model..."
python src/finetune_qwen.py --data_path ./data/arabic_instruction_dataset --output_dir ./models/qwen2_arabic_finetuned --num_epochs 3 --batch_size 1 --gradient_accumulation_steps 16 --learning_rate 2e-5

# Step 5: Load and merge the fine-tuned model
echo "Loading and merging the fine-tuned model..."
python src/load_and_merge_model.py

# Step 6: Convert to GGUF format
echo "Converting to GGUF format..."
python src/convert_hf_to_gguf.py ./models/qwen2_arabic_merged_full --outfile ./models/qwen_arabic_merged_full.gguf

# Step 7: Quantize the model
echo "Quantizing the model..."
./tools/llama-quantize ./models/qwen_arabic_merged_full.gguf ./models/qwen_arabic_merged_full_q4_k_m.gguf q4_k_m

# Step 8: Create Ollama model
echo "Creating Ollama model..."
ollama create qwen-arabic-custom -f Modelfile

# Step 9: Evaluate the model
echo "Evaluating the model..."
python src/evaluate_arabic_model.py

# Step 10: Compare models
echo "Comparing models..."
python src/compare_qwen_models.py

echo "Pipeline completed!"
```
