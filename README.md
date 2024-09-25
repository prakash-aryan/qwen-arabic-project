# Qwen Arabic Fine-tuning Project

This project fine-tunes the Qwen2-1.5B model for Arabic language tasks using Quantized LoRA (QLoRA).

## Prerequisites

- Ubuntu (or similar Linux distribution)
- Python 3.10
- CUDA-compatible GPU with at least 4GB VRAM
- At least 12GB system RAM
- Ollama installed and configured

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/prakash-aryan/qwen-arabic-project.git
   cd qwen-arabic-project
   ```

2. Create and activate a virtual environment:
   ```
   python3.10 -m venv qwen_env
   source qwen_env/bin/activate
   ```

3. Install the required packages:
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Install PyTorch with CUDA support:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Project Structure

```
qwen-arabic-project/
├── data/
│   └── arabic_instruction_dataset/
├── models/
├── results/
├── src/
│   ├── compare_qwen_models.py
│   ├── evaluate_arabic_model.py
│   ├── finetune_qwen.py
│   ├── get_datasets.py
│   ├── load_and_merge_model.py
│   ├── preprocess_datasets.py
│   └── validate_dataset.py
├── tools/
│   └── llama-quantize
├── requirements.txt
├── run_pipeline.sh
├── Modelfile
└── README.md
```

## Usage

1. Download and prepare datasets:
   ```
   python src/get_datasets.py
   ```

2. Preprocess and combine datasets:
   ```
   python src/preprocess_datasets.py
   ```

3. Validate the dataset:
   ```
   python src/validate_dataset.py
   ```

4. Fine-tune the model:
   ```
   python src/finetune_qwen.py --data_path ./data/arabic_instruction_dataset --output_dir ./models/qwen2_arabic_finetuned --num_epochs 3 --batch_size 1 --gradient_accumulation_steps 16 --learning_rate 2e-5
   ```

5. Load and merge the fine-tuned model:
   ```
   python src/load_and_merge_model.py
   ```

6. Convert to GGUF format:
   ```
   python src/convert_hf_to_gguf.py ./models/qwen2_arabic_merged_full --outfile ./models/qwen_arabic_merged_full.gguf
   ```

7. Quantize the model:
   ```
   ./tools/llama-quantize ./models/qwen_arabic_merged_full.gguf ./models/qwen_arabic_merged_full_q4_k_m.gguf q4_k_m
   ```

8. Create Ollama model:
   ```
   ollama create qwen-arabic-custom -f Modelfile
   ```

9. Evaluate the model:
   ```
   python src/evaluate_arabic_model.py
   ```

10. Compare models:
    ```
    python src/compare_qwen_models.py
    ```

## Running the Full Pipeline

To run the entire pipeline from data preparation to model evaluation, use the provided shell script:

```
chmod +x run_pipeline.sh
./run_pipeline.sh
```

## Notes

- Ensure you have sufficient disk space for the datasets and model files.
- The fine-tuning process can take several hours to days, depending on your hardware.
- Monitor GPU memory usage during fine-tuning and adjust batch size or gradient accumulation steps if necessary.
- Make sure to have Ollama installed for the model creation and evaluation steps.

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try reducing the batch size or increasing gradient accumulation steps.
- For any other issues, please check the error logs or open an issue in the GitHub repository.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

This means:
- You can use, modify, and distribute this software.
- If you distribute modified versions, you must also distribute them under the GPL-3.0.
- You must include the original copyright notice and the license text.
- You must disclose your source code when you distribute the software.
- There's no warranty for this free software.

For more details, see the [LICENSE](LICENSE) file in this repository or visit [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Acknowledgements

This project uses the following main libraries and tools:
- Transformers by Hugging Face
- PyTorch
- PEFT (Parameter-Efficient Fine-Tuning)
- Ollama
- GGUF (for model conversion)
