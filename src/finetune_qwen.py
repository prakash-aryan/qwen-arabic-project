import os
import sys
import argparse
import logging
import random
import numpy as np
from typing import Dict, List, Union, Any
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup,
    TrainerState,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from torch.cuda.amp import autocast, GradScaler

import wandb
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variable for PyTorch memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Custom TrainOutput class
class TrainOutput:
    def __init__(self, global_step, training_loss, metrics=None):
        self.global_step = global_step
        self.training_loss = training_loss
        self.metrics = metrics or {}

# Set random seeds for reproducibility
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Custom dataset class
class QwenDataset(Dataset):
    def __init__(self, data: Dict[str, List], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data['instruction'])

    def __getitem__(self, idx):
        instruction = self.data['instruction'][idx]
        output = self.data['output'][idx]
        
        prompt = f"Human: {instruction}\n\nAssistant: {output}"
        encoded = self.tokenizer.encode_plus(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'labels': encoded['input_ids'].flatten()
        }

# Load and prepare dataset
def load_and_prepare_data(data_path: str, tokenizer, test_size: float = 0.1):
    logger.info(f"Loading dataset from {data_path}")
    dataset = load_from_disk(data_path)
    logger.info(f"Total samples in dataset: {len(dataset)}")
    logger.info(f"Dataset columns: {dataset.column_names}")
    logger.info(f"First few samples: {dataset[:5]}")

    dataset = dataset.shuffle(seed=42)
    train_size = int((1 - test_size) * len(dataset))
    train_data = dataset[:train_size]
    val_data = dataset[train_size:]
    
    logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    train_dataset = QwenDataset(train_data, tokenizer)
    val_dataset = QwenDataset(val_data, tokenizer)
    
    return train_dataset, val_dataset

# Load model and tokenizer
def load_model_and_tokenizer(model_name: str, quantization_config: BitsAndBytesConfig):
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        max_memory={0: "3.5GB", "cpu": "12GB"}  # Adjusted for 4GB VRAM and 12GB system RAM
    )
    
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

# Configure LoRA
def configure_lora(model, lora_config: LoraConfig):
    logger.info("Configuring LoRA")
    return get_peft_model(model, lora_config)

# Custom trainer with gradient clipping and wandb logging
class QwenTrainer(Trainer):
    def __init__(self, *args, max_grad_norm: float = 1.0, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_grad_norm = max_grad_norm
        self.scaler = GradScaler()
        self.tokenizer = tokenizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with autocast():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        return loss.detach()

    def train(self):
        train_dataloader = self.get_train_dataloader()
        
        if self.args.max_steps > 0:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
            num_train_samples = self.args.max_steps * self.args.train_batch_size * self.args.gradient_accumulation_steps
        else:
            num_train_epochs = self.args.num_train_epochs
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_train_samples = len(self.train_dataset) * self.args.num_train_epochs
        
        self.create_optimizer_and_scheduler(num_training_steps=num_train_epochs * num_update_steps_per_epoch)
        
        self.state = TrainerState()
        self.state.epoch = 0
        self.state.global_step = 0
        
        total_loss = 0.0
        logging_loss = 0.0
        start_time = time.time()
        
        for epoch in range(num_train_epochs):
            self.state.epoch = epoch
            for step, inputs in enumerate(train_dataloader):
                self.state.global_step += 1
                
                with autocast():
                    loss = self.training_step(self.model, inputs)
                
                total_loss += loss.item()
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    self.model.zero_grad()
                
                if self.state.global_step % self.args.logging_steps == 0:
                    avg_loss = (total_loss - logging_loss) / self.args.logging_steps
                    logging_loss = total_loss
                    
                    # Log GPU memory usage
                    gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                    gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # Convert to GB
                    
                    # Log RAM usage
                    ram_used = psutil.virtual_memory().used / 1024**3  # Convert to GB
                    
                    # Calculate time remaining
                    elapsed_time = time.time() - start_time
                    steps_per_second = self.state.global_step / elapsed_time
                    remaining_steps = self.args.max_steps - self.state.global_step
                    estimated_time_remaining = remaining_steps / steps_per_second
                    
                    logger.info(f"Step {self.state.global_step}: loss = {avg_loss:.4f}, GPU Memory Used: {gpu_memory_used:.2f} GB, GPU Memory Cached: {gpu_memory_cached:.2f} GB, RAM: {ram_used:.2f} GB, Est. Time Remaining: {estimated_time_remaining/3600:.2f} hours")
                    wandb.log({
                        "loss": avg_loss,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "epoch": self.state.epoch,
                        "step": self.state.global_step,
                        "gpu_memory_used": gpu_memory_used,
                        "gpu_memory_cached": gpu_memory_cached,
                        "ram_used": ram_used,
                    })
                
                if self.state.global_step % self.args.eval_steps == 0:
                    self.evaluate_and_save_samples(self.state.global_step)
                
                if self.state.global_step >= self.args.max_steps > 0:
                    break
            
            if self.state.global_step >= self.args.max_steps > 0:
                break
        
        return TrainOutput(self.state.global_step, total_loss / self.state.global_step)

    def evaluate_and_save_samples(self, step):
        logger.info(f"Evaluating and saving samples at step {step}")
        test_data = load_test_data("test_data.json")
        outputs = []
        
        for item in test_data:
            prompt = f"Human: {item['text']}\n\nAssistant:"
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            
            with torch.no_grad():
                output = self.model.generate(input_ids, max_new_tokens=100, num_return_sequences=1)
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            outputs.append({"prompt": item['text'], "response": response})
        
        os.makedirs('results', exist_ok=True)
        with open(os.path.join('results', f"sample_outputs_step_{step}.json"), "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)


def init_wandb(args):
    wandb.init(
        project="qwen2-arabic-finetuning",
        config={
            "model_name": args.model_name,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        }
    )

def load_test_data(file_path):
    with open(os.path.join('data', file_path), 'r', encoding='utf-8') as f:
        return json.load(f)

# Training function
def train(args):
    set_seeds(args.seed)
    
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, bnb_config)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Apply LoRA
    model = configure_lora(model, lora_config)
    
    # Load and prepare data
    train_dataset, val_dataset = load_and_prepare_data(args.data_path, tokenizer, args.test_size)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        save_total_limit=3,
        logging_steps=10,
        remove_unused_columns=False,
        push_to_hub=False,
        label_names=["input_ids", "attention_mask"],
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        optim="adamw_8bit",
    )
    
    # Initialize wandb
    init_wandb(args)

    # Create Trainer instance
    trainer = QwenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        max_grad_norm=args.max_grad_norm,
        tokenizer=tokenizer  # Pass the tokenizer here

    )
    
    # Start training
    logger.info("Starting training")
    train_result = trainer.train()
    
    # Save the fine-tuned model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info(f"Training completed! Steps: {train_result.global_step}, Loss: {train_result.training_loss:.4f}")
    wandb.finish()

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2 model using QLoRA")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="Model name or path")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="./qwen2_arabic_finetuned", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Max gradient norm for clipping")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max number of training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of data to use for validation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    try:
        train(args)
    except Exception as e:
        logger.exception("An error occurred during training:")
        raise