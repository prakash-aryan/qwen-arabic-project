import json
import re
from datasets import load_from_disk, DatasetDict, concatenate_datasets
from typing import Dict, Any
import logging
from tqdm import tqdm
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text by removing diacritics and normalizing certain characters.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[ًٌٍَُِّْٰٖٕٓٔ]', '', text)  # Remove diacritics
    text = re.sub(r'[إأآا]', 'ا', text)  # Normalize alifs
    text = re.sub(r'ى', 'ي', text)  # Normalize ya
    text = re.sub(r'ة', 'ه', text)  # Normalize ta marbuta
    return text.strip()

def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to a maximum length while keeping whole words.
    """
    if not isinstance(text, str):
        return ""
    if len(text) <= max_length:
        return text
    return ' '.join(text[:max_length+1].split()[:-1])

def process_bactrian(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Process Bactrian dataset entries.
    """
    return {
        "instruction": normalize_arabic_text(example.get("instruction", "")),
        "input": normalize_arabic_text(example.get("input", "")),
        "output": normalize_arabic_text(example.get("output", ""))
    }

def process_oasst(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Process OpenAssistant dataset entries.
    """
    return {
        "instruction": normalize_arabic_text(example.get("question", "")),
        "input": "",
        "output": normalize_arabic_text(example.get("answer", ""))
    }

def process_wikipedia(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Process Wikipedia dataset entries.
    """
    title = normalize_arabic_text(example.get("title", ""))
    text = normalize_arabic_text(example.get("text", ""))
    
    questions = [
        f"ما هو {title}؟",
        f"اشرح بالتفصيل عن {title}.",
        f"ما هي أهم المعلومات عن {title}؟"
    ]
    
    return {
        "instruction": random.choice(questions),
        "input": "",
        "output": truncate_text(text, 1000)
    }

def inspect_dataset(name: str, dataset: Any) -> None:
    """
    Inspect and log details about a dataset.
    """
    logger.info(f"\nInspecting {name} dataset:")
    logger.info(f"Number of examples: {len(dataset)}")
    logger.info(f"Column names: {dataset.column_names}")
    logger.info("First example:")
    logger.info(json.dumps(dataset[0], indent=2, ensure_ascii=False))

def load_and_process_dataset(name: str, path: str, process_func: callable) -> Any:
    """
    Load and process a dataset.
    """
    try:
        # Load the dataset from disk
        dataset = load_from_disk(path)
        
        # If the dataset has multiple splits, use the 'train' split
        if isinstance(dataset, DatasetDict):
            dataset = dataset['train']
        
        logger.info(f"Loaded {name} dataset. Size: {len(dataset)}")
        
        # Inspect the dataset
        inspect_dataset(name, dataset)
        
        # Process the dataset
        processed = dataset.map(process_func, remove_columns=dataset.column_names)
        logger.info(f"Processed {len(processed)} entries from {path}")
        return processed
    except Exception as e:
        logger.error(f"Error processing dataset {name}: {str(e)}")
        return None

def main():
    # Define datasets
    datasets = {
        "bactrian": ("./bactrian_dataset", process_bactrian),
        "oasst": ("./oasst_dataset", process_oasst),
        "wikipedia": ("./wikipedia_dataset", process_wikipedia)
    }
    
    processed_datasets = []
    for name, (path, process_func) in datasets.items():
        dataset = load_and_process_dataset(name, path, process_func)
        if dataset:
            processed_datasets.append(dataset)
    
    # Combine datasets
    combined_dataset = concatenate_datasets(processed_datasets)
    logger.info(f"Combined dataset size: {len(combined_dataset)}")
    
    # Shuffle the combined dataset
    combined_dataset = combined_dataset.shuffle(seed=42)
    
    # Filter out empty entries
    combined_dataset = combined_dataset.filter(lambda x: x["instruction"] and x["output"])
    logger.info(f"Dataset size after filtering: {len(combined_dataset)}")
    
    # Save the combined dataset
    combined_dataset.save_to_disk("./arabic_instruction_dataset")
    logger.info("Dataset saved to ./arabic_instruction_dataset")
    
    # Print sample entries
    print("Sample entries:")
    for i in range(3):
        print(json.dumps(combined_dataset[i], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()