import os
from datasets import load_from_disk
import random
import json
import re
from collections import Counter
from tqdm import tqdm

def load_dataset(path):
    return load_from_disk(path)

def check_empty_fields(dataset):
    empty_instructions = sum(1 for item in dataset if not item['instruction'].strip())
    empty_outputs = sum(1 for item in dataset if not item['output'].strip())
    print(f"Empty instructions: {empty_instructions}")
    print(f"Empty outputs: {empty_outputs}")

def check_length_distribution(dataset):
    instruction_lengths = [len(item['instruction'].split()) for item in dataset]
    output_lengths = [len(item['output'].split()) for item in dataset]
    
    print("Instruction length statistics:")
    print(f"  Min: {min(instruction_lengths)}")
    print(f"  Max: {max(instruction_lengths)}")
    print(f"  Average: {sum(instruction_lengths) / len(instruction_lengths):.2f}")
    
    print("\nOutput length statistics:")
    print(f"  Min: {min(output_lengths)}")
    print(f"  Max: {max(output_lengths)}")
    print(f"  Average: {sum(output_lengths) / len(output_lengths):.2f}")

def check_language(dataset, sample_size=1000):
    def is_arabic(text):
        # Simple check for Arabic characters
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        return bool(arabic_pattern.search(text))
    
    sample = random.sample(list(dataset), min(sample_size, len(dataset)))
    non_arabic_instructions = sum(1 for item in sample if not is_arabic(item['instruction']))
    non_arabic_outputs = sum(1 for item in sample if not is_arabic(item['output']))
    
    print(f"Non-Arabic instructions in sample: {non_arabic_instructions}/{sample_size}")
    print(f"Non-Arabic outputs in sample: {non_arabic_outputs}/{sample_size}")

def check_duplicates(dataset):
    instruction_counter = Counter(item['instruction'] for item in dataset)
    output_counter = Counter(item['output'] for item in dataset)
    
    duplicate_instructions = sum(1 for count in instruction_counter.values() if count > 1)
    duplicate_outputs = sum(1 for count in output_counter.values() if count > 1)
    
    print(f"Duplicate instructions: {duplicate_instructions}")
    print(f"Duplicate outputs: {duplicate_outputs}")

def print_random_samples(dataset, num_samples=5):
    samples = random.sample(list(dataset), num_samples)
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(json.dumps(sample, ensure_ascii=False, indent=2))

def main():
    dataset_path = "./arabic_instruction_dataset"
    dataset = load_dataset(dataset_path)
    
    print(f"Dataset size: {len(dataset)}")
    
    print("\nChecking for empty fields:")
    check_empty_fields(dataset)
    
    print("\nChecking length distribution:")
    check_length_distribution(dataset)
    
    print("\nChecking language (sample):")
    check_language(dataset)
    
    print("\nChecking for duplicates:")
    check_duplicates(dataset)
    
    print("\nRandom samples:")
    print_random_samples(dataset)

if __name__ == "__main__":
    main()