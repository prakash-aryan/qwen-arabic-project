import os
from datasets import load_dataset

def download_and_save_dataset(name, repo, config=None):
    if os.path.exists(f'{name}_dataset'):
        print(f"{name} dataset already exists. Skipping download.")
        return

    print(f"Downloading {name} dataset...")
    try:
        if config:
            dataset = load_dataset(repo, config)
        else:
            dataset = load_dataset(repo)
        print(f"Saving {name} dataset...")
        dataset.save_to_disk(f'{name}_dataset')
        print(f"{name} dataset saved successfully.")
    except Exception as e:
        print(f"Error downloading {name} dataset: {str(e)}")

# List of datasets to download
datasets = [
    ("bactrian", 'M-A-D/Mixed-Arabic-Datasets-Repo', "Ara--MBZUAI--Bactrian-X"),
    ("wikipedia", 'M-A-D/Mixed-Arabic-Datasets-Repo', "Ara--Wikipedia"),
    ("oasst", 'M-A-D/Mixed-Arabic-Datasets-Repo', "Ara--OpenAssistant--oasst1"),
]

# Download and save each dataset
for dataset_info in datasets:
    name, repo, config = dataset_info
    download_and_save_dataset(name, repo, config)

print("All datasets processed.")
