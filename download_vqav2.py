"""
Script to download and prepare VQAv2 dataset including COCO images.
"""

import os
import json
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import gdown
import config


def download_file(url, destination, description="Downloading"):
    """Download file with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save file
        description: Description for progress bar
    """
    print(f"{description}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
    
    print(f"Downloaded to {destination}")


def extract_zip(zip_path, extract_to):
    """Extract zip file.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def download_vqav2_annotations(data_dir):
    """Download VQAv2 annotations.
    
    Args:
        data_dir: Directory to save data
    """
    print("\n" + "="*80)
    print("Downloading VQAv2 Annotations")
    print("="*80)
    
    base_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa"
    
    files_to_download = {
        "train": [
            ("v2_Questions_Train_mscoco.zip", f"{base_url}/v2_Questions_Train_mscoco.zip"),
            ("v2_Annotations_Train_mscoco.zip", f"{base_url}/v2_Annotations_Train_mscoco.zip"),
        ],
        "val": [
            ("v2_Questions_Val_mscoco.zip", f"{base_url}/v2_Questions_Val_mscoco.zip"),
            ("v2_Annotations_Val_mscoco.zip", f"{base_url}/v2_Annotations_Val_mscoco.zip"),
        ]
    }
    
    for split in config.VQAV2_CONFIG["download_splits"]:
        print(f"\nDownloading {split} split...")
        split_dir = os.path.join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for filename, url in files_to_download[split]:
            dest_path = os.path.join(split_dir, filename)
            
            if os.path.exists(dest_path):
                print(f"File already exists: {dest_path}")
                continue
            
            download_file(url, dest_path, f"Downloading {filename}")
            
            # Extract
            extract_zip(dest_path, split_dir)
            
            # Remove zip file to save space
            os.remove(dest_path)
            print(f"Removed {dest_path}")


def download_coco_images(coco_dir):
    """Download COCO images.
    
    Args:
        coco_dir: Directory to save COCO images
    """
    print("\n" + "="*80)
    print("Downloading COCO Images")
    print("="*80)
    
    os.makedirs(coco_dir, exist_ok=True)
    
    coco_files = {
        "train2014": "http://images.cocodataset.org/zips/train2014.zip",
        "val2014": "http://images.cocodataset.org/zips/val2014.zip",
    }
    
    for split_name, url in coco_files.items():
        print(f"\nDownloading {split_name}...")
        
        split_dir = os.path.join(coco_dir, split_name)
        if os.path.exists(split_dir) and len(os.listdir(split_dir)) > 0:
            print(f"{split_name} already exists with {len(os.listdir(split_dir))} images")
            continue
        
        zip_path = os.path.join(coco_dir, f"{split_name}.zip")
        
        if not os.path.exists(zip_path):
            download_file(url, zip_path, f"Downloading {split_name}")
        
        # Extract
        extract_zip(zip_path, coco_dir)
        
        # Remove zip file to save space
        os.remove(zip_path)
        print(f"Removed {zip_path}")


def create_dataset_index(data_dir, coco_dir):
    """Create an index mapping questions to image paths.
    
    Args:
        data_dir: VQAv2 data directory
        coco_dir: COCO images directory
    """
    print("\n" + "="*80)
    print("Creating Dataset Index")
    print("="*80)
    
    for split in config.VQAV2_CONFIG["download_splits"]:
        print(f"\nProcessing {split} split...")
        
        # Map split name to COCO split
        coco_split = "train2014" if split == "train" else "val2014"
        
        # Load questions
        questions_file = os.path.join(
            data_dir, split, 
            f"v2_OpenEnded_mscoco_{split}2014_questions.json"
        )
        
        if not os.path.exists(questions_file):
            print(f"Questions file not found: {questions_file}")
            continue
        
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
        
        # Load annotations
        annotations_file = os.path.join(
            data_dir, split,
            f"v2_mscoco_{split}2014_annotations.json"
        )
        
        annotations_dict = {}
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                annotations_data = json.load(f)
                for ann in annotations_data['annotations']:
                    annotations_dict[ann['question_id']] = ann
        
        # Create index
        index = []
        for question in questions_data['questions']:
            image_id = question['image_id']
            image_filename = f"COCO_{coco_split}_{image_id:012d}.jpg"
            image_path = os.path.join(coco_dir, coco_split, image_filename)
            
            entry = {
                "question_id": question['question_id'],
                "question": question['question'],
                "image_id": image_id,
                "image_path": image_path,
            }
            
            # Add answer if available
            if question['question_id'] in annotations_dict:
                ann = annotations_dict[question['question_id']]
                entry['answers'] = [a['answer'] for a in ann['answers']]
                entry['answer_type'] = ann['answer_type']
                entry['question_type'] = ann['question_type']
            
            index.append(entry)
        
        # Save index
        index_file = os.path.join(data_dir, f"{split}_index.json")
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"Created index with {len(index)} entries: {index_file}")
        
        # Print sample
        if len(index) > 0:
            print("\nSample entry:")
            sample = index[0]
            print(f"  Question ID: {sample['question_id']}")
            print(f"  Question: {sample['question']}")
            print(f"  Image Path: {sample['image_path']}")
            if 'answers' in sample:
                print(f"  Answers: {sample['answers'][:3]}")


def verify_dataset(data_dir, coco_dir):
    """Verify dataset is properly downloaded.
    
    Args:
        data_dir: VQAv2 data directory
        coco_dir: COCO images directory
    """
    print("\n" + "="*80)
    print("Verifying Dataset")
    print("="*80)
    
    issues = []
    
    for split in config.VQAV2_CONFIG["download_splits"]:
        print(f"\nVerifying {split} split...")
        
        # Check index file
        index_file = os.path.join(data_dir, f"{split}_index.json")
        if not os.path.exists(index_file):
            issues.append(f"Missing index file: {index_file}")
            continue
        
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        print(f"  Total questions: {len(index)}")
        
        # Check a few images exist
        sample_size = min(5, len(index))
        missing_images = 0
        
        for entry in index[:sample_size]:
            if not os.path.exists(entry['image_path']):
                missing_images += 1
        
        if missing_images > 0:
            issues.append(f"{split}: {missing_images}/{sample_size} sample images missing")
        else:
            print(f"  Sample images verified: {sample_size}/{sample_size} found")
    
    print("\n" + "="*80)
    if len(issues) == 0:
        print("✅ Dataset verification passed!")
    else:
        print("❌ Dataset verification found issues:")
        for issue in issues:
            print(f"  - {issue}")
    print("="*80)


def check_dataset_exists(data_dir, coco_dir):
    """Check if dataset already exists and is complete.
    
    Args:
        data_dir: VQAv2 data directory
        coco_dir: COCO images directory
        
    Returns:
        bool: True if dataset exists and appears complete
    """
    # Check if index files exist
    for split in config.VQAV2_CONFIG["download_splits"]:
        index_file = os.path.join(data_dir, f"{split}_index.json")
        if not os.path.exists(index_file):
            return False
    
    # Check if COCO directories exist and have images (efficient for large dirs)
    for split_name in ["train2014", "val2014"]:
        split_dir = os.path.join(coco_dir, split_name)
        if not os.path.exists(split_dir):
            return False
        # Efficient check for non-empty directory
        try:
            next(os.scandir(split_dir))
        except StopIteration:
            return False
    
    return True


def main():
    """Main function to download and prepare VQAv2 dataset."""
    print("="*80)
    print("VQAv2 Dataset Download and Preparation")
    print("="*80)
    
    # Get directories from config
    data_dir = config.VQAV2_CONFIG["data_dir"]
    coco_dir = config.VQAV2_CONFIG["coco_dir"]
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(coco_dir, exist_ok=True)
    
    print(f"\nData directory: {data_dir}")
    print(f"COCO directory: {coco_dir}")
    print(f"Splits to download: {config.VQAV2_CONFIG['download_splits']}")
    
    # Check if dataset already exists
    dataset_exists = check_dataset_exists(data_dir, coco_dir)
    if dataset_exists:
        print("\n✅ Dataset already exists and appears complete!")
        print("Skipping download. If you want to re-download, delete the data directories.")
        # Still verify to show stats
        verify_dataset(data_dir, coco_dir)
        
        print("\n" + "="*80)
        print("Dataset preparation complete!")
        print("="*80)
        print("\nTo use the dataset:")
        print("  import json")
        print(f"  with open('{data_dir}/train_index.json') as f:")
        print("      data = json.load(f)")
        print("  # Access questions and image paths")
        print()
        return
    
    # Download VQAv2 annotations
    download_vqav2_annotations(data_dir)
    
    # Download COCO images
    download_coco_images(coco_dir)
    
    # Create dataset index
    create_dataset_index(data_dir, coco_dir)
    
    # Verify dataset
    verify_dataset(data_dir, coco_dir)
    
    print("\n" + "="*80)
    print("Dataset preparation complete!")
    print("="*80)
    print("\nTo use the dataset:")
    print("  import json")
    print(f"  with open('{data_dir}/train_index.json') as f:")
    print("      data = json.load(f)")
    print("  # Access questions and image paths")
    print()


if __name__ == "__main__":
    main()
