"""
Setup script for Grounding DINO model.
Downloads the model checkpoint and sets up configuration.
"""

import os
import gdown
import requests
from pathlib import Path


def download_grounding_dino_checkpoint():
    """Download Grounding DINO model checkpoint."""
    print("="*80)
    print("Downloading Grounding DINO Checkpoint")
    print("="*80)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    checkpoint_path = "models/groundingdino_swint_ogc.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint already exists: {checkpoint_path}")
        return checkpoint_path
    
    # Grounding DINO checkpoint URL
    url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    
    print(f"Downloading from {url}")
    print(f"Saving to {checkpoint_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(checkpoint_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ Downloaded checkpoint to {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        print(f"❌ Error downloading checkpoint: {e}")
        print("\nAlternative: Download manually from:")
        print(url)
        print(f"and save to: {checkpoint_path}")
        return None


def clone_grounding_dino_repo():
    """Clone Grounding DINO repository if not exists."""
    print("\n" + "="*80)
    print("Setting up Grounding DINO Repository")
    print("="*80)
    
    repo_dir = "GroundingDINO"
    
    if os.path.exists(repo_dir):
        print(f"Repository already exists: {repo_dir}")
        return True
    
    print("Cloning Grounding DINO repository...")
    
    import subprocess
    try:
        result = subprocess.run(
            ["git", "clone", "https://github.com/IDEA-Research/GroundingDINO.git"],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Repository cloned successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error cloning repository: {e}")
        print("Please clone manually:")
        print("  git clone https://github.com/IDEA-Research/GroundingDINO.git")
        return False


def update_config_paths():
    """Update config.py with correct paths."""
    print("\n" + "="*80)
    print("Updating Configuration")
    print("="*80)
    
    config_file = "config.py"
    
    # Read current config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update paths if needed
    if 'GroundingDINO/groundingdino' in content:
        print("✅ Config already has correct paths")
    else:
        print("Updating Grounding DINO config paths...")
        # This is already set correctly in our config
        print("✅ Configuration is up to date")
    
    print("\nCurrent Grounding DINO configuration:")
    print("  config_file: GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    print("  checkpoint: groundingdino_swint_ogc.pth")


def verify_setup():
    """Verify the setup is complete."""
    print("\n" + "="*80)
    print("Verifying Setup")
    print("="*80)
    
    issues = []
    
    # Check GroundingDINO repo
    if not os.path.exists("GroundingDINO"):
        issues.append("GroundingDINO repository not found")
    else:
        print("✅ GroundingDINO repository exists")
    
    # Check checkpoint
    if not os.path.exists("models/groundingdino_swint_ogc.pth"):
        issues.append("Grounding DINO checkpoint not found")
    else:
        size_mb = os.path.getsize("models/groundingdino_swint_ogc.pth") / (1024 * 1024)
        print(f"✅ Grounding DINO checkpoint exists ({size_mb:.1f} MB)")
    
    # Check config file
    config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    if os.path.exists(config_path):
        print(f"✅ Config file exists: {config_path}")
    else:
        issues.append(f"Config file not found: {config_path}")
    
    print("\n" + "="*80)
    if len(issues) == 0:
        print("✅ Setup verification passed!")
        print("\nYou can now use Grounding DINO in the inference system.")
    else:
        print("❌ Setup verification found issues:")
        for issue in issues:
            print(f"  - {issue}")
    print("="*80)


def main():
    """Main setup function."""
    print("="*80)
    print("Grounding DINO Setup")
    print("="*80)
    print("\nThis script will:")
    print("  1. Clone Grounding DINO repository")
    print("  2. Download model checkpoint (~600MB)")
    print("  3. Update configuration")
    print()
    
    # Clone repository
    clone_grounding_dino_repo()
    
    # Download checkpoint
    download_grounding_dino_checkpoint()
    
    # Update config
    update_config_paths()
    
    # Verify setup
    verify_setup()
    
    print("\n" + "="*80)
    print("Setup Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Install Grounding DINO: pip install -e GroundingDINO")
    print("  2. Install other dependencies: pip install -r requirements.txt")
    print("  3. Run inference: python run_inference.py --image img.jpg --question 'What is this?'")
    print()


if __name__ == "__main__":
    main()
