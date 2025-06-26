"""
AetherCV WandB Setup and Configuration Helper
"""

import sys
import os
from pathlib import Path
import subprocess

def install_wandb():
    """Install WandB if not already installed"""
    print("📦 Installing WandB...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        print("✅ WandB installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install WandB: {e}")
        return False

def setup_wandb_login():
    """Guide user through WandB login"""
    print("\n🔑 Setting up WandB authentication...")
    print("Please run the following command to login to WandB:")
    print("   wandb login")
    print("\nThis will open a browser where you can:")
    print("1. Create a free WandB account (if needed)")
    print("2. Copy your API key")
    print("3. Paste it in the terminal")

def check_config_files():
    """Check and show available config files"""
    print("\n📋 Checking configuration files...")
    
    configs_dir = Path("configs")
    if not configs_dir.exists():
        print("❌ configs/ directory not found")
        return False
    
    yaml_files = list(configs_dir.glob("*.yaml"))
    if not yaml_files:
        print("❌ No YAML config files found")
        return False
    
    print(f"✅ Found {len(yaml_files)} configuration files:")
    for yaml_file in sorted(yaml_files):
        print(f"   - {yaml_file.stem}")
    
    return True

def show_customization_guide():
    """Show how to customize WandB settings"""
    print("\n🔧 Customizing WandB Settings:")
    print("=" * 40)
    
    print("\n1. Edit any config file in configs/ directory")
    print("2. Find the 'wandb:' section")
    print("3. Update these settings:")
    print("   - project_name: Your project name")
    print("   - entity: Your WandB username or team")
    print("   - tags: List of tags for organization")
    print("   - notes: Description of your experiment")
    
    print("\nExample configuration:")
    print("""
wandb:
  enabled: true
  project_name: "my-aethercv-experiments"
  entity: "my-username"
  tags: ["research", "comparison", "cifar10"]
  notes: "Comparing YAT vs Linear models"
  log_frequency: 10
  log_gradients: true
  log_parameters: true
  log_images: true
""")

def show_usage_examples():
    """Show usage examples"""
    print("\n🚀 Usage Examples:")
    print("=" * 20)
    
    examples = [
        ("Train single YAT CNN with WandB", "python train.py --config single_yat_cnn"),
        ("Compare all models with WandB", "python train.py --config full_comparison"),
        ("Train without WandB logging", "python train.py --config single_yat_cnn --no-wandb"),
        ("CNN comparison on STL10", "python train.py --config cnn_stl10"),
        ("ResNet comparison", "python train.py --config resnet_comparison"),
    ]
    
    for i, (description, command) in enumerate(examples, 1):
        print(f"\n{i}. {description}:")
        print(f"   {command}")

def main():
    """Main setup function"""
    print("🔧 AetherCV WandB Setup Helper")
    print("=" * 35)
    
    # Step 1: Install WandB
    try:
        import wandb
        print("✅ WandB is already installed")
    except ImportError:
        if not install_wandb():
            return False
    
    # Step 2: Check config files
    if not check_config_files():
        print("❌ Configuration files not found. Make sure you're in the AetherCV directory.")
        return False
    
    # Step 3: Guide through setup
    setup_wandb_login()
    
    # Step 4: Show customization guide
    show_customization_guide()
    
    # Step 5: Show usage examples
    show_usage_examples()
    
    print("\n🎉 Setup Complete!")
    print("\n📋 Quick Start:")
    print("1. Run: wandb login")
    print("2. Edit configs/*.yaml to set your project name")
    print("3. Run: python train.py --config single_yat_cnn")
    print("4. View results at: https://wandb.ai/")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
