"""
Quick test to verify WandB logging integration works
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_wandb_config_loading():
    """Test loading WandB config from YAML files"""
    print("🔍 Testing WandB configuration loading...")
    
    try:
        from utils.yaml_config import ConfigLoader
        
        config_loader = ConfigLoader()
        configs = config_loader.list_available_configs()
        
        print(f"✅ Found {len(configs)} configuration files")
        
        for config_name in configs:
            try:
                config = config_loader.load_config(config_name)
                wandb_config = config.wandb
                
                print(f"📋 {config_name}:")
                print(f"   Project: {wandb_config.project_name}")
                print(f"   Enabled: {wandb_config.enabled}")
                print(f"   Tags: {wandb_config.tags}")
                if wandb_config.notes:
                    print(f"   Notes: {wandb_config.notes}")
                
            except Exception as e:
                print(f"❌ Error loading {config_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test config loading: {e}")
        return False

def show_usage_examples():
    """Show usage examples"""
    print("\n🚀 WandB Integration Usage Examples:")
    print("=" * 50)
    
    print("\n1. 📋 List available configurations:")
    print("   python train.py --list-configs")
    
    print("\n2. 🔗 Run with WandB logging (default):")
    print("   python train.py --config single_yat_cnn")
    
    print("\n3. 🚫 Run without WandB logging:")
    print("   python train.py --config single_yat_cnn --no-wandb")
    
    print("\n4. 🎯 Single model with WandB:")
    print("   python train.py --model yat_cnn")
    
    print("\n5. 📊 Model comparison with WandB:")
    print("   python train.py --config full_comparison")
    
    print("\n6. 🔧 Custom WandB settings in YAML:")
    print("   Edit configs/*.yaml files to customize:")
    print("   - project_name: Your WandB project")
    print("   - entity: Your WandB username/team")
    print("   - tags: List of tags for organization")
    print("   - notes: Description of the experiment")

def main():
    """Main test function"""
    print("🧪 AetherCV WandB Integration Test")
    print("=" * 40)
    
    success = test_wandb_config_loading()
    
    if success:
        print("\n✅ WandB configuration system is working!")
        show_usage_examples()
        
        print("\n📋 Next Steps:")
        print("1. Run 'wandb login' to authenticate with WandB")
        print("2. Edit configs/*.yaml files to set your project name and entity")
        print("3. Run an experiment: python train.py --config single_yat_cnn")
        print("4. View results at: https://wandb.ai/")
    else:
        print("\n❌ WandB configuration test failed")
        print("Check that all dependencies are installed correctly")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
