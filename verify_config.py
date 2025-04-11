import json
from pathlib import Path

def verify_configs():
    """Verify all configuration files for consistency."""
    config_dir = Path("config")
    
    # Load all configs
    try:
        with open(config_dir / "deepfashion_config.json") as f:
            deepfashion_config = json.load(f)
        with open(config_dir / "fashion_config.json") as f:
            fashion_config = json.load(f)
        with open(config_dir / "dataset_config.json") as f:
            dataset_config = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Missing config file: {e.filename}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        return False
    
    # Verify category consistency
    df_categories = set(deepfashion_config['categories'].keys())
    fashion_categories = set(fashion_config['FASHION_CATEGORIES'].keys())
    
    if df_categories != fashion_categories:
        print("❌ Category mismatch between configs:")
        print(f"DeepFashion categories: {df_categories}")
        print(f"Fashion categories: {fashion_categories}")
        return False
    
    # Verify dataset paths
    dataset_path = Path(dataset_config['dataset']['base_dir'])
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_path}")
        return False
    
    print("✓ All configurations verified successfully!")
    return True

if __name__ == "__main__":
    verify_configs()