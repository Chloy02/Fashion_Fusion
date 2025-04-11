import json
from pathlib import Path

def check_annotations():
    with open('datasets/deepfashion/annotations.json', 'r') as f:
        data = json.load(f)
    
    print("\nChecking annotations:")
    print("====================")
    print(f"Total annotations: {len(data['annotations'])}")
    print("\nFirst 5 file paths from annotations:")
    for ann in data['annotations'][:5]:
        print(f"- {ann['file_name']}")
        # Check if file exists
        if not (Path("datasets/deepfashion/images") / ann['file_name']).exists():
            print(f"  ❌ File missing!")

if __name__ == "__main__":
    check_annotations()