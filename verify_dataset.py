from pathlib import Path
import json
import sys
import os
from PIL import Image
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def resize_image(image_path, target_size=(224, 224), output_dir=None):
    """Resize image to target size while maintaining aspect ratio with padding."""
    try:
        # Create output directory if it doesn't exist
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate aspect ratio
            aspect = img.size[0] / img.size[1]
            
            # Create new image with padding
            new_img = Image.new('RGB', target_size, (255, 255, 255))  # white background
            
            if aspect > 1:
                # Width is the limiting factor
                new_width = target_size[0]
                new_height = int(new_width / aspect)
                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                y_offset = (target_size[1] - new_height) // 2
                new_img.paste(resized, (0, y_offset))
            else:
                # Height is the limiting factor
                new_height = target_size[1]
                new_width = int(new_height * aspect)
                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                x_offset = (target_size[0] - new_width) // 2
                new_img.paste(resized, (x_offset, 0))
            
            if output_dir:
                # Preserve directory structure
                rel_path = Path(image_path).relative_to(Path("datasets/deepfashion/images"))
                output_path = Path(output_dir) / rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                new_img.save(output_path, 'JPEG', quality=95)
                return (True, str(output_path))
            return (True, new_img)
            
    except Exception as e:
        return (False, f"Error resizing {image_path}: {e}")

def verify_image_quality(image_path):
    """Verify image dimensions and quality."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            format = img.format
            
            issues = []
            needs_resize = False
            
            if mode not in ['RGB', 'RGBA']:
                issues.append(f"Invalid mode: {mode} (expected RGB/RGBA)")
            
            if width < 224 or height < 224:
                issues.append(f"Small dimensions: {width}x{height} (min 224x224)")
                needs_resize = True
            
            try:
                img.verify()
            except Exception:
                issues.append("Corrupted file")
            
            return not bool(issues), {
                'size': (width, height),
                'mode': mode,
                'format': format,
                'issues': issues,
                'needs_resize': needs_resize
            }
            
    except Exception as e:
        return False, {'issues': [str(e)], 'needs_resize': False}

def process_images(images_to_resize, output_dir):
    """Process images that need resizing using multiple threads."""
    total = len(images_to_resize)
    print(f"\nResizing {total} images to 224x224...")
    
    success_count = 0
    failed_images = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for img_path in images_to_resize:
            futures.append(executor.submit(resize_image, img_path, (224, 224), output_dir))
        
        for future in tqdm(futures, total=total, desc="Resizing images"):
            success, result = future.result()
            if success:
                success_count += 1
            else:
                failed_images.append(result)
    
    return success_count, failed_images

def check_images_recursive(resize_small=False):
    images_dir = Path("datasets/deepfashion/images")
    output_dir = Path("datasets/deepfashion/images_resized") if resize_small else None
    
    # Extended list of valid image formats
    valid_extensions = (
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', 
        '.tiff', '.tif', '.webp', '.ico', '.jfif',
        '.ppm', '.pgm', '.pbm', '.pnm', '.heic',
        '.heif', '.avif'
    )
    
    stats = {
        'total_images': 0,
        'valid_images': 0,
        'invalid_images': 0,
        'categories': defaultdict(int),
        'issues': defaultdict(int),
        'formats': defaultdict(int),
        'modes': defaultdict(int),
        'extensions': defaultdict(int),
        'sizes': defaultdict(int),
        'corrupt_files': []
    }
    
    print("\nScanning for all image formats...")
    print("Valid extensions:", ", ".join(valid_extensions))
    
    # Directory analysis with detailed stats
    unique_files = set()
    extension_counts = defaultdict(int)
    corrupt_files = []
    large_files = []
    small_files = []
    
    for root, _, files in os.walk(images_dir):
        for file in files:
            file_lower = file.lower()
            if file_lower.endswith(valid_extensions):
                file_path = Path(root) / file
                rel_path = file_path.relative_to(images_dir)
                
                # Count by extension
                ext = Path(file_lower).suffix
                extension_counts[ext] += 1
                unique_files.add(str(file_path))
                
                # Check file size
                file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
                
                try:
                    with Image.open(file_path) as img:
                        # Image format and mode
                        stats['formats'][img.format] += 1
                        stats['modes'][img.mode] += 1
                        
                        # Image dimensions
                        width, height = img.size
                        size_category = f"{width}x{height}"
                        stats['sizes'][size_category] += 1
                        
                        # Check for small images
                        if width < 224 or height < 224:
                            small_files.append((str(rel_path), (width, height)))
                        
                        # Check for very large images
                        if width > 4096 or height > 4096:
                            large_files.append((str(rel_path), (width, height)))
                        
                        # Verify image data
                        try:
                            img.verify()
                        except Exception as e:
                            corrupt_files.append((str(rel_path), str(e)))
                            
                except Exception as e:
                    corrupt_files.append((str(rel_path), str(e)))
                    stats['corrupt_files'].append((str(rel_path), str(e)))
    
    # Print detailed analysis
    print("\nDetailed Image Analysis:")
    print("======================")
    print(f"Total unique image files: {len(unique_files)}")
    
    print("\nFile Extension Distribution:")
    for ext, count in sorted(extension_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {ext}: {count:,} files")
    
    print("\nImage Format Distribution:")
    for format, count in sorted(stats['formats'].items(), key=lambda x: x[1], reverse=True):
        print(f"- {format}: {count:,} images")
    
    print("\nColor Mode Distribution:")
    for mode, count in sorted(stats['modes'].items(), key=lambda x: x[1], reverse=True):
        print(f"- {mode}: {count:,} images")
    
    if corrupt_files:
        print("\nCorrupt Files Found:")
        print(f"Total corrupt files: {len(corrupt_files)}")
        print("Sample of corrupt files (first 5):")
        for path, error in corrupt_files[:5]:
            print(f"- {path}: {error}")
    
    if small_files:
        print("\nSmall Images (< 224x224):")
        print(f"Total small images: {len(small_files)}")
        print("Sample of small images (first 5):")
        for path, size in small_files[:5]:
            print(f"- {path}: {size[0]}x{size[1]}")
    
    if large_files:
        print("\nVery Large Images (> 4096x4096):")
        print(f"Total large images: {len(large_files)}")
        print("Sample of large images (first 5):")
        for path, size in large_files[:5]:
            print(f"- {path}: {size[0]}x{size[1]}")
    
    # Return data for potential resizing
    return unique_files, stats, small_files

def verify_dataset_structure():
    """Verify the DeepFashion dataset structure and contents."""
    dataset_root = Path("datasets/deepfashion")
    required_structure = {
        "root": dataset_root,
        "images": dataset_root / "images",
        "annotations": dataset_root / "annotations.json"
    }
    
    print("\nVerifying Dataset Structure:")
    print("===========================")
    
    # Check root directory
    if not required_structure["root"].exists():
        print(f"❌ Root directory missing: {required_structure['root']}")
        return False
    
    print(f"✓ Found root directory: {required_structure['root']}")
    
    # Check images directory
    if not required_structure["images"].exists():
        print(f"❌ Images directory missing: {required_structure['images']}")
        return False
    
    # Verify images
    image_files, stats, small_files = check_images_recursive()  # Updated to handle three return values
    
    if not image_files:
        print("\n❌ No valid images found in the dataset!")
        return False
    
    # Check annotations file
    if not required_structure["annotations"].exists():
        print(f"❌ Annotations file missing: {required_structure['annotations']}")
        return False
    
    try:
        with open(required_structure["annotations"]) as f:
            annotations = json.load(f)
            ann_count = len(annotations.get('annotations', []))
            print(f"\n✓ Found annotations file with {ann_count} entries")
            
            # Verify annotation coverage
            annotated_files = {ann['file_name'] for ann in annotations['annotations']}
            missing_annotations = [img for img in image_files if img not in annotated_files]
            
            if missing_annotations:
                print(f"\n⚠️  Found {len(missing_annotations)} images without annotations")
                print("Sample missing annotations:")
                for path in missing_annotations[:5]:
                    print(f"- {path}")
    except json.JSONDecodeError:
        print("❌ Annotations file contains invalid JSON")
        return False
    except Exception as e:
        print(f"❌ Error reading annotations file: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Verify and process DeepFashion dataset images')
    parser.add_argument('--resize', action='store_true', help='Resize small images to 224x224')
    args = parser.parse_args()
    
    if not verify_dataset_structure():
        print("\n❌ Dataset verification failed!")
        sys.exit(1)
    
    check_images_recursive(resize_small=args.resize)
