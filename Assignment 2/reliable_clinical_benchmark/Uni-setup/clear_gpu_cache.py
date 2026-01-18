"""
Clear GPU cache, PyTorch cache, Hugging Face cache, and Python cache on C drive.
Run this before regenerating Study C entries that had OOM errors.
"""
import sys
import os
import shutil
import glob
from pathlib import Path

def get_cache_size(path: Path) -> float:
    """Get total size of directory in GB."""
    if not path.exists():
        return 0.0
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total += os.path.getsize(filepath)
            except (OSError, FileNotFoundError):
                pass
    return total / (1024**3)

def clear_directory(path: Path, name: str) -> None:
    """Clear a cache directory."""
    if not path.exists():
        print(f"{name} cache not found at {path}")
        return
    
    size_before = get_cache_size(path)
    print(f"\n{name} cache: {size_before:.2f} GB at {path}")
    
    try:
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {name} cache cleared successfully ({size_before:.2f} GB freed)")
    except Exception as e:
        print(f"✗ Failed to clear {name} cache: {e}")

def is_conda_env(path: Path) -> bool:
    """Check if a path is inside a conda/virtual environment."""
    path_str = str(path).lower()
    return any(marker in path_str for marker in [
        r'\.conda\envs',
        r'\.virtualenvs',
        r'\.venv',
        r'\venv\',
        r'\env\',
        r'\anaconda',
        r'\miniconda',
        r'\site-packages',
    ])

def clear_python_cache(root_dir: Path = None) -> None:
    """Clear Python cache files (__pycache__, .pyc, etc.) from a directory tree.
    Excludes conda/virtual environments to avoid breaking them.
    """
    if root_dir is None:
        # Default to user home or current working directory
        root_dir = Path.cwd()
    
    print(f"\nScanning for Python cache files in: {root_dir}")
    print("(Excluding conda/virtual environments)")
    
    patterns_to_remove = [
        ("__pycache__", "directories"),
        ("*.pyc", "files"),
        ("*.pyo", "files"),
        ("*.pyd", "files"),
        (".pytest_cache", "directories"),
        (".mypy_cache", "directories"),
        (".coverage", "files"),
        ("*.egg-info", "directories"),
        ("build", "directories"),
        ("dist", "directories"),
        (".tox", "directories"),
        (".ipynb_checkpoints", "directories"),
        (".DS_Store", "files"),
        ("Thumbs.db", "files"),
        ("*.log", "files"),
        (".ruff_cache", "directories"),
        (".hypothesis", "directories"),
    ]
    
    total_freed = 0.0
    items_removed = 0
    skipped_env = 0
    
    for pattern, item_type in patterns_to_remove:
        if item_type == "directories":
            # Find all matching directories
            for item in root_dir.rglob(pattern):
                if item.is_dir():
                    # Skip if inside conda/virtual environment
                    if is_conda_env(item):
                        skipped_env += 1
                        continue
                    
                    size = get_cache_size(item)
                    try:
                        shutil.rmtree(item)
                        total_freed += size
                        items_removed += 1
                        print(f"  ✓ Removed {item} ({size:.2f} GB)")
                    except Exception as e:
                        print(f"  ✗ Failed to remove {item}: {e}")
        else:
            # Find all matching files
            for item in root_dir.rglob(pattern):
                if item.is_file():
                    # Skip if inside conda/virtual environment
                    if is_conda_env(item):
                        skipped_env += 1
                        continue
                    
                    try:
                        size = item.stat().st_size / (1024**3)
                        item.unlink()
                        total_freed += size
                        items_removed += 1
                        if size > 0.01:  # Only print if > 10 MB
                            print(f"  ✓ Removed {item} ({size:.2f} GB)")
                    except Exception as e:
                        print(f"  ✗ Failed to remove {item}: {e}")
    
    if skipped_env > 0:
        print(f"\n  (Skipped {skipped_env} items in conda/virtual environments)")
    print(f"\n✓ Python cache cleanup complete: {items_removed} items removed, {total_freed:.2f} GB freed")

# Clear GPU cache
try:
    import torch
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"Memory before: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved")
        
        # Clear PyTorch's CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats(0)
        
        print(f"Memory after: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved")
        print("✓ GPU cache cleared successfully.")
    else:
        print("CUDA not available.")
except ImportError:
    print("PyTorch not installed. Skipping GPU cache clear.")

# Clear PyTorch cache on C drive
user_home = Path.home()
pytorch_cache = user_home / ".cache" / "torch"
clear_directory(pytorch_cache, "PyTorch")

# Clear Hugging Face cache on C drive
hf_cache = user_home / ".cache" / "huggingface"
clear_directory(hf_cache, "Hugging Face")

# Also check for transformers cache
transformers_cache = user_home / ".cache" / "transformers"
if transformers_cache.exists():
    clear_directory(transformers_cache, "Transformers")

# Clear Python cache in common locations
print("\n" + "="*60)
print("Clearing Python cache files...")
print("="*60)

# Clear Python cache in user home
clear_python_cache(user_home)

# Clear Python cache in current project directory
project_root = Path(__file__).parent.parent.parent
if project_root.exists():
    clear_python_cache(project_root)

# Clear temp directories
print("\n" + "="*60)
print("Clearing temporary files...")
print("="*60)

temp_dirs = [
    (Path(os.environ.get("TEMP", "C:\\Temp")), "Windows Temp"),
    (Path(os.environ.get("TMP", "C:\\Temp")), "Windows TMP"),
    (user_home / "AppData" / "Local" / "Temp", "User Temp"),
]

for temp_dir, name in temp_dirs:
    if temp_dir.exists():
        # Only clear Python-related temp files to avoid breaking other apps
        python_temp_patterns = ["**/__pycache__", "**/*.pyc", "**/*.pyo", "**/*.pyd"]
        total_freed = 0.0
        items_removed = 0
        
        for pattern in python_temp_patterns:
            for item in temp_dir.rglob(pattern):
                # Skip if inside conda/virtual environment
                if is_conda_env(item):
                    continue
                
                try:
                    if item.is_dir():
                        size = get_cache_size(item)
                        shutil.rmtree(item)
                        total_freed += size
                        items_removed += 1
                    elif item.is_file():
                        size = item.stat().st_size / (1024**3)
                        item.unlink()
                        total_freed += size
                        items_removed += 1
                except Exception:
                    pass
        
        if items_removed > 0:
            print(f"✓ {name}: {items_removed} Python temp items removed ({total_freed:.2f} GB)")

print("\n" + "="*60)
print("Cache clearing complete!")
print("="*60)
print("\nNote: If LM Studio is running and holding GPU memory, you may need to:")
print("  1. Unload models in LM Studio")
print("  2. Or restart LM Studio")
print("  3. Or kill the LM Studio process: taskkill /F /IM LMStudio.exe")
print("\nIf you get import errors after running this script, reinstall the broken package:")
print("  conda activate mh-llm-local-env")
print("  pip install --force-reinstall --no-cache-dir regex")
print("  # Or reinstall all packages:")
print("  pip install --force-reinstall --no-cache-dir -r requirements.txt")