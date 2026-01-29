import json
import random
import collections
from pathlib import Path
import sys
from datetime import datetime

def get_complexity_tier(score):
    if score < 2:
        return "tier_1_simple"
    elif score <= 4:
        return "tier_2_medium"
    else:
        return "tier_3_complex"

def calculate_complexity(parsed_schedule):
    score = 0
    # +1 per object
    score += len(parsed_schedule)
    
    core_fields = {'opens_at', 'closes_at', 'freq', 'byday'}
    
    for item in parsed_schedule:
        for k, v in item.items():
            if v is not None:
                # Complexity: Non-core fields
                if k not in core_fields:
                    score += 0.5
                    
                # Complexity: Numbered day pattern
                if k == 'byday' and isinstance(v, str) and any(char.isdigit() for char in v):
                    score += 1
                    
                # Complexity: Date ranges
                if k in ['dtstart', 'until']:
                    score += 0.5
    return score

def split_items(items):
    """
    Splits items into train/val/test with handling for small dataset sizes.
    Target: 70% Train, 15% Val, 15% Test.
    """
    n = len(items)
    if n == 0: 
        return [], [], []
    
    # Priority for small N: Train -> Val -> Test
    if n == 1:
        return items, [], []
    if n == 2:
        return items[:1], items[1:], [] # 1 Train, 1 Val
    if n == 3:
        return items[:1], items[1:2], items[2:] # 1 Train, 1 Val, 1 Test
        
    # For n >= 4, use ratios but ensure minimums
    n_val = int(n * 0.15)
    if n_val == 0: n_val = 1
    
    n_test = int(n * 0.15)
    if n_test == 0: n_test = 1
    
    n_train = n - n_val - n_test
    
    return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]

def create_splits():
    # Define paths
    current_dir = Path(__file__).parent.resolve()
    base_dir = current_dir.parent
    validated_dir = base_dir / 'data' / 'processed' / 'validated'
    splits_dir = base_dir / 'data' / 'processed' / 'splits'
    analysis_dir = base_dir / 'data' / 'analysis'

    # Ensure output directories exist
    splits_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find Input
    print(f"Looking for validated data in: {validated_dir}")
    if not validated_dir.exists():
        print("Validated directory not found.")
        sys.exit(1)

    valid_files = list(validated_dir.glob('valid_*.jsonl'))
    if not valid_files:
        print("No valid_*.jsonl files found in validated directory.")
        sys.exit(1)
        
    # Get latest file
    input_file = max(valid_files, key=lambda f: f.stat().st_mtime)
    print(f"Using latest validated file: {input_file.name}")

    # 2. Load and Stratify
    print("Loading and stratifying data...")
    stratified_groups = collections.defaultdict(list)
    total_records = 0
    
    try:
        with open(input_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    record = json.loads(line)
                    parsed_schedule = record.get('parsed_schedule', [])
                    
                    # Calculate stratification key
                    length = len(parsed_schedule)
                    score = calculate_complexity(parsed_schedule)
                    tier = get_complexity_tier(score)
                    
                    key = f"{length}_{tier}"
                    stratified_groups[key].append(record)
                    total_records += 1
                    
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    print(f"Total records loaded: {total_records}")
    print(f"Stratification groups found: {len(stratified_groups)}")

    # 3. Split
    train_data = []
    val_data = []
    test_data = []
    
    random.seed(42) # Ensure reproducibility

    print("Splitting data...")
    for key, items in stratified_groups.items():
        random.shuffle(items)
        
        group_train, group_val, group_test = split_items(items)
        
        train_data.extend(group_train)
        val_data.extend(group_val)
        test_data.extend(group_test)

    print(f"Train size: {len(train_data)}")
    print(f"Val size:   {len(val_data)}")
    print(f"Test size:  {len(test_data)}")

    # 4. Save
    timestamp = datetime.now().strftime("%m.%d.%H:%M:%S")
    
    def save_jsonl(data, path):
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    train_path = splits_dir / f"train_{timestamp}.jsonl"
    val_path = splits_dir / f"val_{timestamp}.jsonl"
    test_path = splits_dir / f"test_{timestamp}.jsonl"
    report_path = analysis_dir / f"split_distribution_{timestamp}.json"

    print(f"Saving train set to: {train_path.name}")
    save_jsonl(train_data, train_path)
    
    print(f"Saving val set to:   {val_path.name}")
    save_jsonl(val_data, val_path)
    
    print(f"Saving test set to:  {test_path.name}")
    save_jsonl(test_data, test_path)

    # Save distribution report
    report = {
        "timestamp": timestamp,
        "input_file": input_file.name,
        "total_records": total_records,
        "splits": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        },
        "stratification_groups": {k: len(v) for k, v in stratified_groups.items()}
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Split report saved to: {report_path.name}")

if __name__ == "__main__":
    create_splits()
