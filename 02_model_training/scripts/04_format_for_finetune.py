import json
import os
import argparse
from pathlib import Path

# Paths (relative to script location in scripts/)
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"

def load_schema():
    schema_path = CONFIG_DIR / "hsds_schema.json"
    with open(schema_path, "r") as f:
        return json.load(f)

def format_for_finetune(input_path, output_path, schema):
    """
    Reads validated JSONL data and converts it to ChatML format for fine-tuning.
    """
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    system_content_template = """You are a helpful assistant that converts unstructured schedule text into structured JSON arrays based on RFC 5545 RRULES.

The output must conform to this schema:
{schema_json}

Rules:
- Times must be HH:MM:SS format (e.g., 09:00:00 not 9am)
- Days must be MO,TU,WE,TH,FR,SA,SU
- freq must be DAILY, WEEKLY, MONTHLY, or YEARLY
- Create separate objects in parsed_schedule array for different time periods
- Include original_text in output
"""
    
    # Pre-render system content since schema is constant
    system_content = system_content_template.format(schema_json=json.dumps(schema, indent=2))

    print(f"Processing {input_path} -> {output_path}")
    
    count = 0
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                original_text = data.get("original_text", "")
                parsed_schedule = data.get("parsed_schedule", [])
                
                # Construct the assistant response (Ground Truth)
                # We want the model to output the full JSON object including original_text
                assistant_response = {
                    "original_text": original_text,
                    "parsed_schedule": parsed_schedule
                }
                
                # Create ChatML structure
                message_object = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_content
                        },
                        {
                            "role": "user",
                            "content": original_text
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(assistant_response, ensure_ascii=False)
                        }
                    ]
                }
                
                outfile.write(json.dumps(message_object) + "\n")
                count += 1
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {input_path}")
                continue
                
    print(f"Formatted {count} examples.")

def get_latest_split_files(base_splits_dir):
    """
    Finds the most recent timestamped split files in the splits directory.
    Returns a dictionary mapping split name to file path.
    """
    if not base_splits_dir.exists():
        return None
        
    # Pattern: {split}_{timestamp}.jsonl
    # We look for train_*.jsonl files and extract the timestamp
    train_files = list(base_splits_dir.glob("train_*.jsonl"))
    
    if not train_files:
        # Fallback: check for non-timestamped files
        if (base_splits_dir / "train.jsonl").exists():
             return {
                 "train": base_splits_dir / "train.jsonl",
                 "val": base_splits_dir / "val.jsonl",
                 "test": base_splits_dir / "test.jsonl"
             }
        return None
        
    # Extract timestamps and sort
    # Filename format example: train_01.28.08:49:29.jsonl
    # We can just sort by filename to get the latest if the format is consistent
    # timestamps are roughly MM.DD.HH:MM:SS, which sorts correctly within a year
    # But to be safer, we can sort by modification time or just try to be consistent.
    # Given the format '01.28...', string sort works for date/time if month comes first.
    
    latest_train = sorted(train_files, key=lambda x: x.name)[-1]
    timestamp_suffix = latest_train.name.replace("train_", "")
    
    print(f"Detected latest timestamp suffix: {timestamp_suffix}")
    
    return {
        "train": latest_train,
        "val": base_splits_dir / f"val_{timestamp_suffix}",
        "test": base_splits_dir / f"test_{timestamp_suffix}"
    }

def main():
    parser = argparse.ArgumentParser(description="Format data for fine-tuning")
    parser.add_argument("--splits-dir", default=str(DATA_DIR / "processed" / "splits"), help="Base directory containing timestamped split files")
    parser.add_argument("--output-dir", default=str(DATA_DIR / "processed" / "finetune_formatted"), help="Output directory for formatted data")
    
    args = parser.parse_args()
    
    schema = load_schema()
    base_splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    
    split_files = get_latest_split_files(base_splits_dir)
    
    if not split_files:
        print(f"Error: No split files found in {base_splits_dir}")
        return
        
    print(f"Using split files: {split_files['train'].name} (and matching val/test)")
    
    # Extract timestamp from train filename to use in output
    # name format: train_{timestamp}.jsonl
    # We want: {split}_from_{timestamp}.jsonl
    timestamp_suffix = split_files['train'].name.replace("train_", "")
    
    for split in ["train", "val", "test"]:
        input_file = split_files.get(split)
        output_file = output_dir / f"{split}_from_{timestamp_suffix}"
        
        if input_file and input_file.exists():
            format_for_finetune(input_file, output_file, schema)
        else:
            print(f"Warning: {split} file not found at {input_file}, skipping.")

if __name__ == "__main__":
    main()
