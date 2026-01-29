import json
import argparse
from pathlib import Path
import collections
import sys
from datetime import datetime

def analyze_complexity():
    # Define paths relative to this script
    current_dir = Path(__file__).parent.resolve()
    base_dir = current_dir.parent
    analysis_dir = base_dir / 'data' / 'analysis'
    validated_dir = base_dir / 'data' / 'processed' / 'validated'
    raw_dir = base_dir / 'data' / 'raw'

    # Ensure analysis directory exists
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Determine input file
    input_files = []
    
    # Try to find latest valid file in validated_dir
    print(f"Checking for validated data in: {validated_dir}")
    if validated_dir.exists():
        valid_files = list(validated_dir.glob('valid_*.jsonl'))
        if valid_files:
            # Sort by modification time to get the latest
            latest_file = max(valid_files, key=lambda f: f.stat().st_mtime)
            print(f"Using latest validated file: {latest_file.name}")
            input_files = [latest_file]
    
    if not input_files:
        print("No validated files found. Falling back to raw data.")
        input_files = list(raw_dir.glob('*.jsonl'))
        if not input_files:
            print("No data files found to analyze.")
            sys.exit(1)

    print(f"Analyzing {len(input_files)} file(s)...")

    # Analysis counters
    total_examples = 0
    array_length_dist = collections.defaultdict(int)
    field_usage = collections.defaultdict(int)
    complexity_tiers = {"tier_1_simple": 0, "tier_2_medium": 0, "tier_3_complex": 0}
    
    # Fields config
    core_fields = {'opens_at', 'closes_at', 'freq', 'byday'}
    
    parsed_schedule_count = 0 

    for file_path in input_files:
        print(f"Reading {file_path}...")
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        parsed_schedule = data.get('parsed_schedule', [])
                        
                        if not isinstance(parsed_schedule, list):
                            continue
                            
                        total_examples += 1
                        
                        # 1. Array Length
                        length = len(parsed_schedule)
                        if length >= 4:
                            array_length_dist['4+'] += 1
                        else:
                            array_length_dist[str(length)] += 1
                            
                        # Calculate Complexity Score
                        score = 0
                        score += length  # +1 per object
                        
                        for item in parsed_schedule:
                            parsed_schedule_count += 1
                            # 2. Field Usage
                            for k, v in item.items():
                                if v is not None:
                                    field_usage[k] += 1
                                    
                                    # Complexity: Non-core fields
                                    if k not in core_fields:
                                        score += 0.5
                                        
                                    # Complexity: Numbered day pattern
                                    if k == 'byday' and isinstance(v, str) and any(char.isdigit() for char in v):
                                        score += 1
                                        
                                    # Complexity: Date ranges
                                    if k in ['dtstart', 'until']:
                                        score += 0.5
                        
                        # 3. Complexity Tiers
                        if score < 2:
                            complexity_tiers['tier_1_simple'] += 1
                        elif score <= 4:
                            complexity_tiers['tier_2_medium'] += 1
                        else:
                            complexity_tiers['tier_3_complex'] += 1
                            
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line in {file_path}")
                        continue
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    # Prepare output report
    timestamp = datetime.now().strftime("%m.%d.%H:%M:%S")
    report = {
        "timestamp": timestamp,
        "total_examples": total_examples,
        "array_length_distribution": dict(array_length_dist),
        "field_usage": {},
        "complexity_tiers": complexity_tiers
    }
    
    # Process field usage stats
    for field, count in field_usage.items():
        percentage = (count / parsed_schedule_count * 100) if parsed_schedule_count > 0 else 0
        report['field_usage'][field] = {
            "count": count,
            "percentage": round(percentage, 1)
        }

    output_path = analysis_dir / f'array_complexity_{timestamp}.json'
    try:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nAnalysis complete. Report saved to {output_path}")
        print(f"Total examples analyzed: {total_examples}")
        print(f"Total schedule objects: {parsed_schedule_count}")
        
    except Exception as e:
        print(f"Error saving report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    analyze_complexity()
