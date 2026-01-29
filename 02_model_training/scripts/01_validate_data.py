import json
import jsonschema
from pathlib import Path
import sys
from datetime import datetime

def validate_data():
    # Define paths relative to this script
    current_dir = Path(__file__).parent.resolve()
    base_dir = current_dir.parent
    schema_path = base_dir / 'config' / 'hsds_schema.json'
    data_dir = base_dir / 'data' / 'raw'
    validated_dir = base_dir / 'data' / 'processed' / 'validated'

    # Ensure output directory exists
    validated_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%m.%d.%H:%M:%S")
    valid_output_path = validated_dir / f"valid_{timestamp}.jsonl"
    invalid_output_path = validated_dir / f"invalid_{timestamp}.jsonl"

    print(f"Looking for schema at: {schema_path}")
    if not schema_path.exists():
        print(f"Error: Schema file not found at {schema_path}")
        sys.exit(1)

    print(f"Looking for data in: {data_dir}")
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        sys.exit(1)

    # Load schema
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        print("Schema loaded successfully.")
    except Exception as e:
        print(f"Error loading schema: {e}")
        sys.exit(1)

    # Validate data files
    jsonl_files = list(data_dir.glob('*.jsonl'))
    if not jsonl_files:
        print("No .jsonl files found in data directory.")
        return

    total_valid = 0
    total_invalid = 0
    total_errors = 0

    print(f"Writing valid records to: {valid_output_path}")
    print(f"Writing invalid records to: {invalid_output_path}")

    with open(valid_output_path, 'w') as f_valid, open(invalid_output_path, 'w') as f_invalid:
        for file_path in jsonl_files:
            print(f"\nValidating {file_path.name}...")
            file_valid = 0
            file_invalid = 0
            
            try:
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line_content = line.strip()
                        if not line_content:
                            continue
                        
                        try:
                            record = json.loads(line_content)
                            jsonschema.validate(instance=record, schema=schema)
                            
                            # Write to valid file
                            f_valid.write(line_content + '\n')
                            file_valid += 1
                        except json.JSONDecodeError as e:
                            print(f"  Line {line_num}: Invalid JSON - {e}")
                            # Write to invalid file with error info
                            error_record = {
                                "original_line": line_content,
                                "error_type": "json_decode_error",
                                "error_message": str(e)
                            }
                            f_invalid.write(json.dumps(error_record) + '\n')
                            file_invalid += 1
                        except jsonschema.exceptions.ValidationError as e:
                            print(f"  Line {line_num}: Validation Error - {e.message}")
                            # Write to invalid file with error info
                            error_record = {
                                "record": record,
                                "error_type": "schema_validation_error",
                                "error_message": e.message
                            }
                            f_invalid.write(json.dumps(error_record) + '\n')
                            file_invalid += 1
                        except Exception as e:
                            print(f"  Line {line_num}: User Error - {e}")
                            f_invalid.write(json.dumps({"line": line_content, "error": str(e)}) + '\n')
                            file_invalid += 1

            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                total_errors += 1
                continue

            print(f"  Result: {file_valid} valid, {file_invalid} invalid")
            total_valid += file_valid
            total_invalid += file_invalid

    print("\n" + "="*30)
    print(f"Total Valid Records:   {total_valid}")
    print(f"Total Invalid Records: {total_invalid}")
    print(f"Total File Errors:     {total_errors}")
    print("="*30)
    
    if total_invalid > 0 or total_errors > 0:
        # We exit with 0 to allow the pipeline to continue if desired, 
        # or we could exit with 1. Given we separated the data, success might be preferable.
        # But if there are file errors (IO), that's bad.
        if total_errors > 0:
            sys.exit(1)
        # Invalid records are just part of the process, so we exit 0
        sys.exit(0)
    else:
        print("Validation successful!")
        sys.exit(0)

if __name__ == "__main__":
    validate_data()
