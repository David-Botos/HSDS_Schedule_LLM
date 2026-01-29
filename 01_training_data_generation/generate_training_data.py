import asyncio
import argparse
import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add project root to path to import baml_client
# This assumes the script is located at HSDS_Schedule_LLM/model_training/generate_training_data.py
# and baml_client is at HSDS_Schedule_LLM/baml_client
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    from baml_client.async_client import b
    # Ensure types are imported if needed for type hinting, though strictly optional for runtime if not used in isinstance
    # from baml_client.types import HSDSSchedule 
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"Error: Could not import baml_client: {e}", file=sys.stderr)
    print(f"sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1)

OUTPUT_FILE = "02_model_training/training_data.jsonl"
CONCURRENCY_LIMIT = 8

async def parse_schedule(sem: asyncio.Semaphore, text: str) -> dict | None:
    if not text or not text.strip():
        return None
    
    async with sem:
        try:
            # ParseSchedule is the function defined in schedule_parser.baml
            # Now returns a list of HSDSSchedule objects
            response_list = await b.ParseSchedule(text)
            
            # Ensure it's a list (handle potential single object return if BAML behaves unexpectedly, though strict typing should prefer list)
            if not isinstance(response_list, list):
                response_list = [response_list]

            parsed_data_list = []
            for response in response_list:
                # Convert BAML response object to dict
                # handling both Pydantic v1/v2 style model_dump/dict if applicable, or generic object
                item_data = None
                if hasattr(response, 'model_dump'):
                    item_data = response.model_dump()
                elif hasattr(response, 'dict'):
                    item_data = response.dict()
                else:
                    # Fallback for simple types or if it's already a dict
                    item_data = dict(response)
                
                # Clean up None values if desired, or keep them to match BAML output structure
                parsed_data_list.append(item_data)

            return {
                "original_text": text,
                "parsed_schedule": parsed_data_list
            }
        except Exception as e:
            # print(f"Error parsing '{text[:50]}...': {e}", file=sys.stderr)
            # Return None to skip failed items
            return None

async def main():
    parser = argparse.ArgumentParser(description="Generate training data from schedule CSVs")
    parser.add_argument("--limit", type=int, help="Limit the number of records to process for testing")
    args = parser.parse_args()

    root = project_root # Use the resolved project root
    
    # Determine output path early
    output_file_name = OUTPUT_FILE
    if args.limit:
        # Avoid overwriting the main training file when testing
        path_parts = list(Path(OUTPUT_FILE).parts)
        path_parts[-1] = f"test_{path_parts[-1]}"
        output_file_name = "/".join(path_parts)
        
    output_path = root / output_file_name
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    # Load existing progress for resumability
    processed_texts = set()
    if output_path.exists():
        print(f"Checking for existing progress in {output_path}...")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'original_text' in data:
                                processed_texts.add(data['original_text'])
                        except json.JSONDecodeError:
                            continue
            print(f"Found {len(processed_texts)} already processed records. These will be skipped.")
        except Exception as e:
            print(f"Error reading existing file: {e}", file=sys.stderr)

    # Dynamic CSV loading
    source_data_dir = root / "01_training_data_generation" / "source_data"
    input_files = list(source_data_dir.glob("*.csv"))
    print(f"Found {len(input_files)} CSV files in {source_data_dir}")

    print("Reading CSV files...")
    unique_texts = set()
    # Read all CSVs
    for file_path in input_files:
        if not file_path.exists():
             # Should not happen with glob but good for safety if something raced
            print(f"Warning: File not found {file_path}", file=sys.stderr)
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or 'HOURS_OF_OPERATION' not in reader.fieldnames:
                    # Only warn if verbose, otherwise can be noisy with many files
                    # print(f"Warning: 'HOURS_OF_OPERATION' column not found in {file_path.name}", file=sys.stderr)
                    continue
                
                for row in reader:
                    text = row.get('HOURS_OF_OPERATION')
                    if text and text.strip():
                        unique_texts.add(text.strip())
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)

    print(f"Found {len(unique_texts)} unique schedule strings in inputs.")

    # Filter out already processed items
    unique_texts_list = [t for t in unique_texts if t not in processed_texts]
    if len(processed_texts) > 0:
        print(f"Skipping {len(processed_texts)} items, {len(unique_texts_list)} remaining to process.")

    # Apply limit
    if args.limit:
        print(f"Limiting to first {args.limit} records for testing.")
        unique_texts_list = unique_texts_list[:args.limit]

    if not unique_texts_list:
        print("No new records to process.")
        return

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [parse_schedule(sem, text) for text in unique_texts_list]
    
    print(f"Starting processing with concurrency limit {CONCURRENCY_LIMIT}...")
    
    total = len(tasks)
    completed = 0
    successful_count = 0
    
    # Open file in append mode to support incremental writing
    try:
        with open(output_path, 'a', encoding='utf-8') as f_out:
            batch_results = []
            
            for future in asyncio.as_completed(tasks):
                res = await future
                completed += 1
                
                if completed % 10 == 0 or completed == total:
                    print(f"Processed {completed}/{total} ({(completed/total)*100:.1f}%)", end='\r')
                
                if res:
                    batch_results.append(res)
                    successful_count += 1
                
                # Write batch every 100 results
                if len(batch_results) >= 100:
                    for entry in batch_results:
                        f_out.write(json.dumps(entry) + "\n")
                    f_out.flush()
                    batch_results = []
            
            # Write key remaining results
            if batch_results:
                for entry in batch_results:
                    f_out.write(json.dumps(entry) + "\n")
                f_out.flush()
                
        print(f"\nCompleted. Successful this run: {successful_count}")
        print(f"Total records in {output_path}: {len(processed_texts) + successful_count}")
        
    except Exception as e:
        print(f"\nError during processing/saving: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
