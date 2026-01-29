import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths (relative to script location in scripts/)
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"

def load_schema():
    schema_path = CONFIG_DIR / "hsds_schema.json"
    with open(schema_path, "r") as f:
        return json.load(f)

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

def run_inference(model_path, prompt):
    print(f"Loading model from {model_path}...")
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load schema and prepare system prompt
    try:
        schema = load_schema()
        system_content = system_content_template.format(schema_json=json.dumps(schema, indent=2))
    except Exception as e:
        print(f"Error loading schema or formatting prompt: {e}")
        # Fallback if schema fails (though ideally we want to fail hard, but for now allow fallback or return)
        print("Using fallback system prompt due to error.")
        system_content = "You are a helpful assistant that parses schedule data."

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(device)

    print("Generating response...")
    print(f"--- PROMPT ---\n{text}\n----------------")
    generated_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.2, # Prevent repetition loops (common in small models)
        pad_token_id=tokenizer.eos_token_id,
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("-" * 40)
    print("Response:")
    print(response)
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--prompt", type=str, default="Here is a schedule text: Monday-Friday 9am-5pm", help="Prompt to test")
    
    args = parser.parse_args()
    run_inference(args.model_path, args.prompt)
