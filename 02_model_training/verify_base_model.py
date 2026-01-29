from transformers import AutoModelForCausalLM, AutoTokenizer
import os

import os

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "base_model")

print(f"Loading model from {model_path}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    print("Successfully loaded model and tokenizer!")
    
    # Print some model info
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(model.config)

except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)
