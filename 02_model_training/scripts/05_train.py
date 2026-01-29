
import os
import yaml
import pathlib
import modal
from datetime import datetime

# ==============================================================================
# 1. Modal App & Image Configuration
# ==============================================================================

app = modal.App("hsds-schedule-finetune")

# Define the image with necessary dependencies for Unsloth & LoRA
# Using a recent CUDA/Torch version compatible with L40S/A100
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("cmake", "git", "curl", "libcurl4-openssl-dev", "build-essential", "automake", "autoconf")
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.18.1",
        "transformers==4.57.6",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2026.1.4",
        "unsloth_zoo==2026.1.4",
        "protobuf<3.21",
        "python-dotenv",
        "pyyaml",
        "torchao==0.14.0",
        "fbgemm-gpu-genai==1.2.0",
        "gguf",
        "sentencepiece",
        "mistral_common"
    )
    .env({"HF_HOME": "/model_cache", "UV_SYSTEM_PYTHON": "1"})
    .add_local_dir("02_model_training/config", remote_path="/config")
    .add_local_dir("02_model_training/data/processed/finetune_formatted", remote_path="/dataset")
)

# ==============================================================================
# 2. Volume Configuration
# ==============================================================================

# Volume to cache base model weights (avoids re-downloading every run)
model_cache_volume = modal.Volume.from_name("hsds-model-cache", create_if_missing=True)

# Volume to store training checkpoints and final adapters
checkpoint_volume = modal.Volume.from_name("hsds-checkpoints", create_if_missing=True)

# ==============================================================================
# 3. Training Logic (Remote)
# ==============================================================================

@app.function(
    image=train_image,
    gpu="L40S",  # Powerful enough for 0.6B model, fast
    timeout=3600 * 16,  # 16 hours max
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume
    },
    secrets=[modal.Secret.from_name("unsloth_env")] # Secrets make environment variables available.
)
def train_remote(config_path: str, experiment_name_override: str = None, dry_run: bool = False):
    """
    Executes the training loop on the remote GPU.
    """
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset
    import glob
    import os
    
    # --- 1. Load Configuration ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Override paths for REMOTE environment
    # The config file has local paths (e.g. data/processed/...), we need to map them to /dataset
    # We assume filename consistency
    
    # Detect filenames from valid paths
    # We dynamically find the latest files in /dataset to ensure we use the freshest data
    def get_latest_file(pattern, directory="/dataset"):
        search_path = os.path.join(directory, pattern)
        files = glob.glob(search_path)
        if not files:
            raise FileNotFoundError(f"No files found matching '{pattern}' in '{directory}'")
        return sorted(files)[-1]

    train_path = get_latest_file("train_from_*.jsonl")
    val_path = get_latest_file("val_from_*.jsonl")
    
    print(f"Loading data from: {train_path}")
    
    # Determines Experiment Name
    if experiment_name_override:
        exp_name = experiment_name_override
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"exp_{timestamp}"
    
    output_dir = f"/checkpoints/{exp_name}"
    
    # --- 2. Load Model & Tokenizer ---
    # We use the HF ID if available, or fall back to local mount if we were mounting it.
    # Implementation Decision: Use the HF ID "osmosis-ai/Osmosis-Structure-0.6B"
    # If the user has a local-only model, they should upload it to the Volume first.
    # For now, we assume it's pullable or the user edits this line.
    
    model_id = "osmosis-ai/Osmosis-Structure-0.6B" 
    # If the config points to a local directory, we might need a workaround. 
    # Let's trust the HF pulling for now, or check if /config/base_model exists (if mounted).
    
    max_seq_length = config["data"].get("max_seq_length", 2048)
    dtype = None # Auto detection
    load_in_4bit = False # User requested fp16, so 4bit loading is off
    
    print(f"Loading model: {model_id}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length,
        dtype = torch.float16, # Enforce fp16 as requested
        load_in_4bit = load_in_4bit,
    )
    
    # --- 3. Configure LoRA & QAT ---
    print("Configuring LoRA with QAT (Int4)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = config["training"]["lora"]["r"],
        target_modules = config["training"]["lora"]["target_modules"],
        lora_alpha = config["training"]["lora"]["lora_alpha"],
        lora_dropout = config["training"]["lora"]["lora_dropout"],
        bias = config["training"]["lora"]["bias"],
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
        qat_scheme = "int4", # Enable QAT!
    )
    
    # --- 4. Load Data ---
    print("Loading datasets...")
    # Load JSONL files
    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})
    
    # Unsloth/TRL expects a 'text' field or chat template. 
    # Our data is in ChatML format: {"messages": [...]}
    # We need to format this for the trainer.
    
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "chatml",
        mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"}
    )
    
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }
        
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    
    # --- 5. Training Arguments ---
    hyperparams = config["training"]["hyperparameters"]
    
    # Prioritize dry_run, then config, then default to -1 (num_epochs determines duration)
    if dry_run:
        max_steps = 5
    else:
        max_steps = hyperparams.get("max_steps", -1)

    num_train_epochs = 1 if dry_run else hyperparams["num_epochs"]

    trainer_args = TrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = hyperparams["batch_size"],
        gradient_accumulation_steps = hyperparams["gradient_accumulation_steps"],
        warmup_steps = hyperparams["warmup_steps"],
        max_steps = max_steps,
        num_train_epochs = num_train_epochs,
        learning_rate = float(hyperparams["learning_rate"]),
        fp16 = True,
        bf16 = False,
        logging_steps = 10,
        eval_strategy = "steps",
        eval_steps = 50 if dry_run else 100,
        save_strategy = "steps",
        save_steps = 50 if dry_run else 500,
        optim = "adamw_8bit",
        weight_decay = hyperparams["weight_decay"],
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use wandb if needed, "none" for simple stdout
    )
    
    # --- 6. Initialize Trainer ---
    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["validation"],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, 
        args = trainer_args,
    )
    
    # --- 7. Train ---
    print(f"Starting training (Dry run: {dry_run})...")
    trainer_stats = trainer.train()
    
    # --- 8. Save Adapter ---
    print(f"Saving adapters to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # --- 9. Merge & Export Stages ---
    
    # 9.1 Merge to 16-bit SafeTensors
    # This combines LoRA adapters with base model into a deployable artifact
    merged_16bit_dir = f"{output_dir}/merged_16bit"
    print(f"Saving merged 16-bit model to {merged_16bit_dir}...")
    model.save_pretrained_merged(merged_16bit_dir, tokenizer, save_method = "merged_16bit")
    
    # 9.2 Convert to GGUF (Quantized)
    # This creates a logical GGUF file for local inference (e.g. Ollama)
    # We save it to a separate directory to keep things organized
    # Helper to move GGUF files from CWD to target
    import shutil
    import glob
    def move_ggufs(target_dir, filter_str=""):
        print(f"=== DEBUG: Searching for GGUF files in CWD to move to {target_dir} ===")
        cwd_ggufs = glob.glob("*.gguf")
        print(f"GGUF files in CWD ({os.getcwd()}): {cwd_ggufs}")
        
        found = False
        for gfile in cwd_ggufs:
            if filter_str and filter_str not in gfile.lower():
                continue
                
            target = f"{target_dir}/{gfile}"
            print(f"Moving {gfile} to {target}...")
            try:
                shutil.move(gfile, target)
                found = True
            except Exception as e:
                print(f"Failed to move {gfile}: {e}")
        
        if not found:
            print(f"WARNING: No GGUF files found to move to {target_dir}")
        print("=== DEBUG END ===")

    # 9.2 Convert to GGUF (Query)
    # This creates a logical GGUF file for local inference (e.g. Ollama)
    # We save it to a separate directory to keep things organized
    gguf_dir = f"{output_dir}/gguf"
    print(f"Saving GGUF (f16) to {gguf_dir}...")
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method = "f16")
    
    # IMMEDIATELY Move f16 file to safety
    move_ggufs(gguf_dir, "f16")
    
    
    # 9.3 Convert to GGUF (QAT Int4) - QAT Optimized
    # Since we trained with qat_scheme="int4", this export should leverage that training
    gguf_qat_dir = f"{output_dir}/gguf_qat_int4"
    print(f"Saving GGUF (QAT q4_k_m) to {gguf_qat_dir}...")
    model.save_pretrained_gguf(gguf_qat_dir, tokenizer, quantization_method = "q4_k_m")

    # IMMEDIATELY Move qat file to safety
    move_ggufs(gguf_qat_dir, "q4_k_m")

    # Sync wait to ensure volume writes persist before download
    import time
    print("Waiting 10s for volume sync...")
    time.sleep(10)
    
    return output_dir

# ==============================================================================
# 4. Local Entrypoint
# ==============================================================================

@app.local_entrypoint()
def main(dry_run: bool = False):
    print("🚀 Starting Modal Training Job")
    
    # Local paths (relative to project root where command is run)
    config_file = "02_model_training/config/training_config.yaml"
    
    # Check if config exists
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        return

    # Trigger remote training
    # This mounts the local folder content to the remote container automatically based on @app.function mounts
    # The remote path is fixed at /config/training_config.yaml due to .add_local_dir mapping
    remote_output_dir = train_remote.remote("/config/training_config.yaml", dry_run=dry_run)
    
    print(f"✅ Remote training finished. Output saved at: {remote_output_dir}")
    
    # Download artifacts
    # We use modal.Volume to read the data back
    print("⬇️  Downloading artifacts to local '02_model_training/experiments/'...")
    
    local_exp_dir = pathlib.Path("02_model_training/experiments") / pathlib.Path(remote_output_dir).name
    local_exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate files in the volume location and write them locally
    
    vol_name = "hsds-checkpoints"
    # We download to 02_model_training/experiments/
    local_dest = "02_model_training/experiments/"
    
    # The remote_output_dir includes the mount prefix "/checkpoints/"
    # For 'modal volume get', we need the path relative to the volume root.
    # e.g., /checkpoints/exp_123 -> exp_123
    volume_relative_path = remote_output_dir.replace("/checkpoints/", "")
    
    print(f"   Run: modal volume get {vol_name} {volume_relative_path} {local_dest}")
    
    # We can programmatically execute the download
    import subprocess
    subprocess.run(["modal", "volume", "get", vol_name, volume_relative_path, local_dest], check=True)
    
    print(f"✨ Done! Artifacts in: {local_exp_dir}")

