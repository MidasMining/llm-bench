"""
Quantize Magistral-Small-2509 to AWQ 4-bit
The model is Mistral3ForConditionalGeneration (multimodal) but we extract
the text model as MistralForCausalLM for AWQ quantization.

Usage:
    python quantize_magistral2509_awq.py

Output: /home/llm/models/Magistral-Small-2509-AWQ/
"""
import torch
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from awq import AutoAWQForCausalLM

MODEL_ID = "mistralai/Magistral-Small-2509"
OUTPUT_DIR = "/home/llm/models/Magistral-Small-2509-AWQ"
EXTRACT_DIR = "/home/llm/models/Magistral-Small-2509-text"

# Step 1: Extract the text model from the multimodal wrapper
print("Step 1: Extracting text model from multimodal wrapper...")
if not os.path.exists(os.path.join(EXTRACT_DIR, "config.json")):
    from transformers import AutoModel

    print(f"  Loading full multimodal model: {MODEL_ID}")
    full_model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Extract the text/language model
    text_model = full_model.language_model
    print(f"  Text model type: {type(text_model).__name__}")

    # Get text config and convert to standalone MistralForCausalLM config
    text_config = full_model.config.text_config
    text_config.architectures = ["MistralForCausalLM"]
    text_config.model_type = "mistral"

    print(f"  Saving text model to {EXTRACT_DIR}...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    text_model.save_pretrained(EXTRACT_DIR)
    text_config.save_pretrained(EXTRACT_DIR)

    # Copy tokenizer from original
    print("  Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(EXTRACT_DIR)

    del full_model, text_model
    torch.cuda.empty_cache()
    print("  Text model extracted successfully")
else:
    print(f"  Text model already exists at {EXTRACT_DIR}")

# Step 2: AWQ quantize the text model
print("\nStep 2: AWQ Quantization...")
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}
print(f"  Config: {quant_config}")

print(f"  Loading text model from {EXTRACT_DIR}...")
model = AutoAWQForCausalLM.from_pretrained(EXTRACT_DIR, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(EXTRACT_DIR, trust_remote_code=True)

print("  Quantizing (this will take a while)...")
model.quantize(tokenizer, quant_config=quant_config)

print(f"\nStep 3: Saving to {OUTPUT_DIR}...")
model.save_quantized(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nDone! Test with:")
print(f"  vllm serve {OUTPUT_DIR} --tensor-parallel-size 8 --gpu-memory-utilization 0.90")
