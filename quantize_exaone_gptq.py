"""
Quantize EXAONE-4.0-32B to GPTQ 4-bit with group_size=32, desc_act=False
This unlocks Marlin kernel at TP=8 (3424/32=107, exact alignment)

Usage:
    python quantize_exaone_gptq.py

Output: /home/llm/models/EXAONE-4.0-32B-GPTQ-g32/
"""
import torch
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-32B"
OUTPUT_DIR = "/home/llm/models/EXAONE-4.0-32B-GPTQ-g32"

print(f"Loading model: {MODEL_ID}")
print(f"Output: {OUTPUT_DIR}")
print(f"Config: bits=4, group_size=32, desc_act=False, sym=True")

quant_config = QuantizeConfig(
    bits=4,
    group_size=32,
    desc_act=False,
    sym=True,
    device="cuda:0",
    auto_forward_data_parallel=False,
    wait_for_submodule_finalizers=True,
)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Loading model (this will use ~64GB RAM)...")
model = GPTQModel.load(
    MODEL_ID,
    quant_config,
    torch_dtype=torch.bfloat16,
)

print("Preparing calibration data (wikitext-2, 128 samples)...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calibration = [row["text"] for row in dataset if len(row["text"].strip()) > 50][:128]
print(f"  Using {len(calibration)} calibration samples")

print("Quantizing (this will take a while with 64 layers)...")
model.quantize(
    calibration=calibration,
    tokenizer=tokenizer,
    batch_size=1,
)

print(f"Saving to {OUTPUT_DIR}...")
model.save(OUTPUT_DIR)

# Also save tokenizer for easy loading
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done! Test with:")
print(f"  vllm serve {OUTPUT_DIR} --tensor-parallel-size 8 --gpu-memory-utilization 0.90")
