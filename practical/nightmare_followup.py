#!/usr/bin/env python3
"""
Nightmare Follow-up Test
========================
Targeted follow-up questions for bugs missed in the initial nightmare test.

This simulates a realistic workflow where a developer asks follow-up questions
about specific areas that weren't fully analyzed.

Run: python nightmare_followup.py [--api-url URL] [--model MODEL]
"""

import argparse
import json
import re
import requests

FOLLOWUP_PROMPTS = [
    {
        "name": "byte_order_hint",
        "description": "Hint about byte order in block header",
        "prompt": """Looking at the _build_header method again:

```python
def _build_header(self, job_id, nonce, ntime) -> bytes:
    version = struct.pack('<I', self.block_template['version'])
    prev_hash = bytes.fromhex(self.block_template['previousblockhash'])
    merkle_root = self._compute_merkle(job_id)
    time_bytes = struct.pack('<I', int(ntime, 16))
    bits = bytes.fromhex(self.block_template['bits'])
    nonce_bytes = struct.pack('<I', int(nonce, 16))
    
    return version + prev_hash + merkle_root + time_bytes + bits + nonce_bytes
```

The node's RPC returns `previousblockhash` in a specific format. 
Bitcoin/crypto block headers have specific byte ordering conventions.

Is there a bug related to how prev_hash is handled?""",
        "check_patterns": [
            r"reverse",
            r"\[::-1\]",
            r"byte.*order",
            r"endian",
            r"little.*endian",
            r"big.*endian",
            r"swap",
        ]
    },
    {
        "name": "info_leak_hint", 
        "description": "Hint about information leak",
        "prompt": """Looking at the _submit_block method:

```python
async def _submit_block(self, job_id, nonce, ntime):
    header = self._build_header(job_id, nonce, ntime)
    block_hex = header.hex() + self.block_template['transactions_hex']
    
    logger.info(f"BLOCK FOUND! Submitting: {block_hex[:64]}...")
    
    # Submit to node
    result = await self._rpc_call('submitblock', [block_hex])
```

The symptom was "block theft" - competitors sometimes submit our blocks first.

Is there an OPSEC issue here? Think about the order of operations and what
gets exposed before vs after submission.""",
        "check_patterns": [
            r"log.*before",
            r"before.*submit",
            r"leak",
            r"expose",
            r"broadcast.*first",
            r"order.*operation",
            r"opsec",
            r"competitor",
        ]
    }
]

def strip_thinking(content: str) -> str:
    """Remove model thinking tags."""
    content = re.sub(r'<seed:think>.*?</seed:think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<seed:[^>]+>', '', content)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    return content.strip()

def evaluate_followup(response: str, patterns: list) -> bool:
    """Check if response identifies the bug."""
    response_lower = response.lower()
    for pattern in patterns:
        if re.search(pattern, response_lower, re.IGNORECASE):
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Nightmare Follow-up Test")
    parser.add_argument('--api-url', default='http://localhost:8000/v1')
    parser.add_argument('--model', default='seed-oss')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--test', choices=['byte_order', 'info_leak', 'all'], default='all')
    args = parser.parse_args()
    
    print("=" * 60)
    print("NIGHTMARE FOLLOW-UP TEST")
    print("=" * 60)
    
    results = {}
    
    for followup in FOLLOWUP_PROMPTS:
        if args.test != 'all' and args.test not in followup['name']:
            continue
            
        print(f"\n--- {followup['description']} ---")
        
        try:
            r = requests.post(
                f"{args.api_url}/chat/completions",
                json={
                    'model': args.model,
                    'messages': [{'role': 'user', 'content': followup['prompt']}],
                    'max_tokens': 4000,
                    'temperature': 0.0
                },
                timeout=args.timeout
            )
            data = r.json()
            content = data['choices'][0]['message']['content']
            clean = strip_thinking(content)
            
            print(f"\n{clean[:1500]}...")
            
            found = evaluate_followup(clean, followup['check_patterns'])
            results[followup['name']] = found
            
            status = "✓ FOUND" if found else "✗ MISSED"
            print(f"\n{status}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            results[followup['name']] = False
    
    print("\n" + "=" * 60)
    print("FOLLOW-UP RESULTS")
    print("=" * 60)
    
    for name, found in results.items():
        status = "✓" if found else "✗"
        print(f"  {status} {name}")
    
    total = sum(1 for v in results.values() if v)
    print(f"\nTotal: {total}/{len(results)} bugs found with hints")
    
    if total == len(results):
        print("✓ Model found all bugs with follow-up hints")
    else:
        print("✗ Some bugs still missed even with hints")

if __name__ == "__main__":
    main()
