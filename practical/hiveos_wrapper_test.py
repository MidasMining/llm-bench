#!/usr/bin/env python3
"""
HiveOS Wrapper Creation Test - Practical (8 criteria)
=====================================================
Tests model's ability to create multi-file bash scripts following conventions.

Evaluation criteria:
1. Uses heredocs for file creation
2. h-manifest.conf with CUSTOM_NAME
3. Uses CUSTOM_* variables (not MINER_*)
4. Creates h-config.sh
5. Creates h-run.sh with exec
6. Creates h-stats.sh with JSON output
7. Correct directory structure
8. API port configured

Run: python hiveos_wrapper_test.py [--api-url URL] [--model MODEL]
"""

import argparse
import json
import re
import requests

PROMPT = '''PRACTICAL FILE CREATION TEST - HIVEOS CUSTOM MINER WRAPPER

TASK:
Create a HiveOS custom miner wrapper for "ethminer" with the following requirements:

1. Create directory structure at ~/Downloads/ethminer/
2. Create h-manifest.conf with:
   - CUSTOM_NAME="ethminer"
   - CUSTOM_VERSION and other required fields
   
3. Create h-config.sh that:
   - Reads from CUSTOM_URL, CUSTOM_TEMPLATE, CUSTOM_PASS
   - Does NOT use MINER_* variables (those are for built-in miners)
   - Exports configuration for h-run.sh
   
4. Create h-run.sh that:
   - Uses `exec` to replace shell (NOT screen or nohup)
   - Starts ethminer with configured pool/wallet
   - Handles CUDA device selection via CUDA_VISIBLE_DEVICES
   
5. Create h-stats.sh that:
   - Outputs JSON with: hs (array of hashrates), temp, fan, uptime, ver
   - Gets stats from ethminer API on port 3333
   - Example output: {"hs":[25.5,26.1],"temp":[65,68],"fan":[70,75]}

Use bash heredocs (cat > file << 'EOF') to create the files.
Show all commands needed to create a working wrapper.

IMPORTANT CONVENTIONS:
- Custom miners use CUSTOM_* variables, not MINER_*
- h-run.sh must use exec, never screen
- h-stats.sh must output valid JSON
- API port is typically 3333 for ethminer
'''

def strip_thinking(content: str) -> str:
    """Remove model thinking tags."""
    content = re.sub(r'<seed:think>.*?</seed:think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<seed:[^>]+>', '', content)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    return content.strip()

def evaluate(response: str) -> dict:
    """Evaluate response for HiveOS wrapper requirements."""
    response_lower = response.lower()
    response_orig = response  # Keep original case for some checks
    
    checks = {
        "uses_heredocs": False,
        "h_manifest_custom_name": False,
        "uses_custom_vars": False,
        "creates_h_config": False,
        "creates_h_run_exec": False,
        "creates_h_stats_json": False,
        "directory_structure": False,
        "api_port_configured": False,
    }
    
    # Check 1: Uses heredocs
    if "<<" in response_orig and ("EOF" in response_orig or "eof" in response_lower):
        checks["uses_heredocs"] = True
    if re.search(r"cat\s*>\s*\S+\s*<<", response_orig):
        checks["uses_heredocs"] = True
    
    # Check 2: h-manifest with CUSTOM_NAME
    if "h-manifest" in response_lower and "custom_name" in response_lower:
        checks["h_manifest_custom_name"] = True
    
    # Check 3: Uses CUSTOM_* variables
    custom_vars = ["CUSTOM_URL", "CUSTOM_TEMPLATE", "CUSTOM_PASS", "CUSTOM_USER_CONFIG"]
    if any(var in response_orig for var in custom_vars):
        checks["uses_custom_vars"] = True
    
    # Check 4: Creates h-config.sh
    if "h-config.sh" in response_lower or "h-config" in response_lower:
        checks["creates_h_config"] = True
    
    # Check 5: Creates h-run.sh with exec
    if ("h-run.sh" in response_lower or "h-run" in response_lower) and "exec" in response_lower:
        checks["creates_h_run_exec"] = True
    
    # Check 6: Creates h-stats.sh with JSON
    if "h-stats.sh" in response_lower or "h-stats" in response_lower:
        if '{"hs"' in response_orig or '"hs"' in response_orig or "json" in response_lower:
            checks["creates_h_stats_json"] = True
    
    # Check 7: Correct directory structure
    if "~/downloads/ethminer" in response_lower or "mkdir" in response_lower:
        checks["directory_structure"] = True
    if re.search(r"ethminer/?", response_lower):
        checks["directory_structure"] = True
    
    # Check 8: API port configured
    if "3333" in response_orig or "api" in response_lower:
        checks["api_port_configured"] = True
    
    score = sum(1 for v in checks.values() if v)
    return {
        "score": score,
        "max_score": 8,
        "passed": score >= 6,  # 6/8 is passing
        "checks": checks,
    }

def main():
    parser = argparse.ArgumentParser(description="HiveOS Wrapper Test")
    parser.add_argument('--api-url', default='http://localhost:8000/v1')
    parser.add_argument('--model', default='seed-oss')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    
    print("=" * 60)
    print("HIVEOS WRAPPER CREATION TEST (Practical - 8 criteria)")
    print("=" * 60)
    
    try:
        r = requests.post(
            f"{args.api_url}/chat/completions",
            json={
                'model': args.model,
                'messages': [{'role': 'user', 'content': PROMPT}],
                'max_tokens': 8000,
                'temperature': 0.0
            },
            timeout=args.timeout
        )
        data = r.json()
        content = data['choices'][0]['message']['content']
        clean = strip_thinking(content)
        
        print("\n=== RESPONSE ===\n")
        print(clean[:6000] + "..." if len(clean) > 6000 else clean)
        
        print("\n=== EVALUATION ===")
        result = evaluate(clean)
        
        for check, passed in result['checks'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        print(f"\nScore: {result['score']}/{result['max_score']}")
        if result['score'] >= 7:
            print("🎉 EXCELLENT - Production-ready wrapper")
        elif result['score'] >= 5:
            print("✓ GOOD - Functional wrapper with minor issues")
        elif result['score'] >= 3:
            print("△ PARTIAL - Missing key components")
        else:
            print("✗ FAIL - Does not meet requirements")
            
        print(f"\nTokens: {data.get('usage', {})}")
        
        return result
        
    except Exception as e:
        print(f"ERROR: {e}")
        return {"score": 0, "max_score": 8, "passed": False}

if __name__ == "__main__":
    main()
