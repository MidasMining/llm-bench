#!/usr/bin/env python3
"""
Practical Debugging Tests Runner v2.0
=====================================
Real-world bug-finding tests based on production mining pool code.

Tests progress from easy to nightmare difficulty:
1. zmq_test.py      - Single bug (Easy)
2. pplns_test.py    - 3 bugs (Hard)  
3. expert_test.py   - 4+ bugs including race conditions (Expert)
4. nightmare_test.py - 5+ bugs including crypto/protocol issues (Nightmare)
5. hiveos_wrapper_test.py - File creation (Practical)

Usage:
    python run_practical.py                    # Run all tests
    python run_practical.py --test zmq         # Run single test
    python run_practical.py --test pplns expert # Run specific tests
    python run_practical.py --api-url http://host:port/v1 --model model-name

These tests evaluate a model's ability to find real bugs in production code,
not synthetic benchmarks. All bugs are from actual mining pool incidents.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

VERSION = "2.0"

# Test definitions
TESTS = {
    "zmq": {
        "file": "zmq_test.py",
        "name": "ZMQ Listener Bug",
        "difficulty": "Easy",
        "bugs": 1,
        "description": "Single threading bug causing listener to never run"
    },
    "pplns": {
        "file": "pplns_test.py",
        "name": "PPLNS Mining Pool Bugs",
        "difficulty": "Hard",
        "bugs": 3,
        "description": "Config mismatch, hardcoded values, method name errors"
    },
    "expert": {
        "file": "expert_test.py",
        "name": "Payment System Bugs",
        "difficulty": "Expert",
        "bugs": 4,
        "description": "Race conditions, SQL injection, float precision, atomicity"
    },
    "nightmare": {
        "file": "nightmare_test.py", 
        "name": "Stratum Protocol Bugs",
        "difficulty": "Nightmare",
        "bugs": 5,
        "description": "Crypto byte order, info leaks, memory leaks, input validation"
    },
    "hiveos": {
        "file": "hiveos_wrapper_test.py",
        "name": "HiveOS Wrapper Creation",
        "difficulty": "Practical",
        "bugs": 8,  # 8 evaluation criteria
        "description": "Create multi-file miner wrapper following strict conventions"
    }
}

class Colors:
    HEADER = '\033[1;93m'
    PASS = '\033[92m'
    FAIL = '\033[91m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

C = Colors()

def run_test_module(test_path: str, api_url: str, model: str, timeout: int) -> dict:
    """Import and run a test module, returning its results."""
    # Import the module dynamically
    spec = importlib.util.spec_from_file_location("test_module", test_path)
    module = importlib.util.module_from_spec(spec)
    
    # Temporarily modify sys.argv to pass arguments to the test
    original_argv = sys.argv
    sys.argv = ['test', '--api-url', api_url, '--model', model, '--timeout', str(timeout)]
    
    try:
        spec.loader.exec_module(module)
        # Call main() if it exists and returns results
        if hasattr(module, 'main'):
            result = module.main()
            if result:
                return result
    except Exception as e:
        print(f"{C.FAIL}Error running test: {e}{C.RESET}")
        return {"score": 0, "max_score": 1, "passed": False, "error": str(e)}
    finally:
        sys.argv = original_argv
    
    return {"score": 0, "max_score": 1, "passed": False}

def main():
    parser = argparse.ArgumentParser(
        description=f"Practical Debugging Tests Runner v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_practical.py                      # Run all tests
  python run_practical.py --test zmq           # Run ZMQ test only
  python run_practical.py --test nightmare     # Run nightmare test
  python run_practical.py --list               # List available tests
        """
    )
    
    parser.add_argument('--api-url', default='http://localhost:8000/v1',
                       help='API endpoint URL')
    parser.add_argument('--model', default='seed-oss',
                       help='Model ID to test')
    parser.add_argument('--test', nargs='+', 
                       help='Specific tests to run (zmq, pplns, expert, nightmare, hiveos)')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout per test in seconds')
    parser.add_argument('--output', type=str,
                       help='Output JSON file for results')
    parser.add_argument('--list', action='store_true',
                       help='List available tests and exit')
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')
    
    args = parser.parse_args()
    
    # List tests
    if args.list:
        print(f"\n{C.HEADER}Available Practical Tests:{C.RESET}\n")
        for key, test in TESTS.items():
            print(f"  {C.CYAN}{key:<12}{C.RESET} {test['difficulty']:<10} ({test['bugs']} bugs) {test['description']}")
        print()
        return
    
    # Determine which tests to run
    tests_to_run = args.test if args.test else list(TESTS.keys())
    
    # Validate test names
    for test in tests_to_run:
        if test not in TESTS:
            print(f"{C.FAIL}Unknown test: {test}{C.RESET}")
            print(f"Available: {', '.join(TESTS.keys())}")
            return
    
    # Run tests
    print(f"\n{C.HEADER}{'=' * 60}{C.RESET}")
    print(f"{C.HEADER}Practical Debugging Tests Runner v{VERSION}{C.RESET}")
    print(f"{C.HEADER}{'=' * 60}{C.RESET}")
    print(f"\nModel: {args.model}")
    print(f"API: {args.api_url}")
    print(f"Tests: {', '.join(tests_to_run)}")
    
    results = {}
    total_score = 0
    total_max = 0
    start_time = time.time()
    
    script_dir = Path(__file__).parent
    
    for test_key in tests_to_run:
        test_info = TESTS[test_key]
        test_path = script_dir / test_info['file']
        
        print(f"\n{C.CYAN}{'=' * 60}{C.RESET}")
        print(f"{C.CYAN}Running: {test_info['name']} ({test_info['difficulty']}){C.RESET}")
        print(f"{C.CYAN}{'=' * 60}{C.RESET}")
        
        if not test_path.exists():
            print(f"{C.FAIL}Test file not found: {test_path}{C.RESET}")
            results[test_key] = {"error": "File not found"}
            continue
        
        result = run_test_module(str(test_path), args.api_url, args.model, args.timeout)
        results[test_key] = result
        
        if 'score' in result:
            total_score += result['score']
            total_max += result['max_score']
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{C.HEADER}{'=' * 60}{C.RESET}")
    print(f"{C.HEADER}SUMMARY{C.RESET}")
    print(f"{C.HEADER}{'=' * 60}{C.RESET}")
    
    for test_key, result in results.items():
        test_info = TESTS[test_key]
        if 'error' in result:
            status = f"{C.FAIL}ERROR{C.RESET}"
            score_str = "N/A"
        elif result.get('passed', False):
            status = f"{C.PASS}PASS{C.RESET}"
            score_str = f"{result['score']}/{result['max_score']}"
        else:
            status = f"{C.FAIL}FAIL{C.RESET}"
            score_str = f"{result['score']}/{result['max_score']}"
        
        print(f"  {test_info['name']:<30} {status:<15} {score_str}")
    
    print(f"\n  Total: {total_score}/{total_max} ({total_score/total_max*100:.1f}%)" if total_max > 0 else "")
    print(f"  Time: {elapsed:.1f}s")
    
    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "api_url": args.api_url,
            "total_score": total_score,
            "total_max": total_max,
            "elapsed_seconds": elapsed,
            "tests": results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
