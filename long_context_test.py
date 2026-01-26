#!/usr/bin/env python3
"""
Long Context Test v1.0
======================
Tests model's ability to handle long contexts with retrieval and reasoning tasks.

Tests:
1. Needle in haystack (find specific info in long context)
2. Multi-file understanding (trace logic across files)
3. Cross-reference retrieval (answer questions requiring multiple facts)

Usage:
    python long_context_test.py --api-url http://localhost:8000/v1 --model seed-oss --context-size 64000
"""

import argparse
import json
import os
import random
import re
import string
import sys
import time
import requests
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Test Data Generation
# ============================================================================

def generate_haystack(target_size: int, needle: str, needle_position: float = 0.5) -> Tuple[str, int]:
    """
    Generate a haystack text with a needle hidden at specified position.
    
    Args:
        target_size: Target size in tokens (approximate)
        needle: The text to hide
        needle_position: Position as fraction (0.0 = start, 1.0 = end)
    
    Returns:
        (haystack_text, needle_char_position)
    """
    # Generate filler paragraphs (technical documentation style)
    topics = [
        "database optimization", "network protocols", "memory management",
        "file systems", "concurrency patterns", "API design", "caching strategies",
        "error handling", "logging best practices", "security considerations",
        "performance tuning", "scalability patterns", "testing methodologies",
        "deployment strategies", "monitoring solutions", "backup procedures",
    ]
    
    paragraphs = []
    current_size = 0
    target_chars = target_size * 4  # Approximate 4 chars per token
    
    while current_size < target_chars:
        topic = random.choice(topics)
        paragraph = generate_tech_paragraph(topic)
        paragraphs.append(paragraph)
        current_size += len(paragraph)
    
    # Insert needle at specified position
    total_paragraphs = len(paragraphs)
    insert_idx = int(total_paragraphs * needle_position)
    insert_idx = max(1, min(insert_idx, total_paragraphs - 1))
    
    # Wrap needle in distinct markers
    needle_wrapped = f"\n\n[IMPORTANT CONFIGURATION NOTE]\n{needle}\n[END NOTE]\n\n"
    paragraphs.insert(insert_idx, needle_wrapped)
    
    haystack = "\n\n".join(paragraphs)
    needle_pos = haystack.find(needle)
    
    return haystack, needle_pos


def generate_tech_paragraph(topic: str) -> str:
    """Generate a realistic technical documentation paragraph"""
    templates = [
        f"When implementing {topic}, developers should consider the trade-offs between complexity and maintainability. The standard approach involves analyzing system requirements and selecting appropriate algorithms based on expected load patterns. Performance benchmarks indicate that properly configured systems can achieve significant improvements in throughput while maintaining acceptable latency characteristics.",
        
        f"The {topic} subsystem requires careful attention to resource allocation. Memory usage patterns should be monitored continuously, and garbage collection strategies must be tuned according to workload characteristics. Connection pooling and request throttling can help prevent resource exhaustion under heavy load conditions.",
        
        f"Configuration of {topic} involves setting multiple parameters that interact in complex ways. The default values are suitable for development environments but should be adjusted for production deployments. Key metrics to monitor include response time distributions, error rates, and resource utilization percentages.",
        
        f"Best practices for {topic} include implementing proper error handling, maintaining detailed logs for debugging, and establishing clear boundaries between system components. Regular audits help identify potential issues before they impact production services.",
        
        f"The architecture of {topic} follows established patterns that prioritize reliability and observability. Event-driven designs enable loose coupling between components, while synchronous APIs provide predictable behavior for critical paths. Fallback mechanisms ensure graceful degradation during partial outages.",
    ]
    
    return random.choice(templates)


def generate_code_files() -> Dict[str, str]:
    """Generate realistic mining pool code files for cross-file testing"""
    
    files = {
        "thought_pool.py": '''#!/usr/bin/env python3
"""Thought Network Mining Pool - Main Stratum Server"""
import socket
import threading
import json
import sqlite3
from typing import Dict, Optional
from block_manager import BlockManager
from pplns import PPLNSManager

class StratumServer:
    def __init__(self, config: Dict):
        self.config = config
        self.port = config.get("stratum_port", 3333)
        self.db_path = config.get("db_path", "pool.db")
        self.block_manager = BlockManager(config)
        self.pplns = PPLNSManager(config)
        self.workers = {}
        self.current_job = None
        
    def start(self):
        """Start the stratum server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("0.0.0.0", self.port))
        self.server_socket.listen(100)
        print(f"Stratum server listening on port {self.port}")
        
    def handle_submit(self, worker_id: str, job_id: str, nonce: str):
        """Handle share submission from worker"""
        if not self.current_job or job_id != self.current_job["id"]:
            return {"error": "Stale job"}
        
        # Validate share
        is_valid, is_block = self.block_manager.validate_share(
            job_id, nonce, self.current_job
        )
        
        if is_valid:
            self.pplns.add_share(worker_id, difficulty=1.0)
            if is_block:
                self.block_manager.submit_block(self.current_job, nonce)
                self.pplns.process_block(self.current_job["height"])
            return {"result": True}
        return {"error": "Invalid share"}
''',
        
        "block_manager.py": '''#!/usr/bin/env python3
"""Block Manager - Handles block validation and submission"""
import hashlib
import struct
import requests
from typing import Dict, Tuple, Optional

# CRITICAL: Block hash byte order constant
REVERSE_HASH_BYTE_ORDER = True  # Bitcoin protocol requires reversed bytes

class BlockManager:
    def __init__(self, config: Dict):
        self.config = config
        self.rpc_url = config.get("rpc_url", "http://localhost:8332")
        self.rpc_user = config.get("rpc_user", "user")
        self.rpc_pass = config.get("rpc_pass", "pass")
        self.pending_blocks = {}
        
    def validate_share(self, job_id: str, nonce: str, job: Dict) -> Tuple[bool, bool]:
        """
        Validate a share submission.
        Returns (is_valid_share, is_valid_block)
        """
        # Build block header
        header = self._build_header(job, nonce)
        
        # Hash the header (double SHA256)
        hash1 = hashlib.sha256(header).digest()
        hash2 = hashlib.sha256(hash1).digest()
        
        # CRITICAL: Reverse byte order for comparison
        if REVERSE_HASH_BYTE_ORDER:
            block_hash = hash2[::-1]
        else:
            block_hash = hash2
        
        # Check against target
        target = bytes.fromhex(job["target"])
        is_valid = block_hash < target
        
        # Check if meets network difficulty
        network_target = bytes.fromhex(job["network_target"])
        is_block = block_hash < network_target
        
        return is_valid, is_block
        
    def _build_header(self, job: Dict, nonce: str) -> bytes:
        """Build 80-byte block header"""
        version = struct.pack("<I", job["version"])
        prev_hash = bytes.fromhex(job["prev_hash"])[::-1]  # Reverse for header
        merkle_root = bytes.fromhex(job["merkle_root"])[::-1]
        timestamp = struct.pack("<I", job["timestamp"])
        bits = bytes.fromhex(job["bits"])[::-1]
        nonce_bytes = struct.pack("<I", int(nonce, 16))
        
        return version + prev_hash + merkle_root + timestamp + bits + nonce_bytes
''',
        
        "pplns.py": '''#!/usr/bin/env python3
"""PPLNS Manager - Pay Per Last N Shares reward distribution"""
import sqlite3
import threading
from typing import Dict, List, Optional
from decimal import Decimal

# PPLNS Configuration
DEFAULT_N_MULTIPLIER = 2.0  # N = difficulty * multiplier
MIN_PAYOUT_THRESHOLD = Decimal("1.0")  # Minimum THT for payout

class PPLNSManager:
    def __init__(self, config: Dict):
        self.config = config
        self.db_path = config.get("db_path", "pool.db")
        self.n_multiplier = config.get("pplns_n_multiplier", DEFAULT_N_MULTIPLIER)
        self.min_payout = Decimal(str(config.get("min_payout", MIN_PAYOUT_THRESHOLD)))
        self.lock = threading.Lock()
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shares (
                id INTEGER PRIMARY KEY,
                worker_id TEXT NOT NULL,
                difficulty REAL NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS payouts (
                id INTEGER PRIMARY KEY,
                block_height INTEGER NOT NULL,
                worker_id TEXT NOT NULL,
                amount TEXT NOT NULL,
                status TEXT DEFAULT 'pending'
            )
        """)
        conn.commit()
        conn.close()
        
    def add_share(self, worker_id: str, difficulty: float):
        """Record a valid share"""
        import time
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO shares (worker_id, difficulty, timestamp) VALUES (?, ?, ?)",
                (worker_id, difficulty, time.time())
            )
            conn.commit()
            conn.close()
            
    def process_block(self, block_height: int, reward: Decimal = Decimal("78.5")):
        """Calculate and queue PPLNS payouts for a found block"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get last N shares
            n = int(self.config.get("network_difficulty", 1.0) * self.n_multiplier)
            cursor.execute(
                "SELECT worker_id, difficulty FROM shares ORDER BY timestamp DESC LIMIT ?",
                (n,)
            )
            shares = cursor.fetchall()
            
            if not shares:
                conn.close()
                return
            
            # Calculate total difficulty
            total_diff = sum(s[1] for s in shares)
            
            # Calculate payouts
            payouts = {}
            for worker_id, diff in shares:
                share_pct = Decimal(str(diff)) / Decimal(str(total_diff))
                amount = reward * share_pct
                if worker_id in payouts:
                    payouts[worker_id] += amount
                else:
                    payouts[worker_id] = amount
            
            # Queue payouts
            for worker_id, amount in payouts.items():
                if amount >= self.min_payout:
                    cursor.execute(
                        "INSERT INTO payouts (block_height, worker_id, amount) VALUES (?, ?, ?)",
                        (block_height, worker_id, str(amount))
                    )
            
            conn.commit()
            conn.close()
''',
    }
    
    return files


# ============================================================================
# Test Functions
# ============================================================================

def call_model(api_url: str, model_id: str, prompt: str, 
               max_tokens: int = 1000, temperature: float = 0.0) -> str:
    """Call the model API and return response"""
    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=300
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"ERROR: {response.status_code} - {response.text[:200]}"
            
    except Exception as e:
        return f"ERROR: {str(e)}"


def test_needle_in_haystack(api_url: str, model_id: str, context_size: int) -> Dict:
    """
    Test: Find a specific piece of information hidden in long context
    """
    # Generate unique needle
    secret_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    needle = f"The secret configuration code is: {secret_code}. This code must be used for authentication."
    
    # Generate haystack
    haystack, needle_pos = generate_haystack(context_size, needle, needle_position=0.5)
    
    prompt = f"""Below is a large technical documentation file. Read it carefully and answer the question at the end.

=== DOCUMENTATION START ===
{haystack}
=== DOCUMENTATION END ===

Question: What is the secret configuration code mentioned in the documentation?
Answer with ONLY the code, nothing else."""

    response = call_model(api_url, model_id, prompt)
    
    # Check if secret code is in response
    found = secret_code in response.upper()
    
    return {
        "test": "needle_in_haystack",
        "context_tokens": context_size,
        "needle_position": needle_pos,
        "expected": secret_code,
        "response": response[:200],
        "found": found,
        "status": "PASS" if found else "FAIL"
    }


def test_multi_file_understanding(api_url: str, model_id: str, context_size: int) -> Dict:
    """
    Test: Understand logic flow across multiple code files
    """
    files = generate_code_files()
    
    # Build context with all files
    context_parts = []
    for filename, content in files.items():
        context_parts.append(f"=== {filename} ===\n{content}")
    
    full_context = "\n\n".join(context_parts)
    
    # Trim if needed to fit context_size
    target_chars = context_size * 4
    if len(full_context) > target_chars:
        full_context = full_context[:target_chars]
    
    prompt = f"""Below is the source code for a cryptocurrency mining pool. Analyze the code and answer the question.

{full_context}

Question: Trace the flow when a miner submits a share that finds a valid block. What happens step by step from share submission to payout queuing? Be specific about which methods are called and in which files.

Answer concisely but accurately."""

    response = call_model(api_url, model_id, prompt, max_tokens=1500)
    
    # Check for key concepts that should be mentioned
    key_concepts = [
        "handle_submit",
        "validate_share",
        "submit_block",
        "process_block",
        "pplns",
        "payout",
    ]
    
    found_concepts = [c for c in key_concepts if c.lower() in response.lower()]
    score = len(found_concepts) / len(key_concepts)
    
    return {
        "test": "multi_file_understanding",
        "context_tokens": context_size,
        "expected_concepts": key_concepts,
        "found_concepts": found_concepts,
        "score": score,
        "response": response[:500],
        "status": "PASS" if score >= 0.6 else "FAIL"
    }


def test_cross_reference(api_url: str, model_id: str, context_size: int) -> Dict:
    """
    Test: Answer questions requiring information from multiple locations
    """
    files = generate_code_files()
    
    context_parts = []
    for filename, content in files.items():
        context_parts.append(f"=== {filename} ===\n{content}")
    
    full_context = "\n\n".join(context_parts)
    
    prompt = f"""Below is source code for a mining pool. Answer the questions based on the code.

{full_context}

Answer these questions:
1. What is the PPLNS N multiplier default value?
2. What is the minimum payout threshold in THT?
3. Is block hash byte order reversed for comparison? (yes/no)
4. What port does the stratum server listen on by default?

Format your answer as:
1. [value]
2. [value]
3. [yes/no]
4. [port]"""

    response = call_model(api_url, model_id, prompt)
    
    # Expected answers
    expected = {
        "n_multiplier": "2.0",
        "min_payout": "1.0",
        "byte_order": "yes",
        "port": "3333"
    }
    
    # Parse response
    correct = 0
    details = {}
    
    if "2.0" in response or "2" in response.split("\n")[0]:
        correct += 1
        details["n_multiplier"] = "correct"
    else:
        details["n_multiplier"] = "incorrect"
        
    if "1.0" in response or "1" in response:
        correct += 1
        details["min_payout"] = "correct"
    else:
        details["min_payout"] = "incorrect"
        
    lines = response.lower().split("\n")
    for line in lines:
        if "3." in line and "yes" in line:
            correct += 1
            details["byte_order"] = "correct"
            break
    else:
        details["byte_order"] = "incorrect"
        
    if "3333" in response:
        correct += 1
        details["port"] = "correct"
    else:
        details["port"] = "incorrect"
    
    score = correct / 4
    
    return {
        "test": "cross_reference",
        "context_tokens": context_size,
        "expected": expected,
        "details": details,
        "score": score,
        "correct": correct,
        "total": 4,
        "response": response[:500],
        "status": "PASS" if score >= 0.75 else "FAIL"
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Long context test for LLMs")
    parser.add_argument("--api-url", required=True, help="API endpoint URL")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--context-size", type=int, default=32000, 
                       help="Target context size in tokens")
    parser.add_argument("--test", choices=["needle", "multifile", "crossref", "all"],
                       default="all", help="Which test to run")
    parser.add_argument("--output", "-o", help="Output JSON file")
    
    args = parser.parse_args()
    
    print(f"Long Context Test")
    print(f"=" * 50)
    print(f"API: {args.api_url}")
    print(f"Model: {args.model}")
    print(f"Context Size: {args.context_size} tokens")
    print(f"=" * 50)
    
    results = []
    overall_pass = True
    
    if args.test in ["needle", "all"]:
        print("\n▶ Running needle-in-haystack test...")
        result = test_needle_in_haystack(args.api_url, args.model, args.context_size)
        results.append(result)
        print(f"  {result['status']}: Found={result['found']}")
        if result['status'] != "PASS":
            overall_pass = False
    
    if args.test in ["multifile", "all"]:
        print("\n▶ Running multi-file understanding test...")
        result = test_multi_file_understanding(args.api_url, args.model, args.context_size)
        results.append(result)
        print(f"  {result['status']}: Score={result['score']:.0%}")
        if result['status'] != "PASS":
            overall_pass = False
    
    if args.test in ["crossref", "all"]:
        print("\n▶ Running cross-reference test...")
        result = test_cross_reference(args.api_url, args.model, args.context_size)
        results.append(result)
        print(f"  {result['status']}: {result['correct']}/{result['total']} correct")
        if result['status'] != "PASS":
            overall_pass = False
    
    # Summary
    print(f"\n{'=' * 50}")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'=' * 50}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "context_size": args.context_size,
                "model": args.model,
                "overall": "PASS" if overall_pass else "FAIL",
                "tests": results
            }, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    # Exit code
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
