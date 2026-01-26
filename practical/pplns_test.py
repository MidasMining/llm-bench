#!/usr/bin/env python3
"""
PPLNS Mining Pool Bug Test - Hard (3 bugs)
==========================================
Tests model's ability to find configuration and logic bugs.

Bugs:
1. Hardcoded share_window (10000) ignores config pplns_window (50000)
2. window_size is calculated but never used in the query
3. Config key mismatch (pplns_window vs share_window naming)

Run: python pplns_test.py [--api-url URL] [--model MODEL]
"""

import argparse
import json
import re
import requests

PROMPT = '''PRACTICAL CODING TEST - PPLNS MINING POOL BUGS

CONTEXT:
Our PPLNS mining pool has several issues that are affecting miner payouts:
1. Miners complain their shares aren't being counted properly
2. Payouts are sometimes calculated incorrectly  
3. The share window seems wrong

### pplns.py
```python
import sqlite3
import time
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

class PPLNSManager:
    def __init__(self, config, db_path: str):
        self.config = config
        self.db_path = db_path
        self.n_factor = config.get('pplns_n', 2)  # N in PPLNS
        self.share_window = 10000  # Last N shares to consider
        
    def record_share(self, worker: str, difficulty: float):
        """Record a share from a worker."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(\'\'\'
            INSERT INTO shares (worker, difficulty, timestamp)
            VALUES (?, ?, ?)
        \'\'\', (worker, difficulty, time.time()))
        
        conn.commit()
        conn.close()
        
    def calculate_payouts(self, block_reward: Decimal) -> dict:
        """Calculate PPLNS payouts for a found block."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get network difficulty for window calculation
        net_diff = self.config.get('network_difficulty', 1000000)
        window_size = int(net_diff * self.n_factor)
        
        # Get recent shares within window
        cursor.execute(\'\'\'
            SELECT worker, SUM(difficulty) as total_diff
            FROM shares
            WHERE id > (SELECT MAX(id) - ? FROM shares)
            GROUP BY worker
        \'\'\', (self.share_window,))
        
        shares = cursor.fetchall()
        conn.close()
        
        if not shares:
            logger.warning("No shares found for payout calculation")
            return {}
        
        # Calculate total difficulty in window
        total_diff = sum(s[1] for s in shares)
        
        # Calculate each worker's payout
        payouts = {}
        for worker, worker_diff in shares:
            share_pct = Decimal(str(worker_diff)) / Decimal(str(total_diff))
            payout = block_reward * share_pct
            payouts[worker] = float(payout)
            
        logger.info(f"Calculated payouts for {len(payouts)} workers")
        return payouts
        
    def get_worker_stats(self, worker: str) -> dict:
        """Get statistics for a specific worker."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total shares
        cursor.execute(
            'SELECT COUNT(*), SUM(difficulty) FROM shares WHERE worker = ?',
            (worker,)
        )
        count, total_diff = cursor.fetchone()
        
        # Get shares in current window
        cursor.execute(\'\'\'
            SELECT COUNT(*), SUM(difficulty) FROM shares 
            WHERE worker = ? AND id > (SELECT MAX(id) - ? FROM shares)
        \'\'\', (worker, self.share_window))
        window_count, window_diff = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_shares': count or 0,
            'total_difficulty': total_diff or 0,
            'window_shares': window_count or 0,
            'window_difficulty': window_diff or 0,
        }

    def get_pool_hashrate(self) -> float:
        """Calculate pool hashrate from recent shares."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get shares from last 10 minutes
        ten_min_ago = time.time() - 600
        cursor.execute(\'\'\'
            SELECT SUM(difficulty), MIN(timestamp), MAX(timestamp)
            FROM shares WHERE timestamp > ?
        \'\'\', (ten_min_ago,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result or not result[0]:
            return 0.0
            
        total_diff, min_time, max_time = result
        time_span = max_time - min_time
        
        if time_span <= 0:
            return 0.0
            
        # Hashrate = difficulty / time (simplified)
        return total_diff / time_span
```

### pool_config.json
```json
{
    "pool_name": "ThoughtPool",
    "pplns_window": 50000,
    "pplns_n": 2,
    "network_difficulty": 5000000,
    "block_reward": 78.5,
    "stratum_port": 3333
}
```

TASK:
Find ALL bugs affecting share counting and payout calculations.
For each bug provide:
1. Exact location (file, line/method)
2. Root cause
3. The fix
'''

def strip_thinking(content: str) -> str:
    """Remove model thinking tags."""
    content = re.sub(r'<seed:think>.*?</seed:think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<seed:[^>]+>', '', content)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    return content.strip()

def evaluate(response: str) -> dict:
    """Evaluate response for bug detection."""
    response_lower = response.lower()
    
    checks = {
        "hardcoded_share_window": False,
        "window_size_unused": False,
        "config_key_mismatch": False,
    }
    
    # Bug 1: Hardcoded share_window (10000) ignores config
    hardcoded_patterns = [
        r"hardcoded.*10000",
        r"10000.*hardcoded",
        r"share_window.*10000",
        r"ignor.*config",
        r"not.*read.*config",
        r"should.*pplns_window",
    ]
    for pattern in hardcoded_patterns:
        if re.search(pattern, response_lower):
            checks["hardcoded_share_window"] = True
            break
    
    # Bug 2: window_size calculated but never used
    unused_patterns = [
        r"window_size.*not.*used",
        r"window_size.*never.*used",
        r"calculated.*never.*used",
        r"unused.*window_size",
        r"window_size.*calculated.*but",
        r"dead.*code",
    ]
    for pattern in unused_patterns:
        if re.search(pattern, response_lower):
            checks["window_size_unused"] = True
            break
    
    # Bug 3: Config key mismatch
    mismatch_patterns = [
        r"pplns_window.*mismatch",
        r"config.*key.*mismatch",
        r"pplns_window.*share_window",
        r"share_window.*pplns_window",
        r"wrong.*key",
        r"different.*key",
    ]
    for pattern in mismatch_patterns:
        if re.search(pattern, response_lower):
            checks["config_key_mismatch"] = True
            break
    
    score = sum(1 for v in checks.values() if v)
    return {
        "score": score,
        "max_score": 3,
        "passed": score >= 3,
        "checks": checks,
    }

def main():
    parser = argparse.ArgumentParser(description="PPLNS Bug Test")
    parser.add_argument('--api-url', default='http://localhost:8000/v1')
    parser.add_argument('--model', default='seed-oss')
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    
    print("=" * 60)
    print("PPLNS MINING POOL BUG TEST (Hard - 3 bugs)")
    print("=" * 60)
    
    try:
        r = requests.post(
            f"{args.api_url}/chat/completions",
            json={
                'model': args.model,
                'messages': [{'role': 'user', 'content': PROMPT}],
                'max_tokens': 6000,
                'temperature': 0.0
            },
            timeout=args.timeout
        )
        data = r.json()
        content = data['choices'][0]['message']['content']
        clean = strip_thinking(content)
        
        print("\n=== RESPONSE ===\n")
        print(clean[:3000] + "..." if len(clean) > 3000 else clean)
        
        print("\n=== EVALUATION ===")
        result = evaluate(clean)
        
        for check, passed in result['checks'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        print(f"\nScore: {result['score']}/{result['max_score']}")
        if result['passed']:
            print("✓ PASSED - All bugs found!")
        else:
            print(f"△ PARTIAL - Found {result['score']}/3 bugs")
            
        print(f"\nTokens: {data.get('usage', {})}")
        
        return result
        
    except Exception as e:
        print(f"ERROR: {e}")
        return {"score": 0, "max_score": 3, "passed": False}

if __name__ == "__main__":
    main()
