#!/usr/bin/env python3
"""
Payment System Bug Test - Expert (4 bugs)
=========================================
Tests model's ability to find security and concurrency bugs.

Bugs:
1. Race condition - check-then-act on processing set is non-atomic
2. SQL injection - f-string used in UPDATE and SELECT queries
3. Float precision - using float for currency calculations
4. Transaction atomicity - no rollback on partial failure

Run: python expert_test.py [--api-url URL] [--model MODEL]
"""

import argparse
import json
import re
import requests

PROMPT = '''EXPERT PRACTICAL CODING TEST - PAYMENT SYSTEM BUGS

CONTEXT:
Our mining pool payment system has critical bugs causing:
1. Occasional duplicate payments to the same address
2. Security vulnerability reported by auditor
3. Small discrepancies in payment amounts (fractions of coins disappearing)
4. Payments sometimes fail silently

### payment_processor.py
```python
import sqlite3
import threading
import time
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)

class PaymentProcessor:
    def __init__(self, db_path: str, rpc_url: str):
        self.db_path = db_path
        self.rpc_url = rpc_url
        self.processing = set()  # Track payments being processed
        self.lock = threading.Lock()
        
    def process_pending_payments(self):
        """Process all pending payments."""
        pending = self._get_pending_payouts()
        
        for payout in pending:
            payout_id, address, amount = payout
            
            # Check if already processing
            if address in self.processing:
                continue
                
            self.processing.add(address)
            
            try:
                self._execute_payment(payout_id, address, amount)
            finally:
                self.processing.discard(address)
                
    def _get_pending_payouts(self) -> list:
        """Get list of pending payouts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(\'\'\'
            SELECT id, address, amount FROM payout_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC
        \'\'\')
        
        results = cursor.fetchall()
        conn.close()
        return results
        
    def _execute_payment(self, payout_id: int, address: str, amount: float):
        """Execute a single payment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Mark as processing
            cursor.execute(
                f"UPDATE payout_queue SET status = 'processing' WHERE id = {payout_id}"
            )
            conn.commit()
            
            # Send payment via RPC
            txid = self._send_payment(address, amount)
            
            if txid:
                # Mark as completed
                cursor.execute(\'\'\'
                    UPDATE payout_queue 
                    SET status = 'completed', txid = ?, completed_at = ?
                    WHERE id = ?
                \'\'\', (txid, time.time(), payout_id))
                conn.commit()
                logger.info(f"Payment {payout_id} completed: {txid}")
            else:
                # Mark as failed
                cursor.execute(\'\'\'
                    UPDATE payout_queue SET status = 'failed' WHERE id = ?
                \'\'\', (payout_id,))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Payment {payout_id} error: {e}")
            cursor.execute(\'\'\'
                UPDATE payout_queue SET status = 'pending' WHERE id = ?
            \'\'\', (payout_id,))
            conn.commit()
        finally:
            conn.close()
            
    def _send_payment(self, address: str, amount: float) -> Optional[str]:
        """Send payment via node RPC."""
        try:
            response = requests.post(self.rpc_url, json={
                'method': 'sendtoaddress',
                'params': [address, amount],
            }, timeout=30)
            
            data = response.json()
            if 'result' in data:
                return data['result']
            return None
        except Exception as e:
            logger.error(f"RPC error: {e}")
            return None
            
    def calculate_fee(self, amount: float, fee_percent: float) -> float:
        """Calculate pool fee."""
        fee = amount * (fee_percent / 100)
        return amount - fee
        
    def batch_payments(self, payouts: list) -> dict:
        """Aggregate small payments to reduce transaction fees."""
        aggregated = {}
        
        for address, amount in payouts:
            if address in aggregated:
                aggregated[address] += amount
            else:
                aggregated[address] = amount
                
        return aggregated

    def get_payment_history(self, address: str) -> list:
        """Get payment history for an address."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f\'\'\'
            SELECT * FROM payout_queue 
            WHERE address = '{address}' AND status = 'completed'
            ORDER BY completed_at DESC
        \'\'\')
        
        results = cursor.fetchall()
        conn.close()
        return results
```

TASK:
Find ALL critical bugs including security vulnerabilities.
For each bug provide:
1. Exact location
2. Root cause
3. Severity (Critical/High/Medium)
4. The fix
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
        "race_condition": False,
        "sql_injection": False,
        "float_precision": False,
        "atomicity": False,
    }
    
    # Bug 1: Race condition (check-then-act, TOCTOU)
    race_patterns = [
        r"race.*condition",
        r"toctou",
        r"time.*of.*check",
        r"check.*then.*act",
        r"processing.*set.*not.*atomic",
        r"not.*atomic",
        r"lock.*not.*used",
        r"concurrent",
    ]
    for pattern in race_patterns:
        if re.search(pattern, response_lower):
            checks["race_condition"] = True
            break
    
    # Bug 2: SQL injection
    sql_patterns = [
        r"sql.*inject",
        r"f-string.*sql",
        r"f\".*sql",
        r"f\'.*sql",
        r"format.*string.*sql",
        r"parameterized",
        r"sanitiz",
        r"escape.*sql",
    ]
    for pattern in sql_patterns:
        if re.search(pattern, response_lower):
            checks["sql_injection"] = True
            break
    
    # Also check for direct mentions of the vulnerable lines
    if "f\"update" in response_lower or "f'''select" in response_lower or "{payout_id}" in response_lower or "{address}" in response_lower:
        checks["sql_injection"] = True
    
    # Bug 3: Float precision
    float_patterns = [
        r"float.*precision",
        r"floating.*point",
        r"decimal.*instead",
        r"use.*decimal",
        r"rounding.*error",
        r"0\.1.*\+.*0\.2",
        r"currency.*float",
        r"money.*float",
    ]
    for pattern in float_patterns:
        if re.search(pattern, response_lower):
            checks["float_precision"] = True
            break
    
    # Bug 4: Transaction atomicity
    atomicity_patterns = [
        r"atomic",
        r"transaction",
        r"rollback",
        r"commit.*fail",
        r"partial.*fail",
        r"inconsistent.*state",
    ]
    for pattern in atomicity_patterns:
        if re.search(pattern, response_lower):
            checks["atomicity"] = True
            break
    
    score = sum(1 for v in checks.values() if v)
    return {
        "score": score,
        "max_score": 4,
        "passed": score >= 4,
        "checks": checks,
    }

def main():
    parser = argparse.ArgumentParser(description="Expert Payment Bug Test")
    parser.add_argument('--api-url', default='http://localhost:8000/v1')
    parser.add_argument('--model', default='seed-oss')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    
    print("=" * 60)
    print("PAYMENT SYSTEM BUG TEST (Expert - 4 bugs)")
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
        print(clean[:4000] + "..." if len(clean) > 4000 else clean)
        
        print("\n=== EVALUATION ===")
        result = evaluate(clean)
        
        for check, passed in result['checks'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        print(f"\nScore: {result['score']}/{result['max_score']}")
        if result['passed']:
            print("✓ PASSED - All bugs found!")
        elif result['score'] >= 3:
            print("△ GOOD - Found most bugs")
        else:
            print(f"✗ PARTIAL - Found {result['score']}/4 bugs")
            
        print(f"\nTokens: {data.get('usage', {})}")
        
        return result
        
    except Exception as e:
        print(f"ERROR: {e}")
        return {"score": 0, "max_score": 4, "passed": False}

if __name__ == "__main__":
    main()
